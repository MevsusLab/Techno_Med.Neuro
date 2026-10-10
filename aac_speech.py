"""Serialized Russian TTS on one daemon thread (including Windows COM calls)."""

from collections import deque
import os
import queue
import threading
import time


def russian_voice(voices):
    for voice in voices:
        languages = []
        for language in getattr(voice, 'languages', ()) or ():
            if isinstance(language, bytes):
                language = language.decode('utf-8', errors='ignore')
            languages.append(str(language).lstrip(''.join(chr(i) for i in range(32))).lower().replace('_', '-'))
        if any(language == 'ru' or language.startswith('ru-')
               or language in ('419', '0419', '1049') for language in languages):
            return voice.id
        identity = f'{getattr(voice, "id", "")} {getattr(voice, "name", "")}'.lower()
        if any(token in identity for token in ('russian', 'ru-ru', 'ru_ru', 'русск', 'irina', 'pavel')):
            return voice.id
    raise RuntimeError('Не найден русский голос Windows. Установите русский речевой пакет для синтеза речи.')


def create_engine():
    import pyttsx3
    # A dedicated Engine avoids pyttsx3.init's global weak cache across threads.
    return pyttsx3.Engine(driverName='sapi5' if os.name == 'nt' else None)


class SpeechWorker:
    def __init__(self, engine_factory=create_engine):
        self.engine_factory = engine_factory
        self.condition = threading.Condition()
        self.pending = deque()
        self.errors = queue.SimpleQueue()
        self.thread = None
        self.closed = False
        self.generation = 0
        self.busy = False

    def speak(self, text):
        text = text.strip()
        if not text:
            return
        with self.condition:
            if self.closed:
                return
            if len(self.pending) >= 32:
                self.errors.put('Очередь озвучивания заполнена. Дождитесь окончания речи.')
                return
            self.pending.append((text, self.generation))
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._run, name='AAC Russian TTS', daemon=True)
                self.thread.start()
            self.condition.notify_all()

    def cancel(self):
        with self.condition:
            self.generation += 1
            self.pending.clear()
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.generation += 1
            self.pending.clear()
            self.condition.notify_all()

    def take_errors(self):
        messages = []
        while True:
            try:
                messages.append(self.errors.get_nowait())
            except queue.Empty:
                return messages

    def _run(self):
        engine = None
        com = None
        looping = False
        try:
            if os.name == 'nt':
                import pythoncom
                pythoncom.CoInitialize()
                com = pythoncom
            engine = self.engine_factory()
            engine.setProperty('voice', russian_voice(engine.getProperty('voices')))
            engine.connect('error', lambda name, exception: self.errors.put(f'Озвучивание не удалось: {exception}'))
            engine.startLoop(False)
            looping = True
            engine.iterate()
            while True:
                with self.condition:
                    while not self.pending and not self.closed:
                        self.condition.wait()
                    if self.closed:
                        break
                    text, generation = self.pending.popleft()
                    self.busy = True
                engine.say(text)
                cancel_started = None
                while True:
                    with self.condition:
                        cancelled = self.closed or generation != self.generation
                    if cancelled and cancel_started is None:
                        engine.stop()
                        cancel_started = time.perf_counter()
                    engine.iterate()
                    if not engine.isBusy():
                        break
                    if cancel_started is not None and time.perf_counter() - cancel_started > 0.5:
                        raise RuntimeError('Синтез речи не остановился.')
                    with self.condition:
                        self.condition.wait(0.01)
                with self.condition:
                    self.busy = False
        except Exception as exc:
            self.errors.put(str(exc) if 'русский голос' in str(exc) else f'Озвучивание недоступно: {exc}')
        finally:
            if engine is not None:
                try:
                    engine.stop()
                    if looping:
                        engine.endLoop()
                except Exception:
                    pass
                # Destroy the native driver before uninitializing this COM apartment.
                del engine
            if com is not None:
                com.CoUninitialize()
            with self.condition:
                self.busy = False
                self.pending.clear()
