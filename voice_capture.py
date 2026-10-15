import os
import threading
import time

import numpy as np
try:
    import sounddevice as sd
except ImportError:
    sd = None
from PyQt6.QtCore import QCoreApplication, QThread
from capture_worker import ProcessCaptureWorker
from voice_engine import analyze_voice, load_audio_file
from voice_dependencies import load_voice_dependencies


class VoiceSession(QThread):
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.latest = None
        self.phase = 'voice dependencies'
        self.phase_started = time.perf_counter()
        self.error = ''
        self.reason = 'stopped'

    def phase_change(self, phase):
        with self.lock:
            self.phase, self.phase_started = phase, time.perf_counter()

    def publish(self, **values):
        with self.lock:
            self.latest = values

    def run(self):
        stream = None
        try:
            if self.isInterruptionRequested():
                return
            self.phase_change('voice dependencies')
            self.publish(state='SETUP', elapsed=0, level=0,
                         message='Проверка и загрузка библиотек анализа голоса. Микрофон пока не записывает; дождитесь подсказки.')
            load_voice_dependencies()
            if self.isInterruptionRequested():
                return
            self.phase_change('microphone opening')
            info = sd.query_devices(kind='input')
            sample_rate = int(round(info['default_samplerate']))
            if not 16000 <= sample_rate <= 96000:
                raise ValueError('Microphone default sample rate must be between 16 and 96 kHz.')
            sd.check_input_settings(channels=1, dtype='float32', samplerate=sample_rate)
            self.publish(state='PREPARING', elapsed=0, level=0, message='Подготовьтесь. Когда начнётся ЗАПИСЬ, протяните звук «А» пять секунд.')
            for _ in range(20):
                if self.isInterruptionRequested():
                    return
                self.msleep(100)
            self.phase_change('microphone opening')
            stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', blocksize=1024)
            stream.start()
            blocks = []
            count = 0
            target = sample_rate * 5
            onset_blocks = []
            onset_samples = 0
            wait_started = time.perf_counter()
            self.publish(state='WAITING_VOICE', elapsed=0, level=0,
                         message='Начните тянуть звук «А». Таймер пяти секунд запустится после обнаружения звука. Держите громкость и расстояние до микрофона постоянными.')
            while not self.isInterruptionRequested():
                self.phase_change('microphone read')
                block, overflow = stream.read(1024)
                if self.isInterruptionRequested():
                    return
                if overflow:
                    raise OSError('Audio input overflowed; close CPU-heavy applications and retry.')
                values = np.asarray(block[:, 0], dtype=np.float32).copy()
                if len(values) == 0:
                    raise OSError('Microphone did not deliver audio.')
                if float(np.sqrt(np.mean(values ** 2))) >= 0.005:
                    onset_blocks.append(values)
                    onset_samples += len(values)
                else:
                    onset_blocks = []
                    onset_samples = 0
                self.publish(state='WAITING_VOICE', elapsed=0, level=float(np.max(np.abs(values))),
                             message='Начните тянуть звук «А». Таймер пяти секунд запустится после обнаружения звука. Держите громкость и расстояние до микрофона постоянными.')
                if onset_samples >= int(sample_rate * 0.08):
                    blocks = onset_blocks
                    count = onset_samples
                    break
                if time.perf_counter() - wait_started > 10:
                    raise ValueError('No sustained sound was detected within ten seconds. Check the selected system microphone and input level.')
            while count < target and not self.isInterruptionRequested():
                self.phase_change('microphone read')
                block, overflow = stream.read(min(1024, target - count))
                if overflow:
                    raise OSError('Audio input overflowed; close CPU-heavy applications and retry.')
                if len(block) == 0:
                    raise OSError('Microphone did not deliver audio.')
                values = block[:, 0].copy()
                blocks.append(values)
                count += len(values)
                self.publish(state='RECORDING', elapsed=count / sample_rate,
                             level=float(np.max(np.abs(values))), message='Непрерывно тяните звук «А». Сохраняйте громкость и расстояние до микрофона.')
            self.phase_change('resource close')
            stream.stop()
            stream.close()
            stream = None
            if self.isInterruptionRequested():
                return
            self.phase_change('voice analysis')
            self.publish(state='PROCESSING', elapsed=5, level=0, message='Анализ высоты голоса, периодов, амплитуды и гармоничности. Первый запуск может занять больше времени.')
            result = analyze_voice(np.concatenate(blocks), sample_rate)
            if self.isInterruptionRequested():
                return
            self.publish(state='COMPLETE', elapsed=5, level=0, message='Акустический анализ завершён.', result=result)
            self.reason = 'deadline'
        except Exception as exc:
            self.error, self.reason = f'{type(exc).__name__}: {exc}', 'error'
        finally:
            if stream is not None:
                self.phase_change('resource close')
                try:
                    stream.close()
                except Exception as exc:
                    if not self.error:
                        self.error, self.reason = str(exc), 'error'


def voice_process(connection, stop, duration, preparation):
    qt = QCoreApplication([])
    worker = VoiceSession()
    try:
        worker.start()
        while True:
            if stop.is_set():
                worker.requestInterruption()
            done = worker.wait(0)
            with worker.lock:
                latest = worker.latest
                worker.latest = None
                phase, started = worker.phase, worker.phase_started
            connection.send((latest, [], (None, None, None, 3), phase, started, [], done,
                             worker.error if done else '', worker.reason))
            if done:
                break
            time.sleep(0.03)
    except (BrokenPipeError, EOFError, OSError):
        worker.requestInterruption()
        worker.wait(500)
    finally:
        connection.close()
    if worker.isRunning():
        os._exit(1)


class VoiceCaptureWorker(ProcessCaptureWorker):
    def __init__(self, parent=None):
        super().__init__(duration=5, parent=parent, process_target=voice_process)


class VoiceFileSession(QThread):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.lock = threading.Lock()
        self.latest = None
        self.phase = 'voice dependencies'
        self.phase_started = time.perf_counter()
        self.error = ''
        self.reason = 'stopped'

    def phase_change(self, phase):
        with self.lock:
            self.phase, self.phase_started = phase, time.perf_counter()

    def run(self):
        try:
            if self.isInterruptionRequested():
                return
            self.publish(state='LOADING', elapsed=0, level=0, message='Проверка аудиофайла и подготовка анализа.')
            self.phase_change('voice dependencies')
            from voice_dependencies import load_voice_dependencies
            load_voice_dependencies()
            if self.isInterruptionRequested():
                return
            self.phase_change('audio file loading')
            audio, sample_rate, metadata = load_audio_file(self.path)
            if self.isInterruptionRequested():
                return
            self.phase_change('voice analysis')
            self.publish(state='PROCESSING', elapsed=5, level=0, message='Анализ выбранного пятисекундного фрагмента.')
            result = analyze_voice(audio, sample_rate)
            result.update(metadata)
            self.publish(state='COMPLETE', elapsed=5, level=0, message='Акустический анализ файла завершён.', result=result)
            self.reason = 'deadline'
        except Exception as exc:
            self.error, self.reason = f'{type(exc).__name__}: {exc}', 'error'

    def publish(self, **values):
        with self.lock:
            self.latest = values


def voice_file_process(connection, stop, duration, preparation, path):
    qt = QCoreApplication([])
    worker = VoiceFileSession(path)
    try:
        worker.start()
        while True:
            if stop.is_set():
                worker.requestInterruption()
            done = worker.wait(0)
            with worker.lock:
                latest = worker.latest
                worker.latest = None
                phase, started = worker.phase, worker.phase_started
            connection.send((latest, [], (None, None, None, 3), phase, started, [], done,
                             worker.error if done else '', worker.reason))
            if done:
                break
            time.sleep(0.03)
    except (BrokenPipeError, EOFError, OSError):
        worker.requestInterruption()
        worker.wait(500)
    finally:
        connection.close()
    if worker.isRunning():
        os._exit(1)


class VoiceFileCaptureWorker(ProcessCaptureWorker):
    def __init__(self, path, parent=None):
        super().__init__(duration=5, parent=parent, process_target=voice_file_process, process_args=(str(path),))
