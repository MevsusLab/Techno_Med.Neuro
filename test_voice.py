import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication
from app import VoicePage
from voice_engine import analyze_voice, load_audio_file
from voice_capture import VoiceSession


def voice_signal(f0=180, sr=16000, noisy=False):
    t = np.arange(sr * 5) / sr
    y = 0.15 * np.sin(2 * np.pi * f0 * t) + 0.06 * np.sin(4 * np.pi * f0 * t)
    if noisy:
        y += np.random.default_rng(42).normal(0, 0.01, len(t))
    return y


class FakeVoiceWorker(QObject):
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.error = ''
        self.reason = 'deadline'
        self.start = Mock()
        self.requestInterruption = Mock()
        self.deleteLater = Mock()
        self.take_pending = Mock(return_value=(None, []))


class VoiceEngineTests(unittest.TestCase):
    def test_known_pitch_and_low_perturbation(self):
        for f0 in (100, 180, 300):
            result = analyze_voice(voice_signal(f0), 16000)
            self.assertAlmostEqual(result['f0'], f0, delta=1)
            self.assertLess(result['jitter_local'], 0.05)
            self.assertLess(result['shimmer_local'], 0.05)
            self.assertGreater(result['hnr'], 30)
            self.assertEqual(len(result['mfcc_mean']), 13)

    def test_noise_reduces_harmonicity(self):
        clean = analyze_voice(voice_signal(), 16000)
        noisy = analyze_voice(voice_signal(noisy=True), 16000)
        self.assertLess(noisy['hnr'], clean['hnr'])

    def test_silence_clipping_and_partial_recordings_rejected(self):
        cases = [np.zeros(80000), np.tile([1.0, -1.0], 40000), voice_signal()[:40000]]
        for audio in cases:
            with self.assertRaises(ValueError):
                analyze_voice(audio, 16000)

    def test_long_interruption_is_rejected(self):
        audio = voice_signal()
        audio[20000:50000] = 0
        with self.assertRaises(ValueError):
            analyze_voice(audio, 16000)

    def test_amplitude_modulation_raises_shimmer(self):
        t = np.arange(80000) / 16000
        steady = analyze_voice(voice_signal(), 16000)
        variable = analyze_voice(voice_signal() * (1 + 0.3 * np.sin(2 * np.pi * 10 * t)), 16000)
        self.assertGreater(variable['shimmer_local'], steady['shimmer_local'] + 1)

    def test_audio_file_selects_five_second_active_vowel_from_long_file(self):
        audio = np.concatenate([np.zeros(16000), voice_signal(), np.zeros(16000)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sustained_a.wav'
            sf.write(path, audio, 16000)
            segment, rate, metadata = load_audio_file(path)
        self.assertEqual(rate, 16000)
        self.assertAlmostEqual(len(segment) / rate, 5, places=3)
        self.assertAlmostEqual(metadata['segment_start_sec'], 1, delta=0.05)
        result = analyze_voice(segment, rate)
        self.assertAlmostEqual(result['f0'], 180, delta=1)

    def test_audio_file_too_short_or_silent_fails_with_specific_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            short = Path(directory) / 'short.wav'
            silence = Path(directory) / 'silence.wav'
            sf.write(short, voice_signal()[:32000], 16000)
            sf.write(silence, np.zeros(16000 * 8), 16000)
            with self.assertRaisesRegex(ValueError, 'only'):
                load_audio_file(short)
            with self.assertRaisesRegex(ValueError, 'continuous five-second'):
                load_audio_file(silence)


class VoiceCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])
        cls.result = analyze_voice(voice_signal(), 16000)

    def test_recording_reads_exactly_five_seconds_and_closes_stream(self):
        worker = VoiceSession()
        stream = Mock()
        stream.read.side_effect = lambda count: (np.ones((count, 1), dtype=np.float32) * 0.1, False)
        with patch('voice_capture.sd.query_devices', return_value={'default_samplerate': 16000}), patch('voice_capture.sd.check_input_settings'), patch('voice_capture.sd.InputStream', return_value=stream), patch.object(worker, 'msleep'), patch('voice_capture.analyze_voice', return_value=self.result) as analysis:
            worker.run()
        audio, rate = analysis.call_args[0]
        self.assertEqual(len(audio), 80000)
        self.assertEqual(rate, 16000)
        self.assertEqual(worker.reason, 'deadline')
        stream.stop.assert_called_once()
        stream.close.assert_called_once()

    def test_overflow_rejects_and_closes_stream(self):
        worker = VoiceSession()
        stream = Mock()
        stream.read.return_value = (np.zeros((1024, 1)), True)
        with patch('voice_capture.sd.query_devices', return_value={'default_samplerate': 16000}), patch('voice_capture.sd.check_input_settings'), patch('voice_capture.sd.InputStream', return_value=stream), patch.object(worker, 'msleep'):
            worker.run()
        self.assertEqual(worker.reason, 'error')
        self.assertIn('overflow', worker.error)
        stream.close.assert_called_once()

    def test_delayed_voice_starts_timer_at_sound_and_analyzes_full_vowel(self):
        worker = VoiceSession()
        stream = Mock()
        waveform = voice_signal()
        position = [0]
        silence = [12]
        def read(count):
            if silence[0]:
                silence[0] -= 1
                return np.zeros((count, 1), dtype=np.float32), False
            block = waveform[position[0]:position[0] + count]
            position[0] += count
            return block.astype(np.float32)[:, None], False
        stream.read.side_effect = read
        with patch('voice_capture.sd.query_devices', return_value={'default_samplerate': 16000}), patch('voice_capture.sd.check_input_settings'), patch('voice_capture.sd.InputStream', return_value=stream), patch.object(worker, 'msleep'):
            worker.run()
        self.assertEqual(worker.error, '')
        self.assertEqual(worker.reason, 'deadline')
        self.assertEqual(position[0], 80000)
        self.assertAlmostEqual(worker.latest['result']['f0'], 180, delta=1)
        stream.close.assert_called_once()

    def test_silent_microphone_times_out_without_analysis(self):
        worker = VoiceSession()
        stream = Mock()
        stream.read.side_effect = lambda count: (np.zeros((count, 1), dtype=np.float32), False)
        clock = [0.0]
        def now():
            clock[0] += 0.5
            return clock[0]
        with patch('voice_capture.sd.query_devices', return_value={'default_samplerate': 16000}), patch('voice_capture.sd.check_input_settings'), patch('voice_capture.sd.InputStream', return_value=stream), patch.object(worker, 'msleep'), patch('voice_capture.time.perf_counter', side_effect=now), patch('voice_capture.analyze_voice') as analysis:
            worker.run()
        self.assertIn('No sustained sound', worker.error)
        analysis.assert_not_called()
        stream.close.assert_called_once()

    def test_cancel_while_waiting_for_sound_closes_microphone(self):
        worker = VoiceSession()
        stream = Mock()
        interrupted = [False]
        def read(count):
            interrupted[0] = True
            return np.zeros((count, 1), dtype=np.float32), False
        stream.read.side_effect = read
        with patch('voice_capture.sd.query_devices', return_value={'default_samplerate': 16000}), patch('voice_capture.sd.check_input_settings'), patch('voice_capture.sd.InputStream', return_value=stream), patch.object(worker, 'msleep'), patch.object(worker, 'isInterruptionRequested', side_effect=lambda: interrupted[0]), patch('voice_capture.analyze_voice') as analysis:
            worker.run()
        self.assertEqual(worker.error, '')
        analysis.assert_not_called()
        stream.close.assert_called_once()

    def test_ui_complete_reports_and_reset(self):
        page = VoicePage()
        worker = FakeVoiceWorker()
        worker.take_pending.return_value = ({'state': 'COMPLETE', 'elapsed': 5, 'level': 0,
                                             'message': 'Acoustic analysis complete.', 'result': self.result}, [])
        page.worker = worker
        try:
            page._finished()
            self.assertEqual(page.badge.text(), 'COMPLETE')
            self.assertAlmostEqual(float(page.f0.val.text()), 180, delta=1)
            self.assertFalse(page.protocol_panel.isHidden())
            with tempfile.TemporaryDirectory() as directory:
                text, pdf = Path(directory) / 'voice.txt', Path(directory) / 'voice.pdf'
                with patch('protocol_panel.QFileDialog.getSaveFileName', return_value=(str(text), '')):
                    page.protocol_panel.save_button.click()
                self.assertIn('Джиттер RAP', text.read_text(encoding='utf-8'))
                with patch('protocol_panel.QFileDialog.getSaveFileName', return_value=(str(pdf), '')):
                    page.protocol_panel.export_button.click()
                self.assertTrue(pdf.read_bytes().startswith(b'%PDF-'))
            page.reset.click()
            self.assertEqual(page.f0.val.text(), '---')
            self.assertTrue(page.protocol_panel.isHidden())
        finally:
            page.cleanup()
            page.deleteLater()

    def test_uploaded_file_report_names_file_and_selected_segment(self):
        page = VoicePage()
        result = {**self.result, 'source': 'file', 'file_name': 'sample.wav',
                  'segment_start_sec': 2.5, 'source_duration_sec': 12.0}
        worker = FakeVoiceWorker()
        worker.take_pending.return_value = ({'state': 'COMPLETE', 'elapsed': 5, 'level': 0,
                                             'message': 'Complete', 'result': result}, [])
        page.worker = worker
        try:
            page._finished()
            with tempfile.TemporaryDirectory() as directory:
                text = Path(directory) / 'file_report.txt'
                with patch('protocol_panel.QFileDialog.getSaveFileName', return_value=(str(text), '')):
                    page.protocol_panel.save_button.click()
                report = text.read_text(encoding='utf-8')
                self.assertIn('sample.wav', report)
                self.assertIn('Исходный аудиофайл:', report)
                self.assertIn('2.50', report)
                self.assertIn('12.00', report)
        finally:
            page.cleanup()
            page.deleteLater()

    def test_file_upload_uses_file_worker_without_microphone_capture(self):
        page = VoicePage()
        worker = FakeVoiceWorker()
        with patch('app.check_voice_dependencies'), patch('app.QFileDialog.getOpenFileName', return_value=('C:/voice.wav', '')), patch('app.VoiceFileCaptureWorker', return_value=worker) as factory:
            page.upload.click()
        factory.assert_called_once_with('C:/voice.wav', parent=page)
        self.assertIs(page.worker, worker)
        self.assertFalse(page.upload.isEnabled())
        worker.take_pending.return_value = ({'state': 'LOADING', 'elapsed': 0, 'level': 0, 'message': 'Loading audio file'}, [])
        page._poll()
        self.assertIn('Загрузка аудиофайла', page.guidance.text())
        worker.take_pending.return_value = ({'state': 'COMPLETE', 'elapsed': 5, 'level': 0,
                                              'message': 'Complete', 'result': {**self.result, 'source': 'file', 'file_name': 'voice.wav', 'segment_start_sec': 1.0, 'source_duration_sec': 7.0}}, [])
        worker.error = ''
        worker.reason = 'deadline'
        page._finished()
        self.assertEqual(page.badge.text(), 'COMPLETE')
        self.assertIn('voice.wav', page.detail.text())
        self.assertTrue(page.upload.isEnabled())
        page.cleanup()
        page.deleteLater()

    def test_cancelled_session_does_not_publish_late_results(self):
        page = VoicePage()
        with patch('app.VoiceCaptureWorker', side_effect=FakeVoiceWorker):
            page.start.click()
            worker = page.worker
            page.reset.click()
            worker.take_pending.return_value = ({'state': 'COMPLETE', 'elapsed': 5, 'level': 0,
                                                 'message': 'Complete', 'result': self.result}, [])
            worker.finished.emit()
            self.assertIsNone(page.result)
            self.assertTrue(page.start.isEnabled())
            self.assertTrue(page.protocol_panel.isHidden())
        page.cleanup()
        page.deleteLater()


if __name__ == '__main__':
    unittest.main()
