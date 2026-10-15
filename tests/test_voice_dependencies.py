import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
import unittest
from unittest.mock import patch

from PyQt6.QtWidgets import QApplication
from app import VoicePage
from voice_capture import VoiceSession
from voice_dependencies import (check_voice_dependencies, load_voice_dependencies,
                                check_voice_file_dependencies, load_voice_file_dependencies)


class VoiceDependencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def test_missing_packages_error_targets_running_interpreter(self):
        with patch('voice_dependencies.importlib.util.find_spec', return_value=None):
            with self.assertRaises(RuntimeError) as error:
                check_voice_dependencies()
        message = str(error.exception)
        self.assertIn('librosa', message)
        self.assertIn('praat-parselmouth', message)
        self.assertIn(sys.executable, message)
        self.assertIn('-m pip install -r', message)
        self.assertIn('requirements.txt', message)

    def test_video_audio_dependencies_do_not_require_microphone_package(self):
        modules = []
        def find_spec(name):
            modules.append(name)
            return object()
        with patch('voice_dependencies.importlib.util.find_spec', side_effect=find_spec):
            check_voice_file_dependencies()
        self.assertEqual(set(modules), {'librosa', 'parselmouth'})
        self.assertNotIn('sounddevice', modules)

    def test_video_audio_dependency_load_does_not_import_sounddevice(self):
        imported = []
        def import_module(name):
            imported.append(name)
            if name == 'librosa':
                return type('Librosa', (), {'feature': type('Feature', (), {'rms': 1, 'mfcc': 1}),
                                             'yin': 1, 'resample': 1, 'stft': 1})
            if name in ('parselmouth', 'parselmouth.praat'):
                return type('Praat', (), {'Sound': 1})
            raise AssertionError('Unexpected module imported: ' + name)
        with patch('voice_dependencies.check_voice_file_dependencies'), patch(
                'voice_dependencies.importlib.import_module', side_effect=import_module):
            load_voice_file_dependencies()
        self.assertNotIn('sounddevice', imported)

    def test_broken_import_is_reported_as_dependency_failure(self):
        with patch('voice_dependencies.check_voice_dependencies'), patch('voice_dependencies.importlib.import_module', side_effect=ImportError('binary dependency failed')):
            with self.assertRaises(RuntimeError) as error:
                load_voice_dependencies()
        self.assertIn('binary dependency failed', str(error.exception))
        self.assertIn(sys.executable, str(error.exception))

    def test_missing_dependencies_do_not_start_microphone_or_timer(self):
        page = VoicePage()
        try:
            with patch('app.check_voice_dependencies', side_effect=RuntimeError('librosa missing')), patch('app.VoiceCaptureWorker') as worker:
                page.start.click()
            worker.assert_not_called()
            self.assertIsNone(page.worker)
            self.assertFalse(page.timer.isActive())
            self.assertTrue(page.start.isEnabled())
            self.assertFalse(page.stop.isEnabled())
            self.assertEqual(page.badge.text(), 'DEPENDENCY ERROR')
            self.assertIn('librosa', page.detail.text())
            self.assertEqual(page.time_label.text(), '0.0 / 5.0')
            self.assertTrue(page.protocol_panel.isHidden())
        finally:
            page.cleanup()
            page.deleteLater()

    def test_child_import_failure_happens_before_audio_device_access(self):
        worker = VoiceSession()
        with patch('voice_capture.load_voice_dependencies', side_effect=RuntimeError('librosa failed to import')), patch('voice_capture.sd.query_devices') as devices, patch('voice_capture.sd.InputStream') as stream:
            worker.run()
        devices.assert_not_called()
        stream.assert_not_called()
        self.assertEqual(worker.reason, 'error')
        self.assertIn('librosa failed to import', worker.error)


if __name__ == '__main__':
    unittest.main()
