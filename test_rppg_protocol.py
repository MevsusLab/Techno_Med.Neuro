import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PyQt6.QtWidgets import QApplication
from app import RPPGPage
from protocol_panel import rppg_protocol_text
from test_rppg_capture import FakeRPPGWorker
from test_rppg_engine import pulse_samples


class RPPGProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.page = RPPGPage()
        self.page.resize(1400, 860)
        self.page.show()
        self.qt.processEvents()

    def tearDown(self):
        self.page.cleanup()
        if self.page.worker is not None:
            self.page.worker.finished.emit()
        self.page.close()
        self.page.deleteLater()
        self.qt.processEvents()

    def complete(self, drift=False):
        samples = pulse_samples()
        if drift:
            for sample in samples:
                sample['geometry'][0] += 0.18 * sample['timestamp'] / 30
        worker = FakeRPPGWorker()
        worker.take_pending.return_value = ({'image': np.zeros((480, 640, 3), dtype=np.uint8),
                                             'frames': 900, 'start': 0, 'progress': 1, 'message': 'Complete'}, samples)
        self.page.worker = worker
        self.page._finished()
        self.qt.processEvents()

    def test_completed_split_and_close_restore_full_video(self):
        full = self.page.video.height()
        self.assertTrue(self.page.protocol_panel.isHidden())
        self.complete()
        panel = self.page.protocol_panel
        self.assertFalse(panel.isHidden())
        self.assertAlmostEqual(self.page.video.height() / (self.page.video.height() + panel.height()), 0.5, delta=0.05)
        self.assertLess(self.page.video.geometry().bottom(), panel.geometry().top())
        self.assertLess(self.page.capture_area.geometry().bottom(), self.page.start.geometry().top())
        panel.close_button.click()
        self.qt.processEvents()
        self.assertTrue(panel.isHidden())
        self.assertGreater(self.page.video.height(), full * 0.95)

    def test_caution_estimate_shows_protocol_without_hiding_warning(self):
        self.complete(drift=True)
        self.assertEqual(self.page.badge.text(), 'CHECK QUALITY')
        self.assertFalse(self.page.protocol_panel.isHidden())
        self.assertIn('медленное смещение головы', self.page.protocol_panel.limitation_label.text())

    def test_reset_new_capture_and_rejection_keep_panel_hidden(self):
        self.complete()
        self.page.reset.click()
        self.assertTrue(self.page.protocol_panel.isHidden())
        self.assertEqual(self.page.protocol_panel.report, '')
        self.complete()
        with patch('app.RPPGCaptureWorker', side_effect=FakeRPPGWorker):
            self.page.start.click()
        self.page.timer.stop()
        self.assertTrue(self.page.protocol_panel.isHidden())
        worker = self.page.worker
        worker.error = 'Capture failed'
        worker.finished.emit()
        self.assertEqual(self.page.badge.text(), 'RETRY')
        self.assertTrue(self.page.protocol_panel.isHidden())

    def test_rate_and_quality_threshold_boundaries(self):
        for bpm, concentration, coverage, spread, expected in [
            (67, 82.6, 0.91, 3, 'В ДИАПАЗОНЕ 60–95'), (59, 80, 0.9, 2, 'НИЖЕ 60'),
            (60, 60, 0.7, 5, 'В ДИАПАЗОНЕ 60–95'), (95, 80, 0.9, 2, 'В ДИАПАЗОНЕ 60–95'),
            (95.1, 80, 0.9, 2, 'ВЫШЕ 95'), (67, 59.9, 0.9, 2, 'НЕНАДЁЖНО'),
            (67, 80, 0.699, 2, 'НЕНАДЁЖНО'), (67, 80, 0.9, 5.1, 'НЕНАДЁЖНО'),
        ]:
            with self.subTest(bpm=bpm, concentration=concentration, coverage=coverage, spread=spread):
                result = {'bpm': bpm, 'concentration': concentration, 'coverage': coverage,
                          'roi_bpm': [bpm, bpm, bpm + spread], 'warnings': []}
                title, detail, actions, limits = rppg_protocol_text(result)
                self.assertIn(expected, title)
                self.assertNotIn('[NORMAL]', title)
                self.assertNotIn('DETECTED', title)
                self.assertIn('исследовательские', limits)
                self.assertIn('срочно обратитесь за медицинской помощью', actions)

    def test_text_and_pdf_exports_include_quality_and_interpretation(self):
        self.complete(drift=True)
        panel = self.page.protocol_panel
        with tempfile.TemporaryDirectory() as directory:
            text = Path(directory) / 'rppg.txt'
            pdf = Path(directory) / 'rppg.pdf'
            with patch('protocol_panel.QFileDialog.getSaveFileName', return_value=(str(text), '')):
                panel.save_button.click()
            report = text.read_text(encoding='utf-8')
            self.assertIn('Предварительная оценка пульса:', report)
            self.assertIn('Разброс всех ROI:', report)
            self.assertIn('медленное смещение головы', report)
            self.assertNotIn('РЕКОМЕНДАЦИИ', report)
            self.assertNotIn('ОГРАНИЧЕНИЕ:', report)
            self.assertTrue(report.rstrip().endswith(panel.action_label.text()))
            with patch('protocol_panel.QFileDialog.getSaveFileName', return_value=(str(pdf), '')):
                panel.export_button.click()
            self.assertTrue(pdf.read_bytes().startswith(b'%PDF-'))


if __name__ == '__main__':
    unittest.main()
