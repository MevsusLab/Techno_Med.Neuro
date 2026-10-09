import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPixmap
from app import TremorPage
from protocol_panel import protocol_text


class ProtocolPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.page = TremorPage()
        self.page.resize(1200, 750)
        self.page.show()
        self.qt.processEvents()

    def tearDown(self):
        self.page.cleanup()
        self.page.close()
        self.page.deleteLater()
        self.qt.processEvents()

    def complete(self):
        t = np.arange(450) / 30
        for ti in t:
            self.page.engine.add_sample(float(10 * np.sin(2 * np.pi * 5 * ti)), 0.0, float(ti))
        self.page.record_start = 0
        self.page.record_end = 15
        self.page._run_analysis()
        self.qt.processEvents()

    def test_completion_splits_space_without_overlap_and_reset_restores_video(self):
        self.assertTrue(self.page.protocol_panel.isHidden())
        full_height = self.page.video_label.height()
        self.complete()
        video = self.page.video_label.geometry()
        panel = self.page.protocol_panel.geometry()
        self.assertFalse(self.page.protocol_panel.isHidden())
        self.assertEqual(video.width(), panel.width())
        self.assertLess(video.bottom(), panel.top())
        self.assertAlmostEqual(video.height() / (video.height() + panel.height()), 0.5, delta=0.02)
        self.assertLess(self.page.capture_area.geometry().bottom(), self.page.btn_start.geometry().top())
        self.page.btn_reset.click()
        self.qt.processEvents()
        self.assertTrue(self.page.protocol_panel.isHidden())
        self.assertGreater(self.page.video_label.height(), full_height * 0.95)

    def test_new_capture_hides_previous_protocol(self):
        self.complete()
        with patch('app.CaptureWorker') as factory:
            self.page.btn_start.click()
            self.page.timer.stop()
            self.assertTrue(self.page.protocol_panel.isHidden())
            self.assertEqual(self.page.protocol_panel.report, '')
            self.page.worker = None
        factory.assert_called_once()

    def test_invalid_analysis_does_not_show_protocol(self):
        self.page._run_analysis()
        self.assertTrue(self.page.protocol_panel.isHidden())

    def test_video_keeps_aspect_ratio_after_resize(self):
        image = QPixmap(640, 480)
        image.fill()
        self.page.video_label.setPixmap(image)
        self.complete()
        self.page.video_label.setPixmap(image)
        self.page.resize(1400, 850)
        self.qt.processEvents()
        shown = self.page.video_label.pixmap()
        self.assertAlmostEqual(shown.width() / shown.height(), 4 / 3, delta=0.01)
        self.assertLessEqual(shown.height(), self.page.video_label.height())
        self.assertLessEqual(shown.width(), self.page.video_label.width())

    def test_metric_branch_boundaries_are_non_diagnostic(self):
        for f, concentration, rms, title in [
            (5, 49.9, 0.5, 'LOW MEASURED'), (5, 80, 0.19, 'LOW MEASURED'),
            (None, 80, 0.5, 'NO CLEAR'), (3.99, 80, 0.5, 'LOW-FREQUENCY'),
            (4, 50, 0.2, '4.0-6.5'), (6.49, 80, 0.5, '4.0-6.5'),
            (6.5, 80, 0.5, '6.5-12.5'), (12.5, 80, 0.5, '6.5-12.5'),
        ]:
            with self.subTest(f=f):
                text = protocol_text({'dominant_freq': f, 'concentration': concentration, 'rms_px': rms, 'max_freq': 4.8})
                self.assertIn(title, text[0])
                self.assertIn('above this range were not assessed', text[3])
                self.assertNotIn('NORMAL', text[0])

    def test_text_and_pdf_exports_include_disclaimer_and_metrics(self):
        self.complete()
        with tempfile.TemporaryDirectory() as directory:
            text = Path(directory) / 'report.txt'
            pdf = Path(directory) / 'report.pdf'
            with patch('protocol_panel.QFileDialog.getSaveFileName', return_value=(str(text), '')):
                self.page.protocol_panel.save_button.click()
            report = text.read_text(encoding='utf-8')
            self.assertIn('СКЗ движения:', report)
            self.assertIn('ОГРАНИЧЕНИЕ:', report)
            with patch('protocol_panel.QFileDialog.getSaveFileName', return_value=(str(pdf), '')):
                self.page.protocol_panel.export_button.click()
            self.assertTrue(pdf.read_bytes().startswith(b'%PDF-'))


if __name__ == '__main__':
    unittest.main()
