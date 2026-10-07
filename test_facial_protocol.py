import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PyQt6.QtWidgets import QApplication
from app import FacialPage
from protocol_panel import facial_protocol_text
from test_face_capture import FakeFaceWorker
from test_facial_engine import symmetric_face


class FacialProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.page = FacialPage()
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

    def complete(self):
        samples = []
        bounds = {}
        for stage, start in [('rest', 0), ('smile', 5), ('brows', 10)]:
            points = symmetric_face()
            if stage == 'smile':
                points[[61, 78], 0] -= 15
                points[[291, 308], 0] += 15
            if stage == 'brows':
                points[[70, 105, 300, 334], 1] -= 15
            bounds[stage] = (start, start + 3)
            samples.extend({'stage': stage, 'timestamp': float(t), 'landmarks': points, 'pose': (0, 0, 0)} for t in np.arange(start, start + 3, 0.05))
        worker = FakeFaceWorker()
        worker.reason = 'deadline'
        worker.take_pending.return_value = (None, samples)
        self.page.worker = worker
        self.page.stage_bounds = bounds
        self.page.stage_counts = {stage: 60 for stage in bounds}
        self.page._finished()
        self.qt.processEvents()
        self.assertEqual(self.page.badge.text(), 'COMPLETE')

    def test_completed_split_and_close_restore_video(self):
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

    def test_reset_and_new_capture_clear_panel(self):
        self.complete()
        self.page.reset.click()
        self.assertTrue(self.page.protocol_panel.isHidden())
        self.assertEqual(self.page.protocol_panel.report, '')
        self.complete()
        with patch('app.FaceCaptureWorker', side_effect=FakeFaceWorker):
            self.page.start.click()
        self.page.timer.stop()
        self.assertTrue(self.page.protocol_panel.isHidden())
        self.assertEqual(self.page.protocol_panel.report, '')

    def test_rejected_capture_keeps_panel_hidden(self):
        worker = FakeFaceWorker()
        worker.error = 'No valid face frames'
        self.page.worker = worker
        with patch('app.sys.stderr'):
            self.page._finished()
        self.assertEqual(self.page.badge.text(), 'RETRY')
        self.assertTrue(self.page.protocol_panel.isHidden())

    def test_threshold_boundaries_prioritize_elevated_measurements(self):
        for asi, smile, brow, expected in [
            (2.11, 6.5, 2.3, 'LOW MEASURED'), (8, 6, 2, 'MODERATE'),
            (15, 20, 2, 'MODERATE'), (9, 21, 2, 'LOWER-FACE'),
            (16, 5, 12, 'SMILE AND BROW'), (2, 21, 11.99, 'LOWER-FACE'),
        ]:
            with self.subTest(asi=asi, smile=smile, brow=brow):
                status, detail, action, limitation = facial_protocol_text({'asymmetry_index': asi, 'smile_imbalance': smile, 'brow_imbalance': brow})
                self.assertIn(expected, status)
                self.assertNotIn('FAST POSITIVE', status)
                self.assertNotIn('NORMAL', status)
                self.assertIn('unvalidated', limitation)
                if 'LOW MEASURED' in status:
                    self.assertIn('negative FAST', detail)
                    self.assertIn('emergency', action)

    def test_exports_include_face_metrics_and_disclaimer(self):
        self.complete()
        panel = self.page.protocol_panel
        with tempfile.TemporaryDirectory() as directory:
            text = Path(directory) / 'face.txt'
            pdf = Path(directory) / 'face.pdf'
            with patch('protocol_panel.QFileDialog.getSaveFileName', return_value=(str(text), '')):
                panel.save_button.click()
            contents = text.read_text(encoding='utf-8')
            self.assertIn('FACIAL SYMMETRY', contents)
            self.assertIn('smile imbalance', contents)
            self.assertIn('DISCLAIMER: Computational screening utility.', contents)
            with patch('protocol_panel.QFileDialog.getSaveFileName', return_value=(str(pdf), '')):
                panel.export_button.click()
            self.assertTrue(pdf.read_bytes().startswith(b'%PDF-'))


if __name__ == '__main__':
    unittest.main()
