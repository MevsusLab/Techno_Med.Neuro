import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication
from app import RPPGPage, MainWindow
from rppg_capture import extract_regions, RPPGSession, REGIONS
from test_rppg_engine import pulse_samples


def roi_face():
    points = np.zeros((478, 3))
    points[:, :2] = [320, 240]
    points[33, :2], points[263, :2] = [230, 220], [410, 220]
    for indices, center in zip(REGIONS, ([320, 140], [260, 280], [380, 280])):
        angles = np.linspace(0, 2 * np.pi, len(indices), endpoint=False)
        for index, angle in zip(indices, angles):
            points[index, :2] = np.array(center) + [25 * np.cos(angle), 20 * np.sin(angle)]
    return points


class FakeRPPGWorker(QObject):
    finished = pyqtSignal()
    failed = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.start = Mock()
        self.requestInterruption = Mock()
        self.deleteLater = Mock()
        self.take_pending = Mock(return_value=(None, []))
        self.error = ''
        self.reason = 'deadline'
        self.snapshot = Mock(return_value=(0, 30, None, 2))


class RPPGCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def test_roi_colors_are_measured_before_overlay(self):
        frame = np.full((480, 640, 3), [80, 110, 130], dtype=np.uint8)
        colors, polygons, reference, center, span = extract_regions(frame, roi_face(), (0, 0, 0))
        self.assertTrue(np.allclose(colors, [[130, 110, 80]] * 3))
        self.assertEqual(reference, 110)
        self.assertEqual(len(polygons), 3)
        self.assertGreater(span, 90)

    def test_pose_and_exposure_fail_capture_quality(self):
        with self.assertRaises(ValueError):
            extract_regions(np.full((480, 640, 3), 100, dtype=np.uint8), roi_face(), (20, 0, 0))
        with self.assertRaises(ValueError):
            extract_regions(np.zeros((480, 640, 3), dtype=np.uint8), roi_face(), (0, 0, 0))

    def test_full_session_records_regions_and_releases_resources(self):
        worker = RPPGSession(duration=30)
        camera, model = Mock(), Mock()
        points = roi_face()
        landmarks = [SimpleNamespace(x=p[0] / 640, y=p[1] / 480, z=0) for p in points]
        model.detect_for_video.return_value = SimpleNamespace(face_landmarks=[landmarks], facial_transformation_matrixes=[np.eye(4)])
        now = [100.0]
        def read():
            now[0] += 1 / 30
            return True, np.full((480, 640, 3), [80, 110, 130], dtype=np.uint8)
        camera.read.side_effect = read
        with patch('rppg_capture.ensure_face_model'), patch('rppg_capture.cv2.VideoCapture', return_value=camera), patch('rppg_capture.FaceLandmarker.create_from_options', return_value=model), patch('rppg_capture.time.perf_counter', side_effect=lambda: now[0]):
            worker.run()
        self.assertEqual(worker.error, '')
        self.assertEqual(worker.reason, 'deadline')
        self.assertGreater(len(worker.samples), 850)
        self.assertTrue(all(sample['timestamp'] >= worker.record_start for sample in worker.samples))
        camera.release.assert_called_once()
        model.close.assert_called_once()

    def test_real_engine_results_render_and_reset(self):
        page = RPPGPage()
        worker = FakeRPPGWorker()
        samples = pulse_samples()
        worker.take_pending.return_value = ({'image': np.zeros((480, 640, 3), dtype=np.uint8), 'frames': 900, 'start': 0, 'progress': 1, 'message': 'Complete'}, samples)
        page.worker = worker
        try:
            page._finished()
            self.assertEqual(page.badge.text(), 'COMPLETE')
            self.assertEqual(page.bpm.val.text(), '72')
            self.assertTrue(page.start.isEnabled())
            page.reset.click()
            self.assertEqual(page.bpm.val.text(), '---')
        finally:
            page.cleanup()
            page.deleteLater()

    def test_reset_during_capture_discards_samples(self):
        page = RPPGPage()
        with patch('app.RPPGCaptureWorker', side_effect=FakeRPPGWorker):
            page.start.click()
            worker = page.worker
            page.reset.click()
            worker.take_pending.return_value = (None, pulse_samples())
            worker.finished.emit()
            self.assertEqual(page.engine.samples, [])
            self.assertEqual(page.bpm.val.text(), '---')
            self.assertTrue(page.start.isEnabled())
        page.cleanup()
        page.deleteLater()

    def test_failed_capture_is_retryable(self):
        page = RPPGPage()
        worker = FakeRPPGWorker()
        worker.error = 'Camera disconnected'
        page.worker = worker
        page._finished()
        self.assertEqual(page.badge.text(), 'RETRY')
        self.assertEqual(page.bpm.val.text(), '---')
        self.assertIsNone(page.worker)
        page.cleanup()
        page.deleteLater()

    def test_navigation_does_not_start_second_camera_module(self):
        window = MainWindow()
        with patch('app.RPPGCaptureWorker', side_effect=FakeRPPGWorker):
            window.switch_page(3)
            window.rppg_page.start.click()
            worker = window.rppg_page.worker
            window.switch_page(1)
            self.assertEqual(window.stack.currentIndex(), 3)
            window.switch_page(2)
            self.assertEqual(window.stack.currentIndex(), 3)
            event = QCloseEvent()
            window.closeEvent(event)
            self.assertFalse(event.isAccepted())
            worker.requestInterruption.assert_called()
            worker.finished.emit()
            self.qt.processEvents()
        window.close()
        window.deleteLater()

    def test_repeated_estimates_do_not_reuse_previous_samples(self):
        page = RPPGPage()
        with patch('app.RPPGCaptureWorker', side_effect=FakeRPPGWorker):
            for bpm in (72, 90, 120):
                page.start.click()
                worker = page.worker
                worker.take_pending.return_value = ({'image': np.zeros((480, 640, 3), dtype=np.uint8), 'frames': 900,
                                                     'start': 0, 'progress': 1, 'message': 'Complete'}, pulse_samples(bpm))
                worker.finished.emit()
                self.assertEqual(page.bpm.val.text(), str(bpm))
                self.assertEqual(len(page.engine.samples), 900)
                self.assertTrue(page.start.isEnabled())
        page.cleanup()
        page.deleteLater()

    def test_slow_drift_displays_bpm_and_quality_caution(self):
        page = RPPGPage()
        samples = pulse_samples()
        for sample in samples:
            sample['geometry'][0] += 0.18 * sample['timestamp'] / 30
        worker = FakeRPPGWorker()
        worker.take_pending.return_value = ({'image': np.zeros((480, 640, 3), dtype=np.uint8), 'frames': 900,
                                             'start': 0, 'progress': 1, 'message': 'Complete'}, samples)
        page.worker = worker
        try:
            page._finished()
            self.assertEqual(page.bpm.val.text(), '72')
            self.assertEqual(page.badge.text(), 'CHECK QUALITY')
            self.assertEqual(page.analysis_badge.text(), 'ESTIMATE / CAUTION')
            self.assertIn('медленное смещение головы', page.detail.text())
            self.assertTrue(page.start.isEnabled())
            page.reset.click()
            self.assertEqual(page.bpm.val.text(), '---')
            self.assertEqual(page.analysis_badge.text(), 'WAITING')
        finally:
            page.cleanup()
            page.deleteLater()


if __name__ == '__main__':
    unittest.main()
