import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch

import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from face_capture import FaceSession, head_pose, ensure_face_model
from test_facial_engine import symmetric_face
from app import FacialPage, MainWindow


class FakeFaceWorker(QObject):
    finished = pyqtSignal()
    failed = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.error = ''
        self.reason = 'stopped'
        self.start = Mock()
        self.requestInterruption = Mock()
        self.take_pending = Mock(return_value=(None, []))
        self.deleteLater = Mock()


class FaceCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def test_pose_identity_and_yaw(self):
        self.assertTrue(np.allclose(head_pose(np.eye(4)), (0, 0, 0)))
        rotation = np.eye(4)
        angle = np.deg2rad(20)
        rotation[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        self.assertAlmostEqual(head_pose(rotation)[0], 20)

    def test_guided_session_captures_each_stage_and_cleans_resources(self):
        worker = FaceSession()
        camera = Mock()
        detector = Mock()
        now = [100.0]
        points = symmetric_face()
        landmarks = [SimpleNamespace(x=p[0] / 640, y=p[1] / 480, z=0) for p in points]
        detector.detect_for_video.return_value = SimpleNamespace(face_landmarks=[landmarks], facial_transformation_matrixes=[np.eye(4)])
        frame = np.random.default_rng(5).integers(50, 200, (480, 640, 3), dtype=np.uint8)
        def read():
            now[0] += 0.05
            return True, frame.copy()
        camera.read.side_effect = read
        with patch('face_capture.ensure_face_model'), patch('face_capture.cv2.VideoCapture', return_value=camera), patch('face_capture.FaceLandmarker.create_from_options', return_value=detector), patch('face_capture.time.perf_counter', side_effect=lambda: now[0]):
            worker.run()
        self.assertEqual(worker.error, '')
        self.assertEqual(worker.reason, 'deadline')
        for stage in ('rest', 'smile', 'brows'):
            samples = [s for s in worker.samples if s['stage'] == stage]
            self.assertGreater(len(samples), 40)
            start, end = worker.bounds[stage]
            self.assertTrue(all(start <= s['timestamp'] <= end for s in samples))
        camera.release.assert_called_once()
        detector.close.assert_called_once()

    def test_multiple_faces_fail_preparation_without_measurements(self):
        worker = FaceSession()
        camera = Mock()
        detector = Mock()
        detector.detect_for_video.return_value = SimpleNamespace(face_landmarks=[[], []], facial_transformation_matrixes=[])
        now = [100.0]
        def read():
            now[0] += 0.2
            return True, np.zeros((480, 640, 3), dtype=np.uint8)
        camera.read.side_effect = read
        with patch('face_capture.ensure_face_model'), patch('face_capture.cv2.VideoCapture', return_value=camera), patch('face_capture.FaceLandmarker.create_from_options', return_value=detector), patch('face_capture.time.perf_counter', side_effect=lambda: now[0]):
            worker.run()
        self.assertIn('Preparation failed', worker.error)
        self.assertEqual(worker.samples, [])
        camera.release.assert_called_once()

    def test_ui_complete_and_reset_with_real_metrics(self):
        page = FacialPage()
        worker = Mock()
        worker.error = ''
        worker.reason = 'deadline'
        samples = []
        bounds = {}
        for stage, start in [('rest', 0), ('smile', 5), ('brows', 10)]:
            points = symmetric_face()
            if stage == 'smile':
                points[[61, 78], 0] -= 15
                points[[291, 308], 0] += 15
                points[[61, 78, 291, 308], 1] -= 10
            if stage == 'brows':
                points[[70, 105, 300, 334], 1] -= 15
            for ti in np.arange(start, start + 3, 0.05):
                samples.append({'stage': stage, 'timestamp': float(ti), 'landmarks': points, 'pose': (0, 0, 0)})
            bounds[stage] = (start, start + 3)
        worker.take_pending.return_value = ({'stage_bounds': bounds, 'counts': {s: 60 for s in bounds}, 'rejected': {s: 0 for s in bounds}, 'message': 'Complete', 'stage': 'brows', 'progress': 1, 'image': np.zeros((480, 640, 3), dtype=np.uint8)}, samples)
        page.worker = worker
        try:
            page._finished()
            self.assertEqual(page.badge.text(), 'COMPLETE')
            self.assertAlmostEqual(float(page.mouth.val.text()), 0)
            self.assertIsNone(page.worker)
            page._reset()
            self.assertEqual(page.index.val.text(), '---')
            self.assertEqual(page.badge.text(), 'IDLE')
        finally:
            page.cleanup()
            page.deleteLater()

    def test_neutral_face_in_every_stage_shows_visible_reason(self):
        page = FacialPage()
        page.resize(1400, 860)
        page.show()
        self.qt.processEvents()
        worker = Mock(error='', reason='deadline')
        samples = []
        bounds = {}
        for stage, start in [('rest', 0), ('smile', 5), ('brows', 10)]:
            bounds[stage] = (start, start + 3)
            for t in np.arange(start, start + 3, 0.05):
                samples.append({'stage': stage, 'timestamp': float(t), 'landmarks': symmetric_face(), 'pose': (0, 0, 0)})
        worker.take_pending.return_value = ({'stage_bounds': bounds, 'counts': {stage: 60 for stage in bounds},
                                             'rejected': {stage: 0 for stage in bounds}, 'stage': 'brows',
                                             'progress': 1, 'message': 'Complete',
                                             'image': np.zeros((480, 640, 3), dtype=np.uint8)}, samples)
        page.worker = worker
        try:
            self.assertLess(page.guidance.geometry().bottom(), page.capture_area.geometry().top())
            with patch('app.sys.stderr'):
                page._finished()
            self.assertEqual(page.badge.text(), 'RETRY')
            self.assertIn('Expression was too small', page.verdict.text())
            self.assertIn('smile excursion', page.verdict.text())
            self.assertIn('brow excursion', page.verdict.text())
            self.assertEqual(page.index.val.text(), '---')
            self.assertTrue(page.start.isEnabled())
        finally:
            page.cleanup()
            page.close()
            page.deleteLater()

    def test_stage_prompt_is_visible_above_camera_during_capture(self):
        page = FacialPage()
        page.resize(1400, 860)
        page.show()
        self.qt.processEvents()
        worker = Mock()
        worker.take_pending.return_value = ({'stage_bounds': {}, 'counts': {}, 'rejected': {},
                                             'stage': 'smile', 'progress': 0.51,
                                             'message': 'Smile clearly and hold / CAPTURING',
                                             'image': np.zeros((480, 640, 3), dtype=np.uint8)}, [])
        page.worker = worker
        try:
            page._poll()
            self.assertIn('Smile clearly', page.guidance.text())
            self.assertEqual(page.badge.text(), 'SMILE')
            self.assertLess(page.guidance.geometry().bottom(), page.capture_area.geometry().top())
        finally:
            page.worker = None
            page.cleanup()
            page.close()
            page.deleteLater()

    def test_stop_and_reset_discard_session_and_allow_next_capture(self):
        page = FacialPage()
        with patch('app.FaceCaptureWorker', side_effect=FakeFaceWorker):
            page.start.click()
            worker = page.worker
            page.reset.click()
            worker.requestInterruption.assert_called()
            self.assertFalse(page.start.isEnabled())
            worker.finished.emit()
            self.assertTrue(page.start.isEnabled())
            self.assertEqual(page.index.val.text(), '---')
            page.start.click()
            self.assertFalse(page.cancelled)
            page.cleanup()
            page.worker.finished.emit()
        page.deleteLater()

    def test_face_process_failure_leaves_retry_and_releases_worker(self):
        page = FacialPage()
        with patch('app.FaceCaptureWorker', side_effect=FakeFaceWorker):
            page.start.click()
            worker = page.worker
            worker.error = 'Capture timed out during face inference'
            worker.reason = 'error'
            worker.finished.emit()
            self.assertEqual(page.badge.text(), 'RETRY')
            self.assertIn('face inference', page.guidance.text())
            self.assertTrue(page.start.isEnabled())
            self.assertIsNone(page.worker)
            worker.deleteLater.assert_called_once()
        page.cleanup()
        page.deleteLater()

    def test_close_waits_for_face_worker(self):
        window = MainWindow()
        with patch('app.FaceCaptureWorker', side_effect=FakeFaceWorker):
            page = window.facial_page
            page.start.click()
            worker = page.worker
            event = QCloseEvent()
            window.closeEvent(event)
            self.assertFalse(event.isAccepted())
            worker.requestInterruption.assert_called()
            worker.finished.emit()
            self.qt.processEvents()
            self.assertIsNone(page.worker)
        window.close()
        window.deleteLater()

    def test_download_rejects_truncation_and_removes_only_temp_file(self):
        output = MagicMock()
        output.name = 'face.part'
        output.tell.return_value = 5
        response = MagicMock()
        response.headers = {'Content-Length': '10'}
        response.read1.side_effect = [b'abcde', b'']
        with patch('face_capture.os.path.isfile', return_value=False), patch('face_capture.os.path.exists', return_value=True), patch('face_capture.os.makedirs'), patch('face_capture.tempfile.NamedTemporaryFile') as temporary, patch('face_capture.urllib.request.urlopen') as request, patch('face_capture.os.unlink') as unlink, patch('face_capture.os.rename') as rename:
            temporary.return_value.__enter__.return_value = output
            request.return_value.__enter__.return_value = response
            with self.assertRaises(OSError):
                ensure_face_model(lambda: False)
            unlink.assert_called_once_with('face.part')
            rename.assert_not_called()


if __name__ == '__main__':
    unittest.main()
