import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import multiprocessing
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PyQt6.QtCore import QCoreApplication

from capture_worker import ProcessCaptureWorker
from gaze_capture import CameraFrames, GazeDetector, GazeSession, gaze_process
from gaze_engine import gaze_features, tracking_features
from gaze_simulation import face_landmarks
from test_gaze import gaze_face

PROBE_GAZE = (6.0, 3.0)


def session_probe(connection, stop, duration, preparation):
    camera = Mock()
    points = face_landmarks(PROBE_GAZE) / [640, 480]
    detector = Mock()
    detector.process.return_value = points

    def read():
        time.sleep(1 / 60)
        return True, np.zeros((480, 640, 3), np.uint8)

    camera.read.side_effect = read
    with patch('gaze_capture.cv2.VideoCapture', return_value=camera), patch('gaze_capture.GazeDetector', return_value=detector):
        gaze_process(connection, stop, .5, 0)


def blocked_gaze_inference(connection, stop, duration, preparation):
    connection.send((None, [], (None, None, None, 2), 'gaze inference',
                     time.perf_counter() - 6, [], False, '', 'stopped'))
    while True:
        time.sleep(1)


def real_gaze_model_probe(connection, stop, duration, preparation):
    detector = GazeDetector(stop.is_set)
    try:
        assert detector.process(np.zeros((480, 640, 3), np.uint8), 1) is None
    finally:
        detector.close()
    connection.send((None, [], (None, None, None, 2), 'resource close',
                     time.perf_counter(), [], True, '', 'deadline'))
    connection.close()


class GazeCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QCoreApplication.instance() or QCoreApplication([])

    def test_solutions_facemesh_uses_requested_configuration_and_centers(self):
        model = Mock()
        landmarks = [SimpleNamespace(x=float(point[0]), y=float(point[1])) for point in gaze_face() / 640]
        model.process.return_value = SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=landmarks)])
        factory = Mock(return_value=model)
        solutions = SimpleNamespace(face_mesh=SimpleNamespace(FaceMesh=factory))
        with patch('gaze_capture.mp.solutions', solutions, create=True):
            detector = GazeDetector(lambda: False)
            factory.assert_called_once_with(refine_landmarks=True, max_num_faces=1,
                                            min_detection_confidence=.5, min_tracking_confidence=.5)
            result = detector.process(np.zeros((20, 30, 3), np.uint8), 1)
            self.assertEqual(result.shape, (478, 2))
            detector.close()
        model.close.assert_called_once()

    def test_current_tasks_api_keeps_single_face_and_same_gaze_landmarks(self):
        model = Mock()
        model.detect_for_video.return_value = SimpleNamespace(face_landmarks=[])
        with patch('gaze_capture.mp.solutions', None, create=True), patch('gaze_capture.ensure_face_model'), patch(
                'gaze_capture.FaceLandmarker.create_from_options', return_value=model) as factory:
            detector = GazeDetector(lambda: False)
            options = factory.call_args.args[0]
            self.assertEqual(options.num_faces, 1)
            self.assertEqual(options.min_face_detection_confidence, .5)
            self.assertIsNone(detector.process(np.zeros((20, 30, 3), np.uint8), 1))
            detector.close()

    def test_camera_continues_capturing_while_consumer_is_busy(self):
        camera = Mock()
        reads = []
        owners = []

        def read():
            owners.append(threading.get_ident())
            time.sleep(.002)
            reads.append(len(reads) + 1)
            return True, np.full((8, 8, 3), reads[-1] % 255, np.uint8)

        camera.read.side_effect = read
        camera.release.side_effect = lambda: owners.append(threading.get_ident())
        with patch('gaze_capture.cv2.VideoCapture', return_value=camera):
            frames = CameraFrames()
            frames.start()
            try:
                deadline = time.perf_counter() + 2
                while len(reads) < 10 and time.perf_counter() < deadline:
                    time.sleep(.005)
                latest, error, *_ = frames.take()
                self.assertGreaterEqual(len(reads), 10)
                self.assertEqual(error, '')
                self.assertGreaterEqual(int(latest[0][0, 0, 0]), 10)
                self.assertEqual(len({thread for thread in owners}), 1)
                self.assertNotEqual(owners[0], threading.get_ident())
            finally:
                frames.request_stop()
                frames.thread.join(2)
        self.assertFalse(frames.thread.is_alive())
        camera.release.assert_called_once()
        self.assertEqual(len(set(owners)), 1)

    def test_stop_before_setup_does_not_open_camera_or_detector(self):
        session = GazeSession()
        with patch.object(session, 'isInterruptionRequested', return_value=True), patch('gaze_capture.GazeDetector') as detector:
            session.run()
        detector.assert_not_called()

    def test_inference_failure_releases_camera_and_detector(self):
        camera = Mock()
        camera.read.side_effect = lambda: (time.sleep(.005) or True, np.zeros((30, 40, 3), np.uint8))
        detector = Mock()
        detector.process.side_effect = RuntimeError('inference failure')
        session = GazeSession()
        with patch('gaze_capture.cv2.VideoCapture', return_value=camera), patch('gaze_capture.GazeDetector', return_value=detector):
            session.run()
        self.assertIn('inference failure', session.error)
        self.assertFalse(session.camera.thread.is_alive())
        camera.release.assert_called_once()
        detector.close.assert_called_once()

    def test_camera_open_failure_is_retryable_and_detector_closed(self):
        camera = Mock()
        camera.isOpened.return_value = False
        detector = Mock()
        session = GazeSession()
        with patch('gaze_capture.cv2.VideoCapture', return_value=camera), patch('gaze_capture.GazeDetector', return_value=detector):
            session.run()
        self.assertIn('Камеру открыть не удалось', session.error)
        detector.close.assert_called_once()
        camera.release.assert_called_once()

    def test_process_transport_delivers_ordered_smoothed_gaze_at_over_30fps(self):
        before = {child.pid for child in multiprocessing.active_children()}
        worker = ProcessCaptureWorker(duration=None, process_target=session_probe)
        worker.start()
        self.assertTrue(worker.wait(15000))
        self.assertEqual(worker.error, '')
        latest, samples = worker.take_pending()
        self.assertGreater(len(samples), 10)
        timestamps = [sample['timestamp'] for sample in samples]
        self.assertTrue(np.all(np.diff(timestamps) > 0))
        pixels = face_landmarks(PROBE_GAZE)
        self.assertTrue(np.allclose(samples[-1]['gaze'], gaze_features(pixels)))
        self.assertTrue(np.allclose(samples[-1]['features'], tracking_features(pixels)))
        self.assertTrue(np.allclose(samples[-1]['raw_features'], tracking_features(pixels)))
        self.assertEqual(latest['image'].shape, (240, 320, 3))
        self.assertTrue(latest['image'].flags['C_CONTIGUOUS'])
        self.assertGreater(1 / np.median(np.diff(timestamps)), 30)
        self.assertGreater(latest['fps'], 30)
        self.assertEqual({child.pid for child in multiprocessing.active_children()}, before)

    def test_gaze_inference_timeout_terminates_and_reaps_child(self):
        before = {child.pid for child in multiprocessing.active_children()}
        worker = ProcessCaptureWorker(duration=None, process_target=blocked_gaze_inference)
        worker.start()
        self.assertTrue(worker.wait(15000))
        self.assertIn('gaze inference', worker.error)
        self.assertEqual({child.pid for child in multiprocessing.active_children()}, before)

    def test_real_installed_mediapipe_gaze_model_initializes_processes_and_closes(self):
        worker = ProcessCaptureWorker(duration=None, process_target=real_gaze_model_probe)
        worker.start()
        self.assertTrue(worker.wait(20000))
        self.assertEqual(worker.error, '')


if __name__ == '__main__':
    unittest.main()
