import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import multiprocessing
import time
import unittest

import numpy as np
from PyQt6.QtCore import QCoreApplication
from capture_worker import ProcessCaptureWorker, HAND_MODEL_PATH, _capture_process


def completed_capture(connection, stop, duration, preparation):
    start = time.perf_counter() - 15
    t = start + np.arange(450) / 30
    samples = [(float(ti), float(10 * np.sin(2 * np.pi * 5 * ti)), 0.0) for ti in t]
    connection.send((None, samples, (start, float(t[-1]), None, 2.0),
                     "resource close", time.perf_counter(), ["CAM READY"], True, "", "deadline"))
    connection.close()


def blocked_inference(connection, stop, duration, preparation):
    connection.send((None, [], (None, None, None, 2.0),
                     "hand inference", time.perf_counter() - 10, [], False, "", "stopped"))
    while True:
        time.sleep(1)


def blocked_cleanup(connection, stop, duration, preparation):
    connection.send((None, [], (None, None, None, 2.0),
                     "resource close", time.perf_counter() - 10, [], False, "", "stopped"))
    while True:
        time.sleep(1)


def cancellation_ignored(connection, stop, duration, preparation):
    while True:
        connection.send((None, [], (None, None, None, 10.0),
                         "model setup", time.perf_counter(), [], False, "", "stopped"))
        time.sleep(0.03)


def native_failure(connection, stop, duration, preparation):
    os._exit(7)


def model_probe(connection, stop, duration, preparation):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
                                    running_mode=RunningMode.VIDEO, num_hands=1)
    with HandLandmarker.create_from_options(options) as model:
        model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                        data=np.zeros((480, 640, 3), dtype=np.uint8)), 1)
    connection.send((None, [], (None, None, None, 2.0), "resource close", time.perf_counter(),
                     [], True, "", "deadline"))
    connection.close()


def worker_probe(connection, stop, duration, preparation):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    camera = Mock()
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
    detector = Mock()
    detector.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[landmarks])

    def read():
        time.sleep(0.01)
        return True, np.zeros((12, 16, 3), dtype=np.uint8)

    camera.read.side_effect = read
    with patch("capture_worker.ensure_model"), patch("capture_worker.cv2.VideoCapture", return_value=camera), patch("capture_worker.HandLandmarker.create_from_options", return_value=detector):
        _capture_process(connection, stop, 0.3, 0)


def face_model_probe(connection, stop, duration, preparation):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
    from face_capture import MODEL_PATH
    with FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH), running_mode=RunningMode.VIDEO,
        num_faces=2, output_facial_transformation_matrixes=True,
    )) as model:
        result = model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                data=np.zeros((480, 640, 3), dtype=np.uint8)), 1)
        assert not result.face_landmarks
    connection.send((None, [], (None, None, None, 10.0), 'resource close', time.perf_counter(), [], True, '', 'deadline'))
    connection.close()


class ProcessCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QCoreApplication.instance() or QCoreApplication([])

    def run_worker(self, target, cancel=False):
        before = {p.pid for p in multiprocessing.active_children()}
        worker = ProcessCaptureWorker(process_target=target)
        worker.start()
        if cancel:
            time.sleep(0.3)
            worker.requestInterruption()
        self.assertTrue(worker.wait(15000), "Supervisor failed to finish")
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)
        return worker

    def test_repeated_process_sessions_preserve_all_scalar_samples(self):
        for trial in range(3):
            worker = self.run_worker(completed_capture)
            self.assertEqual(worker.error, "")
            self.assertEqual(worker.reason, "deadline")
            _, samples = worker.take_pending()
            self.assertEqual(len(samples), 450)
            self.assertEqual(worker.snapshot()[0], samples[0][0])

    def test_child_worker_transport_includes_final_frame_and_recording_bounds(self):
        worker = self.run_worker(worker_probe)
        self.assertEqual(worker.error, "")
        self.assertEqual(worker.reason, "deadline")
        frame, samples = worker.take_pending()
        self.assertGreater(len(samples), 5)
        self.assertEqual(samples[0][1:], (8.0, 6.0))
        self.assertEqual(worker.snapshot()[:2], (samples[0][0], samples[-1][0]))
        self.assertEqual(frame[1], samples[-1][0])

    def test_blocked_inference_is_terminated_and_restart_succeeds(self):
        worker = self.run_worker(blocked_inference)
        self.assertIn("hand inference", worker.error)
        self.assertEqual(self.run_worker(completed_capture).error, "")

    def test_blocked_cleanup_is_terminated(self):
        self.assertIn("resource close", self.run_worker(blocked_cleanup).error)

    def test_cancellation_can_release_unresponsive_native_session(self):
        self.assertIn("did not stop", self.run_worker(cancellation_ignored, cancel=True).error)

    def test_native_process_crash_does_not_crash_supervisor(self):
        worker = self.run_worker(native_failure)
        self.assertTrue(worker.error)
        self.assertEqual(worker.reason, "error")

    def test_real_model_can_initialize_and_close_in_repeated_child_processes(self):
        for trial in range(3):
            self.assertEqual(self.run_worker(model_probe).error, "")

    def test_real_face_model_initializes_and_closes_in_child_process(self):
        self.assertEqual(self.run_worker(face_model_probe).error, '')


if __name__ == "__main__":
    multiprocessing.freeze_support()
    unittest.main()
