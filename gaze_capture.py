# захват видео и распознавание взгляда через mediapipe

from collections import deque
import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from PyQt6.QtCore import QCoreApplication, QThread

from capture_worker import CAMERA_STALL_TIMEOUT, CAMERA_STARTUP_TIMEOUT, ProcessCaptureWorker
from face_capture import MODEL_PATH, ensure_face_model
from gaze_engine import GazeSignalFilter, gaze_features, tracking_features


preview_width = 320
PREVIEW_WIDTH = preview_width


class GazeDetector:
    def __init__(self, cancelled):
        solutions = getattr(mp, 'solutions', None)
        self.is_solutions = solutions is not None and hasattr(solutions, 'face_mesh')
        if self.is_solutions:
            self.model = solutions.face_mesh.FaceMesh(
                refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            ensure_face_model(cancelled)
            if cancelled():
                raise InterruptedError()
            self.model = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH), running_mode=RunningMode.VIDEO,
                num_faces=1, min_face_detection_confidence=0.5, min_tracking_confidence=0.5,
            ))

    def process(self, rgb, timestamp_ms):
        rgb.flags.writeable = False
        try:
            if self.is_solutions:
                result = self.model.process(rgb)
                faces = result.multi_face_landmarks
                landmarks = faces[0].landmark if faces else None
            else:
                result = self.model.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                landmarks = result.face_landmarks[0] if result.face_landmarks else None
        finally:
            rgb.flags.writeable = True
        if landmarks is None:
            return None
        return np.array([[point.x, point.y] for point in landmarks], dtype=float)

    def close(self):
        self.model.close()


class CameraFrames:
    # отдельный поток для камеры без задержек на нейросеть

    def __init__(self):
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.latest = None
        self.error = ''
        self.done = False
        self.phase = 'camera opening'
        self.phase_started = time.perf_counter()
        self.read_timeout = CAMERA_STARTUP_TIMEOUT
        self.thread = threading.Thread(target=self._run, name='AAC camera', daemon=True)

    def _phase(self, phase):
        with self.condition:
            self.phase, self.phase_started = phase, time.perf_counter()

    def start(self):
        self.thread.start()

    def request_stop(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()

    def take(self):
        with self.condition:
            if self.latest is None and not self.done and not self.stop.is_set():
                self.condition.wait(0.03)
            latest, self.latest = self.latest, None
            return latest, self.error, self.phase, self.phase_started, self.read_timeout

    def _run(self):
        camera = None
        try:
            if self.stop.is_set():
                return
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise OSError('Камеру открыть не удалось.')
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            previous = time.perf_counter()
            has_frame = False
            while not self.stop.is_set():
                self._phase('camera read')
                ok, frame = camera.read()
                acquired = time.perf_counter()
                if self.stop.is_set():
                    break
                timeout = CAMERA_STALL_TIMEOUT if has_frame else CAMERA_STARTUP_TIMEOUT
                if acquired - previous > timeout:
                    raise TimeoutError('Камера перестала передавать кадры.' if has_frame
                                       else 'Камера не передала первый кадр за 10 секунд.')
                if not ok:
                    self.stop.wait(0.01)
                    continue
                previous = acquired
                has_frame = True
                with self.condition:
                    self.read_timeout = CAMERA_STALL_TIMEOUT
                    self.latest = (frame, acquired)
                    self.condition.notify()
        except Exception as exc:
            self.error = str(exc) or type(exc).__name__
        finally:
            if camera is not None:
                self._phase('resource release')
                try:
                    camera.release()
                except Exception as exc:
                    self.error = self.error or str(exc) or type(exc).__name__
            with self.condition:
                self.done = True
                self.condition.notify_all()


class GazeSession(QThread):
    def __init__(self, duration=None):
        super().__init__()
        self.duration = duration
        self.lock = threading.Lock()
        self.latest = None
        self.samples = []
        self.camera = None
        self.phase = 'model setup'
        self.phase_started = time.perf_counter()
        self.error = ''
        self.reason = 'stopped'
        self.read_timeout = CAMERA_STARTUP_TIMEOUT

    def phase_change(self, phase):
        with self.lock:
            self.phase, self.phase_started = phase, time.perf_counter()

    def requestInterruption(self):
        super().requestInterruption()
        if self.camera is not None:
            self.camera.request_stop()

    def run(self):
        detector = None
        try:
            if self.isInterruptionRequested():
                return
            detector = GazeDetector(self.isInterruptionRequested)
            if self.isInterruptionRequested():
                return
            self.camera = CameraFrames()
            self.camera.start()
            signal_filter = GazeSignalFilter()
            timestamp_ms = 0
            opened = time.perf_counter()
            frame_times = deque(maxlen=60)
            while not self.isInterruptionRequested():
                latest, error, phase, started, read_timeout = self.camera.take()
                if error:
                    raise OSError(error)
                if latest is None:
                    with self.lock:
                        self.phase, self.phase_started, self.read_timeout = phase, started, read_timeout
                    continue
                frame, acquired = latest
                if self.isInterruptionRequested():
                    break
                if self.duration is not None and acquired - opened >= self.duration:
                    self.reason = 'deadline'
                    break
                self.phase_change('gaze inference')
                rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                timestamp_ms = max(timestamp_ms + 1, int(acquired * 1000))
                points = detector.process(rgb, timestamp_ms)
                if self.isInterruptionRequested():
                    break
                gaze = None
                raw_features = None
                message = 'Держите лицо в кадре и смотрите на экран.'
                if points is not None:
                    h, w = rgb.shape[:2]
                    pixels = points[:, :2] * [w, h]
                    try:
                        ratios = gaze_features(pixels)
                        raw_features = tracking_features(pixels)
                        gaze = ratios
                        message = 'Фиксируйте взгляд на слове или стрелке до заполнения шкалы.'
                    except ValueError as exc:
                        raw_features = None
                        message = str(exc)
                    for indices in ((33, 160, 158, 133, 153, 144), (362, 385, 387, 263, 373, 380)):
                        contour = pixels[list(indices)].astype(np.int32)
                        cv2.polylines(rgb, [contour], True, (60, 125, 185), 1, cv2.LINE_AA)
                    for index in (468, 473):
                        if index < len(pixels):
                            cv2.circle(rgb, tuple(pixels[index].astype(int)), 4, (50, 170, 255), -1, cv2.LINE_AA)
                gaze, features, raw_features = signal_filter.update(gaze, raw_features, acquired)
                frame_times.append(acquired)
                sample = {'gaze': gaze, 'features': features, 'raw_features': raw_features,
                          'message': message, 'timestamp': acquired}
                h, w = rgb.shape[:2]
                # уменьшаем размер превью
                preview = cv2.resize(rgb, (preview_width, max(1, round(h * preview_width / w))),
                                     interpolation=cv2.INTER_AREA) if w > preview_width else rgb
                with self.lock:
                    self.latest = {'image': np.ascontiguousarray(preview), **sample,
                                   'fps': ((len(frame_times) - 1) / max(frame_times[-1] - frame_times[0], 1e-6)
                                           if len(frame_times) > 1 else 0.0)}
                    self.samples.append(sample)
                    # буфер последних кадров
                    if len(self.samples) > 180:
                        del self.samples[:-180]
        except InterruptedError:
            pass
        except Exception as exc:
            self.error, self.reason = f'{type(exc).__name__}: {exc}', 'error'
        finally:
            if self.camera is not None:
                self.camera.request_stop()
                self.phase_change('resource release')
                self.camera.thread.join(0.5)
                if self.camera.thread.is_alive():
                    self.error, self.reason = 'Камера не остановилась; процесс захвата будет освобождён.', 'error'
                elif self.camera.error:
                    self.error, self.reason = self.camera.error, 'error'
            if detector is not None:
                self.phase_change('resource close')
                try:
                    detector.close()
                except Exception as exc:
                    self.error, self.reason = str(exc) or type(exc).__name__, 'error'


def gaze_process(connection, stop, duration, preparation):
    qt = QCoreApplication([])
    worker = GazeSession(duration)
    try:
        worker.start()
        while True:
            if stop.is_set():
                worker.requestInterruption()
            done = worker.wait(0)
            with worker.lock:
                latest, worker.latest = worker.latest, None
                samples, worker.samples = worker.samples, []
                phase, started, read_timeout = worker.phase, worker.phase_started, worker.read_timeout
            connection.send((latest, samples, (None, None, None, read_timeout), phase, started, [], done,
                             worker.error if done else '', worker.reason))
            if done:
                break
            time.sleep(0.01)
    except (BrokenPipeError, EOFError, OSError):
        worker.requestInterruption()
        worker.wait(500)
    finally:
        connection.close()
    if worker.isRunning() or (worker.camera is not None and worker.camera.thread.is_alive()):
        os._exit(1)


class GazeCaptureWorker(ProcessCaptureWorker):
    def __init__(self, parent=None):
        super().__init__(duration=None, parent=parent, process_target=gaze_process)
