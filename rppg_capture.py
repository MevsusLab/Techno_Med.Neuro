import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from PyQt6.QtCore import QCoreApplication, QThread

from capture_worker import ProcessCaptureWorker
from face_capture import MODEL_PATH, ensure_face_model, head_pose


REGIONS = ((10, 109, 108, 151, 337, 338), (50, 101, 205, 187), (280, 330, 425, 411))


def extract_regions(frame, landmarks, pose):
    points = np.asarray(landmarks, dtype=float)
    h, w = frame.shape[:2]
    if points.shape[0] < 468 or not np.isfinite(points).all() or not np.isfinite(pose).all():
        raise ValueError('Face geometry is unavailable.')
    if max(abs(pose[0]), abs(pose[1])) > 12 or abs(pose[2]) > 20:
        raise ValueError('Look directly at the camera and keep the head level.')
    span = np.linalg.norm(points[263, :2] - points[33, :2])
    if span < 90:
        raise ValueError('Move closer; skin regions need more pixels.')
    colors, polygons = [], []
    for indices in REGIONS:
        polygon = points[list(indices), :2]
        polygon = (polygon.mean(axis=0) + 0.75 * (polygon - polygon.mean(axis=0))).astype(np.int32)
        if np.any(polygon < 4) or np.any(polygon[:, 0] >= w - 4) or np.any(polygon[:, 1] >= h - 4):
            raise ValueError('Keep the forehead and both cheeks fully inside the frame.')
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(polygon), 255)
        pixels = frame[mask > 0]
        if len(pixels) < 80:
            raise ValueError('Skin regions are too small for reliable averaging.')
        if np.mean((pixels <= 5) | (pixels >= 250)) > 0.05:
            raise ValueError('A skin region is underexposed or clipped. Use diffuse frontal light.')
        color = pixels[:, ::-1].mean(axis=0)
        if min(color) < 30 or max(color) > 240:
            raise ValueError('Improve lighting on forehead and cheeks.')
        colors.append(color)
        polygons.append(polygon)
    face_min = points[:468, :2].min(axis=0)
    face_max = points[:468, :2].max(axis=0)
    size = max(10, min(h, w) // 12)
    corners = ((0, 0), (w - size, 0), (0, h - size), (w - size, h - size))
    available = [(x, y) for x, y in corners if x + size < face_min[0] or x > face_max[0] or y + size < face_min[1] or y > face_max[1]]
    if not available:
        raise ValueError('Move slightly farther from the camera to leave a background reference.')
    x, y = available[0]
    reference = float(frame[y:y + size, x:x + size, 1].mean())
    center = points[[33, 263], :2].mean(axis=0)
    return np.asarray(colors), polygons, reference, center, float(span)


class RPPGSession(QThread):
    def __init__(self, duration=30):
        super().__init__()
        self.duration = duration
        self.lock = threading.Lock()
        self.phase = 'model setup'
        self.phase_started = time.perf_counter()
        self.latest = None
        self.samples = []
        self.error = ''
        self.reason = 'stopped'
        self.record_start = None
        self.record_end = None
        self.frame_count = 0
        self.read_timeout = 10.0

    def phase_change(self, phase):
        with self.lock:
            self.phase, self.phase_started = phase, time.perf_counter()

    def run(self):
        camera = model = None
        try:
            ensure_face_model(self.isInterruptionRequested)
            if self.isInterruptionRequested():
                return
            self.phase_change('model initialization')
            model = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH), running_mode=RunningMode.VIDEO,
                num_faces=2, output_facial_transformation_matrixes=True, min_face_detection_confidence=0.6,
            ))
            self.phase_change('camera opening')
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise OSError('Camera could not be opened')
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            opened = last_frame = time.perf_counter()
            first = None
            steady = anchor = anchor_span = None
            last_timestamp = 0
            while not self.isInterruptionRequested():
                self.phase_change('camera read')
                ok, frame = camera.read()
                now = time.perf_counter()
                if self.isInterruptionRequested():
                    break
                if now - last_frame > self.read_timeout:
                    raise TimeoutError('Camera stopped delivering rPPG frames')
                if not ok:
                    self.msleep(10)
                    continue
                first = now if first is None else first
                last_frame = now
                self.read_timeout = 2
                if self.record_start is not None and now - self.record_start >= self.duration:
                    self.reason = 'deadline'
                    self.record_end = self.record_start + self.duration
                    break
                if now - opened > self.duration + 25:
                    raise TimeoutError('rPPG preparation timed out')
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.phase_change('face inference')
                last_timestamp = max(last_timestamp + 1, int(now * 1000))
                result = model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame), last_timestamp)
                sample = None
                polygons = []
                message = 'Keep one face visible; no speaking or facial movements.'
                if len(result.face_landmarks) == 1 and result.facial_transformation_matrixes:
                    h, w = frame.shape[:2]
                    points = np.array([[p.x * w, p.y * h, p.z * w] for p in result.face_landmarks[0]])
                    try:
                        pose = head_pose(result.facial_transformation_matrixes[0])
                        colors, polygons, background, center, span = extract_regions(frame, points, pose)
                        if self.record_start is None:
                            if anchor is None or np.linalg.norm(center - anchor) / span > 0.03 or abs(span / anchor_span - 1) > 0.03:
                                anchor, anchor_span, steady = center, span, now
                            if now - steady >= 3:
                                self.record_start = now
                        geometry = np.array([*(center / anchor_span), span / anchor_span, *pose])
                        sample = {'timestamp': now, 'rgb': colors, 'background': background, 'geometry': geometry}
                        message = 'RECORDING: remain still under constant light.' if self.record_start is not None else 'PREPARING: remain still for 3 seconds.'
                    except ValueError as exc:
                        message = str(exc)
                        if self.record_start is None:
                            anchor = steady = None
                elif self.record_start is None:
                    anchor = steady = None
                if self.record_start is None and now - first > 15:
                    raise TimeoutError('Preparation failed: ' + message)
                if self.record_start is not None:
                    self.frame_count += 1
                    self.record_end = now
                    if sample is not None:
                        with self.lock:
                            self.samples.append(sample)
                for i, polygon in enumerate(polygons):
                    cv2.polylines(rgb_frame, [polygon], True, (80, 190, 255), 1)
                    origin = tuple(polygon.min(axis=0))
                    cv2.putText(rgb_frame, ('FOREHEAD', 'R CHEEK', 'L CHEEK')[i], origin, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 190, 255), 1)
                elapsed = 0 if self.record_start is None else now - self.record_start
                with self.lock:
                    self.latest = {'image': rgb_frame, 'message': message, 'progress': elapsed / self.duration,
                                   'start': self.record_start, 'end': self.record_end, 'frames': self.frame_count}
        except InterruptedError:
            pass
        except Exception as exc:
            self.error, self.reason = f'{type(exc).__name__}: {exc}', 'error'
        finally:
            for resource, method in ((camera, 'release'), (model, 'close')):
                if resource is not None:
                    self.phase_change('resource ' + method)
                    try:
                        getattr(resource, method)()
                    except Exception as exc:
                        self.error, self.reason = str(exc), 'error'


def rppg_process(connection, stop, duration, preparation):
    qt = QCoreApplication([])
    worker = RPPGSession(duration)
    try:
        worker.start()
        while True:
            if stop.is_set():
                worker.requestInterruption()
            done = worker.wait(0)
            with worker.lock:
                latest, samples = worker.latest, worker.samples
                worker.latest, worker.samples = None, []
                phase, started = worker.phase, worker.phase_started
                bounds = (worker.record_start, worker.record_end, None, worker.read_timeout)
            connection.send((latest, samples, bounds, phase, started, [phase.upper()] if latest is None else [], done, worker.error if done else '', worker.reason))
            if done:
                break
            time.sleep(0.03)
    except (BrokenPipeError, EOFError, OSError):
        worker.requestInterruption()
        worker.wait(500)
    finally:
        connection.close()
    if worker.isRunning():
        os._exit(1)


class RPPGCaptureWorker(ProcessCaptureWorker):
    def __init__(self, parent=None):
        super().__init__(duration=30, parent=parent, process_target=rppg_process)
