import os
import tempfile
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, FaceLandmarksConnections, RunningMode
from PyQt6.QtCore import QCoreApplication, QThread
from capture_worker import ProcessCaptureWorker
from facial_engine import normalize_face, PAIRS


MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'face_landmarker.task')
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'


def ensure_face_model(cancelled):
    if cancelled():
        raise InterruptedError('Face model setup cancelled')
    if os.path.isfile(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(MODEL_PATH), suffix='.part', delete=False) as output:
            temporary = output.name
            started = time.perf_counter()
            with urllib.request.urlopen(MODEL_URL, timeout=10) as response:
                expected = int(response.headers.get('Content-Length', 0))
                while True:
                    if cancelled():
                        raise InterruptedError('Face model setup cancelled')
                    if time.perf_counter() - started > 60:
                        raise TimeoutError('Face model download timed out')
                    data = response.read1(65536)
                    if not data:
                        break
                    output.write(data)
                if output.tell() == 0 or (expected and output.tell() != expected):
                    raise OSError('Face model download is incomplete')
        if not os.path.exists(MODEL_PATH):
            if cancelled():
                raise InterruptedError('Face model setup cancelled')
            os.rename(temporary, MODEL_PATH)
            temporary = None
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def head_pose(matrix):
    if np.asarray(matrix).shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Head transform is invalid.')
    rotation = np.asarray(matrix)[:3, :3]
    scale = np.linalg.norm(rotation, axis=0)
    if np.min(scale) < 1e-8:
        raise ValueError('Head transform is degenerate.')
    rotation = rotation / scale
    pitch = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0]))
    roll = np.arctan2(rotation[1, 0], rotation[0, 0])
    return tuple(np.degrees([yaw, pitch, roll]))


class FaceSession(QThread):
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.phase = 'model setup'
        self.phase_started = time.perf_counter()
        self.latest = None
        self.samples = []
        self.bounds = {}
        self.error = ''
        self.reason = 'stopped'
        self.rejected = {'rest': 0, 'smile': 0, 'brows': 0}
        self.counts = {'rest': 0, 'smile': 0, 'brows': 0}
        self.read_timeout = 10.0

    def phase_change(self, phase):
        with self.lock:
            self.phase = phase
            self.phase_started = time.perf_counter()

    def run(self):
        camera = model = None
        try:
            ensure_face_model(self.isInterruptionRequested)
            if self.isInterruptionRequested():
                return
            self.phase_change('model initialization')
            model = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH), running_mode=RunningMode.VIDEO,
                num_faces=2, min_face_detection_confidence=0.6,
                output_facial_transformation_matrixes=True,
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
            steady_since = None
            sequence_start = None
            timestamp_ms = 0
            previous_center = None
            previous_width = None
            instructions = {'rest': 'Расслабьте лицо', 'smile': 'Улыбнитесь и удерживайте улыбку', 'brows': 'Поднимите обе брови и удерживайте положение'}
            while not self.isInterruptionRequested():
                self.phase_change('camera read')
                ret, frame = camera.read()
                now = time.perf_counter()
                if self.isInterruptionRequested():
                    break
                if not ret:
                    if now - last_frame > (10 if first is None else 2):
                        raise TimeoutError('Camera stopped delivering face frames')
                    self.msleep(10)
                    continue
                if now - last_frame > (10 if first is None else 2):
                    raise TimeoutError('Face camera frame delivery exceeded its time limit.')
                last_frame = now
                if first is None:
                    first = now
                    self.read_timeout = 2.0
                if now - opened > 40:
                    raise TimeoutError('Face capture timed out')
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.phase_change('face inference')
                timestamp_ms = max(timestamp_ms + 1, int(now * 1000))
                result = model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                points = pose = None
                geometry = None
                quality = 'Keep one face fully visible and look directly at the camera'
                if len(result.face_landmarks) == 1 and result.facial_transformation_matrixes:
                    points = np.array([[p.x * w, p.y * h, p.z * w] for p in result.face_landmarks[0]])
                    pose = head_pose(result.facial_transformation_matrixes[0])
                    try:
                        geometry = normalize_face(points, pose)
                        quality = ''
                    except ValueError as exc:
                        quality = str(exc)
                valid = points is not None and not quality
                if valid:
                    x0, y0 = np.maximum(points[:468, :2].min(axis=0).astype(int), 0)
                    x1, y1 = np.minimum(points[:468, :2].max(axis=0).astype(int), [w - 1, h - 1])
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
                    if x0 < 5 or y0 < 5 or x1 > w - 6 or y1 > h - 6:
                        quality = 'Держите лицо полностью внутри кадра.'
                    elif gray.size == 0 or gray.mean() < 35 or gray.mean() > 235 or cv2.Laplacian(gray, cv2.CV_64F).var() < 12:
                        quality = 'Изображение лица тёмное, пересвеченное или размытое. Улучшите освещение и фокусировку.'
                    valid = not quality
                if sequence_start is None:
                    if valid:
                        center = points[[33, 263], :2].mean(axis=0)
                        width = geometry['width_px']
                        if previous_center is not None and (np.linalg.norm(center - previous_center) > 8 or abs(width / previous_width - 1) > 0.04):
                            steady_since = None
                        previous_center, previous_width = center, width
                    if valid:
                        steady_since = now if steady_since is None else steady_since
                        if now - steady_since >= 2:
                            sequence_start = now
                    else:
                        steady_since = None
                    if now - first > 15:
                        raise TimeoutError('Preparation failed: ' + (quality or 'Keep the head steady'))
                elapsed = 0 if sequence_start is None else now - sequence_start
                if elapsed >= 15:
                    self.bounds['brows'] = (sequence_start + 12, sequence_start + 15)
                    with self.lock:
                        if self.latest is not None:
                            self.latest['stage_bounds'] = dict(self.bounds)
                    self.reason = 'deadline'
                    break
                stage = 'preparing'
                capturing = False
                message = quality or 'Подготовка: расслабьте лицо и смотрите прямо в камеру'
                if sequence_start is not None:
                    stage_index = int(elapsed // 5)
                    stage = ('rest', 'smile', 'brows')[stage_index]
                    capturing = elapsed % 5 >= 2
                    message = (quality + ' | ' if quality else '') + instructions[stage] + (' / ИДЁТ ЗАПИСЬ' if capturing else ' / ПОДГОТОВКА')
                    start = sequence_start + stage_index * 5 + 2
                    self.bounds[stage] = (start, start + 3)
                    if capturing:
                        self.counts[stage] += 1
                    if capturing and valid:
                        with self.lock:
                            self.samples.append({'stage': stage, 'timestamp': now, 'landmarks': points, 'pose': pose})
                    elif capturing:
                        self.rejected[stage] += 1
                if points is not None:
                    for edge in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
                        cv2.line(rgb, tuple(points[edge.start, :2].astype(int)), tuple(points[edge.end, :2].astype(int)), (45, 65, 85), 1)
                    for x, y, _ in points[:468]:
                        cv2.circle(rgb, (int(x), int(y)), 1, (100, 150, 200), -1)
                    cv2.line(rgb, tuple(points[9, :2].astype(int)), tuple(points[152, :2].astype(int)), (80, 180, 255), 1)
                    if valid:
                        for region, pairs in PAIRS.items():
                            ratio = min(geometry['regions'][region] / 10, 1)
                            color = (255, int(240 * (1 - ratio)), int(100 * (1 - ratio)))
                            for pair in pairs:
                                for index in pair:
                                    cv2.circle(rgb, tuple(points[index, :2].astype(int)), 4, color, -1)
                with self.lock:
                    self.latest = {'image': rgb, 'message': message, 'stage': stage,
                                   'progress': min(elapsed / 15, 1), 'stage_bounds': dict(self.bounds),
                                   'rejected': dict(self.rejected), 'counts': dict(self.counts),
                                   'landmark_count': len(points) if points is not None else 0}
        except InterruptedError:
            pass
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self.reason = 'error'
        finally:
            for resource, method in ((camera, 'release'), (model, 'close')):
                if resource is not None:
                    self.phase_change('resource ' + method)
                    try:
                        getattr(resource, method)()
                    except Exception as exc:
                        self.error = str(exc)
                        self.reason = 'error'


def face_process(connection, stop, duration, preparation):
    qt = QCoreApplication([])
    worker = FaceSession()
    try:
        worker.start()
        while True:
            if stop.is_set():
                worker.requestInterruption()
            done = worker.wait(0)
            with worker.lock:
                latest, samples = worker.latest, worker.samples
                worker.latest = None
                worker.samples = []
                phase, phase_started = worker.phase, worker.phase_started
                read_timeout = worker.read_timeout
            connection.send((latest, samples, (None, None, None, read_timeout), phase, phase_started, [phase.upper()] if latest is None else [], done,
                             worker.error if done else '', worker.reason))
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


class FaceCaptureWorker(ProcessCaptureWorker):
    def __init__(self, parent=None):
        super().__init__(parent=parent, process_target=face_process)
