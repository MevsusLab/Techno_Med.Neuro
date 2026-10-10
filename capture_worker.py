import os
import multiprocessing
import tempfile
import threading
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from PyQt6.QtCore import QCoreApplication, QThread, Qt, pyqtSignal


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
CAMERA_STARTUP_TIMEOUT = 10.0
CAMERA_STALL_TIMEOUT = 2.0
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (0,17),(13,17),(17,18),(18,19),(19,20),
]


def ensure_model(cancelled):
    if cancelled():
        raise InterruptedError("Model download cancelled")
    if os.path.isfile(HAND_MODEL_PATH):
        return
    os.makedirs(MODELS_DIR, exist_ok=True)
    temp_path = None
    started = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(dir=MODELS_DIR, suffix=".part", delete=False) as output:
            temp_path = output.name
            with urllib.request.urlopen(HAND_MODEL_URL, timeout=10) as response:
                while True:
                    if cancelled():
                        raise InterruptedError("Model download cancelled")
                    if time.perf_counter() - started > 60:
                        raise TimeoutError("Model download timed out")
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    output.write(chunk)
            if output.tell() == 0:
                raise OSError("Downloaded model is empty")
        if cancelled():
            raise InterruptedError("Model download cancelled")
        if not os.path.exists(HAND_MODEL_PATH):
            try:
                os.rename(temp_path, HAND_MODEL_PATH)
                temp_path = None
            except FileExistsError:
                if not os.path.isfile(HAND_MODEL_PATH):
                    raise
    finally:
        if temp_path is not None:
            os.unlink(temp_path)


class CaptureWorker(QThread):
    status = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, duration=15.0, parent=None, preparation_duration=2.0):
        super().__init__(parent)
        self.duration = duration
        self.preparation_duration = preparation_duration
        self.error = ""
        self.reason = "stopped"
        self._lock = threading.Lock()
        self._latest = None
        self._samples = []
        self._sample_limit = max(1, int(duration * 1000)) if duration is not None else 180
        self._sample_count = 0
        self._record_start = None
        self._record_end = None
        self._read_wait_since = None
        self._read_timeout = CAMERA_STARTUP_TIMEOUT
        self.phase = "starting"
        self.phase_started = time.perf_counter()

    def _set_phase(self, phase):
        with self._lock:
            self.phase = phase
            self.phase_started = time.perf_counter()

    def snapshot(self):
        with self._lock:
            return self._record_start, self._record_end, self._read_wait_since, self._read_timeout

    def take_pending(self):
        with self._lock:
            latest, samples = self._latest, self._samples
            self._latest = None
            self._samples = []
        return latest, samples

    def run(self):
        capture = None
        landmarker = None
        try:
            if self.isInterruptionRequested():
                return
            self.status.emit("MODEL SETUP")
            self._set_phase("model setup")
            ensure_model(self.isInterruptionRequested)
            if self.isInterruptionRequested():
                return
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
                num_hands=1,
                running_mode=RunningMode.VIDEO,
                min_hand_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
            self._set_phase("model initialization")
            landmarker = HandLandmarker.create_from_options(options)
            if self.isInterruptionRequested():
                return
            self.status.emit("CAM OPENING")
            self._set_phase("camera opening")
            capture = cv2.VideoCapture(0)
            if self.isInterruptionRequested():
                return
            if not capture.isOpened():
                raise OSError("Camera could not be opened")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_FPS, 30)
            self.status.emit("CAM READY")
            self.status.emit("PREPARING: Hold one hand still in view for 2 seconds.")
            last_frame = time.perf_counter()
            camera_started = last_frame
            has_frame = False
            ts_ms = 0
            preview_start = None
            steady_since = None
            steady_count = 0
            previous_tip = None
            steady_anchor = None
            while not self.isInterruptionRequested():
                if self._record_start is not None and time.perf_counter() - self._record_start >= self.duration:
                    self.reason = "deadline"
                    break
                with self._lock:
                    self._read_wait_since = time.perf_counter()
                    self._read_timeout = CAMERA_STALL_TIMEOUT if has_frame else CAMERA_STARTUP_TIMEOUT
                self._set_phase("camera read")
                ret, frame = capture.read()
                acquired = time.perf_counter()
                if self.isInterruptionRequested():
                    break
                if not has_frame and acquired - camera_started > CAMERA_STARTUP_TIMEOUT:
                    raise TimeoutError("Camera stopped delivering frames during startup (10-second first-frame timeout)")
                if not ret:
                    reference = last_frame if has_frame else camera_started
                    timeout = CAMERA_STALL_TIMEOUT if has_frame else CAMERA_STARTUP_TIMEOUT
                    if acquired - reference > timeout:
                        raise TimeoutError("Camera stopped delivering frames")
                    self.msleep(10)
                    continue
                if has_frame and acquired - last_frame > CAMERA_STALL_TIMEOUT:
                    raise TimeoutError("Camera delivery stalled")
                if self._record_start is not None and acquired - self._record_start > self.duration:
                    self.reason = "deadline"
                    break
                previous_frame = last_frame
                last_frame = acquired
                has_frame = True
                with self._lock:
                    self._read_wait_since = None
                    self._read_timeout = CAMERA_STALL_TIMEOUT
                if preview_start is None:
                    preview_start = acquired
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts_ms = max(ts_ms + 1, int(acquired * 1000))
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                self._set_phase("hand inference")
                result = landmarker.detect_for_video(image, ts_ms)
                x = y = None
                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                    for a, b in HAND_CONNECTIONS:
                        cv2.line(frame, pts[a], pts[b], (184, 163, 148), 1, cv2.LINE_AA)
                    for px, py in pts:
                        cv2.circle(frame, (px, py), 2, (246, 130, 59), -1, cv2.LINE_AA)
                    tip = landmarks[8]
                    x, y = tip.x * w, tip.y * h
                    cx, cy = int(x), int(y)
                    cv2.circle(frame, (cx, cy), 7, (246, 130, 59), 2)
                    cv2.circle(frame, (cx, cy), 3, (246, 130, 59), -1)
                    cv2.putText(frame, f"({cx}, {cy})", (cx + 12, cy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (246, 130, 59), 1, cv2.LINE_AA)
                display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._lock:
                    if self._record_start is None:
                        moved = previous_tip is not None and x is not None and (
                            (x - previous_tip[0]) ** 2 + (y - previous_tip[1]) ** 2 > 12 ** 2
                        )
                        drifted = steady_anchor is not None and x is not None and (
                            (x - steady_anchor[0]) ** 2 + (y - steady_anchor[1]) ** 2 > 25 ** 2
                        )
                        if x is None or moved or drifted or acquired - previous_frame > 0.2:
                            steady_since = None
                            steady_count = 0
                            steady_anchor = None
                        if x is not None:
                            if steady_since is None:
                                steady_since = acquired
                                steady_anchor = (x, y)
                            steady_count += 1
                        previous_tip = (x, y) if x is not None else None
                        ready = self.preparation_duration == 0 or (
                            steady_since is not None and acquired - steady_since >= self.preparation_duration
                            and steady_count >= 30
                        )
                        if ready:
                            self._record_start = acquired
                        elif acquired - preview_start > 15:
                            raise TimeoutError("Preparation timed out. Keep one hand still, fully visible, and well lit.")
                    if self._record_start is not None:
                        if self._sample_count >= self._sample_limit:
                            raise RuntimeError("Capture sample limit exceeded")
                        self._sample_count += 1
                        self._samples.append((acquired, x, y))
                        self._record_end = acquired
                    self._latest = (display, acquired, x, y)
        except InterruptedError:
            if not self.isInterruptionRequested():
                self.error = "Capture setup interrupted"
                self.reason = "error"
                self.failed.emit(self.error)
        except Exception as exc:
            self.error = str(exc) or type(exc).__name__
            self.reason = "error"
            self.failed.emit(self.error)
        finally:
            with self._lock:
                self._read_wait_since = None
            for resource, method in [(capture, "release"), (landmarker, "close")]:
                if resource is not None:
                    try:
                        self._set_phase(f"resource {method}")
                        getattr(resource, method)()
                    except Exception as exc:
                        if not self.error:
                            self.error = str(exc) or type(exc).__name__
                            self.reason = "error"
                            self.failed.emit(self.error)


def _capture_process(connection, stop, duration, preparation_duration):
    qt = QCoreApplication([])
    worker = CaptureWorker(duration=duration, preparation_duration=preparation_duration)
    statuses = []
    status_lock = threading.Lock()

    def on_status(text):
        with status_lock:
            statuses.append(text)

    worker.status.connect(on_status, Qt.ConnectionType.DirectConnection)
    try:
        worker.start()
        while True:
            if stop.is_set():
                worker.requestInterruption()
            done = worker.wait(0)
            latest, samples = worker.take_pending()
            bounds = worker.snapshot()
            with worker._lock:
                phase = worker.phase
                phase_started = worker.phase_started
            with status_lock:
                updates = statuses[:]
                statuses.clear()
            connection.send((latest, samples, bounds, phase, phase_started, updates,
                             done, worker.error if done else "", worker.reason))
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


class ProcessCaptureWorker(CaptureWorker):
    def __init__(self, duration=15.0, parent=None, preparation_duration=2.0,
                 process_target=_capture_process, process_args=()):
        super().__init__(duration, parent, preparation_duration)
        self.process_target = process_target
        self.process_args = tuple(process_args)

    def run(self):
        context = multiprocessing.get_context("spawn")
        connection, child_connection = context.Pipe(duplex=False)
        stop = context.Event()
        process = context.Process(
            target=self.process_target,
            args=(child_connection, stop, self.duration, self.preparation_duration, *self.process_args),
            daemon=True,
        )
        cancellation_started = None
        phase = "process startup"
        phase_started = time.perf_counter()
        last_packet = phase_started
        try:
            process.start()
            child_connection.close()
            while True:
                now = time.perf_counter()
                if self.isInterruptionRequested() and cancellation_started is None:
                    cancellation_started = now
                    stop.set()
                if connection.poll(0.03):
                    latest, samples, bounds, phase, phase_started, updates, done, error, reason = connection.recv()
                    last_packet = time.perf_counter()
                    with self._lock:
                        if latest is not None:
                            self._latest = latest
                        self._samples.extend(samples)
                        if self.duration is None and len(self._samples) > self._sample_limit:
                            del self._samples[:-self._sample_limit]
                        self._record_start, self._record_end, self._read_wait_since, self._read_timeout = bounds
                    for text in updates:
                        self.status.emit(text)
                    if done:
                        self.error = error
                        self.reason = reason
                        break
                elif not process.is_alive():
                    raise RuntimeError(f"Capture process exited unexpectedly (code {process.exitcode})")
                if cancellation_started is not None:
                    if now - cancellation_started > 3:
                        if not self.error:
                            self.error = f"Capture did not stop during {phase}; its process was released."
                        self.reason = "error"
                        break
                else:
                    timeout = {
                        "process startup": 30, "model setup": 75, "model initialization": 15,
                        "camera opening": 15, "hand inference": 5, "face inference": 5,
                        "gaze inference": 5,
                        "microphone opening": 15, "microphone read": 3, "voice analysis": 120,
                        "audio file loading": 120,
                        "voice dependencies": 120,
                        "resource release": 3, "resource close": 3,
                        "camera read": self._read_timeout + 1,
                    }.get(phase, 10)
                    if now - phase_started > timeout or now - last_packet > 30:
                        raise TimeoutError(f"Capture timed out during {phase}; retry the capture.")
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.reason = "error"
        finally:
            stop.set()
            if process.pid is not None:
                process.join(0.5)
                if process.is_alive():
                    process.terminate()
                    process.join(2)
                if process.is_alive():
                    process.kill()
                    process.join(2)
                if process.is_alive():
                    self.error = "Capture process could not be released; close the application."
                    self.reason = "error"
                else:
                    process.close()
            connection.close()
            child_connection.close()
            with self._lock:
                self._read_wait_since = None
            if self.error:
                self.failed.emit(self.error)
