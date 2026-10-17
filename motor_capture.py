# захват видео и трекинг пальцев для теппинга

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from capture_worker import (ProcessCaptureWorker, HAND_MODEL_PATH, HAND_CONNECTIONS,
                            ensure_model, CAMERA_STARTUP_TIMEOUT, CAMERA_STALL_TIMEOUT)
from motor_engine import finger_separation
from video_capture import validate_video, source_timestamp, _preview, _packet


motor_video_filter = 'Видеофайлы (*.mp4 *.avi *.mov);;Все файлы (*.*)'
MOTOR_VIDEO_FILTER = motor_video_filter


def _motor_process(connection, stop, duration, preparation_duration, path=''):
    capture = model = None
    error, reason = '', 'stopped'
    bounds = (None, None, None, CAMERA_STARTUP_TIMEOUT)
    latest = None
    record_start = record_end = None
    last_image = None

    def phase(name, updates=None):
        _packet(connection, bounds=bounds, phase=name, updates=updates)

    try:
        if stop.is_set():
            return
        file_duration = fps = None
        if path:
            if Path(path).suffix.lower() not in {'.mp4', '.avi', '.mov'}:
                raise ValueError('Для моторики выберите MP4, AVI или MOV.')
            phase('video decoding')
            source, fps, _, file_duration = validate_video(path, 'motor')
            duration = min(duration, file_duration)
        phase('model setup', ['Загрузка модели кисти'])
        ensure_model(stop.is_set)
        if stop.is_set():
            return
        phase('model initialization')
        model = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH), running_mode=RunningMode.VIDEO,
            num_hands=1, min_hand_detection_confidence=.6, min_hand_presence_confidence=.6,
            min_tracking_confidence=.6,
        ))
        if stop.is_set():
            return
        phase('video decoding' if path else 'camera opening')
        capture = cv2.VideoCapture(str(source) if path else 0)
        if not capture.isOpened():
            raise OSError('Не удалось открыть видеофайл.' if path else 'Не удалось открыть камеру.')
        if not path:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_FPS, 30)
        opened = last_arrival = time.perf_counter()
        preview_start = ready_since = None
        frame_index = 0
        previous_timestamp = -1.0
        milliseconds = 0
        if path:
            record_start, record_end = 0.0, duration
        while not stop.is_set():
            now = time.perf_counter()
            if not path and record_start is not None and now - record_start >= duration:
                record_end = record_start + duration
                reason = 'deadline'
                break
            timeout = CAMERA_STALL_TIMEOUT if frame_index else CAMERA_STARTUP_TIMEOUT
            bounds = (record_start, record_end, now if not path else None, timeout)
            phase('video decoding' if path else 'camera read')
            ok, frame = capture.read()
            acquired = time.perf_counter()
            if stop.is_set():
                break
            if path and not ok:
                if previous_timestamp >= duration - 1.5 / fps:
                    reason = 'deadline'
                else:
                    raise ValueError('Декодирование видео прервано до конца пробы.')
                break
            if not path:
                reference = last_arrival if frame_index else opened
                if acquired - reference > timeout:
                    raise TimeoutError('Камера перестала передавать кадры.')
                if not ok:
                    stop.wait(.01)
                    continue
                timestamp = acquired
                if record_start is not None and timestamp - record_start >= duration:
                    record_end = record_start + duration
                    reason = 'deadline'
                    break
                last_arrival = acquired
            else:
                timestamp = source_timestamp(capture.get(cv2.CAP_PROP_POS_MSEC), frame_index, 1 / fps, previous_timestamp)
                previous_timestamp = timestamp
                if timestamp >= duration:
                    reason = 'deadline'
                    break
            frame_index += 1
            rgb = cv2.cvtColor(cv2.flip(frame, 1) if not path else frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            milliseconds = max(milliseconds + 1, round(timestamp * 1000))
            bounds = (record_start, record_end, None, CAMERA_STALL_TIMEOUT)
            phase('hand inference')
            detection = model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), milliseconds)
            distance = palm = None
            hand = ''
            message = 'Держите одну кисть полностью в кадре, ладонью к камере.'
            if len(detection.hand_landmarks) == 1:
                points = np.array([[p.x * w, p.y * h] for p in detection.hand_landmarks[0]])
                try:
                    if np.any(points < 3) or np.any(points > [w - 4, h - 4]):
                        raise ValueError('Держите кисть полностью внутри кадра.')
                    distance, palm = finger_separation(points)
                    handedness = getattr(detection, 'handedness', [])
                    if handedness and handedness[0] and handedness[0][0].score >= .8:
                        hand = handedness[0][0].category_name
                    pts = points.astype(int)
                    for a, b in HAND_CONNECTIONS:
                        cv2.line(rgb, tuple(pts[a]), tuple(pts[b]), (148, 163, 184), 1, cv2.LINE_AA)
                    for point in pts:
                        cv2.circle(rgb, tuple(point), 2, (59, 130, 246), -1, cv2.LINE_AA)
                    for i in (4, 8):
                        cv2.circle(rgb, tuple(pts[i]), 7, (255, 170, 40), 2, cv2.LINE_AA)
                    cv2.line(rgb, tuple(pts[4]), tuple(pts[8]), (255, 170, 40), 1, cv2.LINE_AA)
                except ValueError as exc:
                    message = str(exc)
                    distance = palm = None
            if preview_start is None:
                preview_start = timestamp
            if not path and record_start is None:
                if distance is None:
                    ready_since = None
                elif ready_since is None:
                    ready_since = timestamp
                if ready_since is not None and timestamp - ready_since >= preparation_duration:
                    record_start = timestamp
                elif timestamp - preview_start > 20:
                    raise TimeoutError('Не удалось подготовить кисть. Приблизьте ладонь и держите все пальцы в кадре.')
                elif distance is not None:
                    message = 'ПОДГОТОВКА: держите кисть в кадре. Затем максимально быстро раскрывайте и смыкайте пальцы.'
            samples = []
            if record_start is not None:
                elapsed = timestamp - record_start
                if not path:
                    record_end = timestamp
                samples = [{'timestamp': timestamp, 'distance': distance, 'palm_scale': palm, 'hand': hand}]
                state = 'PLAYBACK' if path else 'CAPTURE'
                if distance is not None:
                    message = 'Быстро и максимально широко раскрывайте, затем смыкайте большой и указательный пальцы.'
            else:
                elapsed = 0.0
                state = 'PREPARING'
            last_image = _preview(rgb)
            latest = {'image': last_image, 'message': message, 'state': state,
                      'progress': min(elapsed / duration, 1), 'distance': distance}
            bounds = (record_start, record_end, None, CAMERA_STALL_TIMEOUT)
            _packet(connection, latest=latest, samples=samples, bounds=bounds, phase='hand inference')
        if stop.is_set():
            reason = 'stopped'
    except InterruptedError:
        reason = 'stopped'
    except Exception as exc:
        error, reason = str(exc) or type(exc).__name__, 'error'
    finally:
        for resource, method in ((capture, 'release'), (model, 'close')):
            if resource is not None:
                try:
                    phase('resource ' + method)
                    getattr(resource, method)()
                except Exception as exc:
                    error, reason = error or str(exc), 'error'
        latest = {'image': last_image, 'message': '', 'state': 'COMPLETE', 'progress': 1} if last_image is not None else None
        bounds = (record_start, record_end, None, CAMERA_STALL_TIMEOUT)
        _packet(connection, latest=latest, bounds=bounds, phase='resource close', done=True, error=error, reason=reason)
        connection.close()


class MotorCaptureWorker(ProcessCaptureWorker):
    def __init__(self, path='', parent=None, duration=15.0, preparation_duration=2.0):
        super().__init__(duration, parent, preparation_duration,
                         process_target=_motor_process, process_args=(str(path),))
