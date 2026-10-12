"""Isolated video-file sources for the camera-based research modules.

All file decoding and MediaPipe inference run in the existing supervised child
process. Packets use ProcessCaptureWorker's established latest-frame / scalar
sample contract, so the existing module analysis and quality gates are reused.
"""

from pathlib import Path
import os
import shutil
import sys
import subprocess
import tempfile
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, HandLandmarker, HandLandmarkerOptions, RunningMode

from capture_worker import HAND_MODEL_PATH, ProcessCaptureWorker, ensure_model
from face_capture import MODEL_PATH, ensure_face_model, head_pose
from facial_engine import normalize_face
from rppg_capture import extract_regions
from voice_dependencies import load_voice_file_dependencies


VIDEO_EXTENSIONS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.wmv', '.webm'}
MAX_VIDEO_BYTES = 500 * 1024 * 1024
MAX_VIDEO_SECONDS = 600.0
PREVIEW_WIDTH = 320
VIDEO_FILTER = 'Видеофайлы (*.mp4 *.mov *.m4v *.avi *.mkv *.wmv *.webm);;Все файлы (*.*)'
RECORDING_SECONDS = {'tremor': 15.0, 'facial': 15.0, 'rppg': 30.0}


def validate_video(path, mode):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError('Выберите видеофайл MP4, MOV, M4V, AVI, MKV, WMV или WEBM.')
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_VIDEO_BYTES:
        raise ValueError('Видео пустое или больше допустимого размера 500 МБ.')
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('Не удалось открыть видеофайл. Проверьте формат и кодек.')
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frames / fps if fps > 0 and frames > 0 else 0.0
    finally:
        capture.release()
    if not np.isfinite([fps, frames, duration]).all() or fps <= 0 or frames <= 0:
        raise ValueError('В метаданных видео отсутствует частота кадров или длительность.')
    # Tremor requires >10s of sampled trajectory; leave room for the final
    # frame being one source interval before a nominal 10s container duration.
    minimum = {'tremor': 10.5, 'facial': 15.0, 'rppg': 20.0, 'voice': 5.0, 'motor': 10.0}[mode]
    if duration < minimum:
        raise ValueError(f'Видео длится {duration:.1f} с; для этого анализа требуется не менее {minimum:g} с.')
    if duration > MAX_VIDEO_SECONDS:
        raise ValueError('Длительность видео превышает ограничение 10 минут.')
    min_rate = 25.0 if mode in ('tremor', 'motor') else 15.0
    if fps < min_rate:
        raise ValueError(f'Частота видео {fps:.1f} кадр/с; требуется не менее {min_rate:.0f} кадр/с.')
    return path, fps, frames, duration


def check_video_audio_dependency():
    """Confirm the bundled FFmpeg executable is available before starting UI work."""
    external = shutil.which('ffmpeg')
    try:
        import imageio_ffmpeg
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except RuntimeError:
            if external:
                return external
            raise
    except Exception as exc:
        if external:
            return external
        raise RuntimeError('Для извлечения звука из видео установите imageio-ffmpeg из requirements.txt.') from exc


def _preview(rgb):
    height, width = rgb.shape[:2]
    if width > PREVIEW_WIDTH:
        rgb = cv2.resize(rgb, (PREVIEW_WIDTH, max(1, round(height * PREVIEW_WIDTH / width))),
                         interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)


def source_timestamp(position_ms, frame_index, frame_period, previous=-1.0):
    """Use container PTS when monotonic; fall back to frame-rate timing if not."""
    candidate = float(position_ms) / 1000
    fallback = frame_index * frame_period
    timestamp = (candidate if np.isfinite(candidate) and candidate > previous + frame_period * 0.5
                 else fallback)
    return max(timestamp, previous + 1e-6)


def _packet(connection, latest=None, samples=None, bounds=None, phase='video decoding',
            started=None, done=False, error='', reason='stopped', updates=None):
    connection.send((latest, samples or [], bounds or (None, None, None, 2.0),
                     phase, started or time.perf_counter(), updates or [], done, error, reason))


def _face_model(cancelled, faces=1):
    ensure_face_model(cancelled)
    if cancelled():
        raise InterruptedError()
    return FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH), running_mode=RunningMode.VIDEO,
        num_faces=faces, min_face_detection_confidence=0.6,
        output_facial_transformation_matrixes=True,
    ))


def _run_video_capture(connection, stop, path, mode):
    capture = model = None
    reason = 'error'
    started = time.perf_counter()
    count = 0
    valid = 0
    rate = 0.0
    duration = 0.0
    error = ''
    latest = None
    batch = []
    bounds = (None, None, None, 2.0)
    phase = 'video decoding'
    phase_started = time.perf_counter()
    facial_counts = {'rest': 0, 'smile': 0, 'brows': 0}
    facial_rejected = {'rest': 0, 'smile': 0, 'brows': 0}
    facial_bounds = {'rest': (2.0, 5.0), 'smile': (7.0, 10.0), 'brows': (12.0, 15.0)}
    last_preview = None
    try:
        source, rate, _, duration = validate_video(path, mode)
        record_duration = min(duration, RECORDING_SECONDS.get(mode, duration))
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise OSError('Не удалось открыть видеофайл.')
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if mode == 'tremor':
            if stop.is_set():
                raise InterruptedError()
            ensure_model(stop.is_set)
            model = HandLandmarker.create_from_options(HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
                running_mode=RunningMode.VIDEO, num_hands=1, min_hand_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            ))
            phase = 'hand inference'
        else:
            model = _face_model(stop.is_set, faces=1 if mode == 'rppg' else 2)
            phase = 'face inference'
        if mode == 'rppg':
            record_start, record_end = 0.0, record_duration
        elif mode == 'tremor':
            record_start, record_end = 0.0, record_duration
        else:
            record_start, record_end = None, None
        timestamps_ms = 0
        frame_index = 0
        frame_period = 1 / rate
        bounds = (record_start, record_end, None, 2.0)
        last_timestamp = -1.0
        while not stop.is_set():
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = source_timestamp(capture.get(cv2.CAP_PROP_POS_MSEC), frame_index,
                                         frame_period, last_timestamp)
            last_timestamp = timestamp
            frame_index += 1
            if timestamp >= record_duration:
                break
            if mode == 'tremor' and timestamp >= 15.0:
                break
            h, w = frame.shape[:2]
            if mode == 'tremor':
                rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                timestamps_ms = max(timestamps_ms + 1, int(timestamp * 1000))
                result = model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamps_ms)
                x = y = None
                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    tip = landmarks[8]
                    x, y = float(tip.x * w), float(tip.y * h)
                    cv2.circle(rgb, (int(x), int(y)), 7, (246, 130, 59), 2)
                    cv2.circle(rgb, (int(x), int(y)), 3, (246, 130, 59), -1)
                    valid += 1
                batch.append((timestamp, x, y))
                latest = (_preview(rgb), timestamp, x, y)
                bounds = (record_start, min(timestamp, record_end), None, 2.0)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamps_ms = max(timestamps_ms + 1, int(timestamp * 1000))
                result = model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamps_ms)
                points = pose = None
                if len(result.face_landmarks) == 1 and result.facial_transformation_matrixes:
                    candidate = np.array([[p.x * w, p.y * h, p.z * w] for p in result.face_landmarks[0]])
                    try:
                        pose = head_pose(result.facial_transformation_matrixes[0])
                        points = candidate
                    except ValueError:
                        pass
                if mode == 'facial':
                    stage = next((name for name, (a, b) in facial_bounds.items() if a <= timestamp < b), None)
                    if stage is not None:
                        facial_counts[stage] += 1
                        if points is not None:
                            try:
                                normalize_face(points, pose)
                                x0, y0 = np.maximum(points[:468, :2].min(axis=0).astype(int), 0)
                                x1, y1 = np.minimum(points[:468, :2].max(axis=0).astype(int), [w - 1, h - 1])
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
                                if x0 < 5 or y0 < 5 or x1 > w - 6 or y1 > h - 6:
                                    raise ValueError('Keep the whole face inside the video frame.')
                                if gray.size == 0 or gray.mean() < 35 or gray.mean() > 235 or cv2.Laplacian(gray, cv2.CV_64F).var() < 12:
                                    raise ValueError('Face image is dark, overexposed, or blurred.')
                                batch.append({'stage': stage, 'timestamp': timestamp,
                                              'landmarks': points, 'pose': pose})
                                valid += 1
                            except ValueError:
                                facial_rejected[stage] += 1
                        else:
                            facial_rejected[stage] += 1
                    message = 'Видео: покой (0–5 с), улыбка (5–10 с), брови (10–15 с).'
                    latest = {'image': _preview(rgb), 'message': message, 'stage': stage or 'preview',
                              'progress': min(timestamp / 15, 1), 'stage_bounds': facial_bounds,
                              'rejected': facial_rejected, 'counts': facial_counts}
                else:
                    if points is not None:
                        try:
                            colors, polygons, background, center, span = extract_regions(frame, points, pose)
                            geometry = np.array([*(center / span), span / min(w, h), *pose])
                            batch.append({'timestamp': timestamp, 'rgb': colors,
                                          'background': background, 'geometry': geometry})
                            valid += 1
                            for polygon in polygons:
                                cv2.polylines(rgb, [polygon], True, (80, 190, 255), 1)
                        except ValueError:
                            pass
                    latest = {'image': _preview(rgb), 'message': 'Видео: анализ областей кожи лица.',
                              'frames': frame_index, 'start': 0.0, 'end': timestamp,
                              'progress': min(timestamp / record_duration, 1.0)}
                    bounds = (0.0, min(timestamp, record_end), None, 2.0)
            count += 1
            phase_started = time.perf_counter()
            if latest is not None:
                last_preview = latest['image'] if isinstance(latest, dict) else latest[0]
            _packet(connection, latest, batch, bounds, phase, phase_started, updates=['VIDEO FILE'])
            batch = []
            latest = None
            if frame_index % 120 == 0:
                connection.send((None, [], bounds, 'video decoding', time.perf_counter(),
                                 ['VIDEO FILE'], False, '', 'stopped'))
        if stop.is_set():
            reason = 'stopped'
        elif mode == 'facial':
            reason = 'deadline'
            # Publish fixed protocol-stage bounds for FacialPage._finished.
            latest = {'image': last_preview if last_preview is not None else np.zeros((1, 1, 3), dtype=np.uint8), 'message': 'Видео обработано.',
                      'stage': 'complete', 'progress': 1.0, 'stage_bounds': facial_bounds,
                      'counts': facial_counts, 'rejected': facial_rejected}
            bounds = (None, None, None, 2.0)
        else:
            reason = 'deadline'
            if mode == 'rppg':
                bounds = (0.0, record_duration, None, 2.0)
    except InterruptedError:
        reason = 'stopped'
    except Exception as exc:
        error, reason = f'{type(exc).__name__}: {exc}', 'error'
    finally:
        for resource, method in ((capture, 'release'), (model, 'close')):
            if resource is not None:
                try:
                    getattr(resource, method)()
                except Exception:
                    pass
    _packet(connection, latest, batch, bounds, 'resource release', time.perf_counter(),
            done=True, error=error, reason=reason)


def _extract_voice(connection, stop, path):
    from voice_engine import analyze_voice, load_audio_file
    _voice_status(connection, 'voice dependencies', 'Проверка библиотек анализа голоса.')
    if stop.is_set():
        return None
    source, _, _, duration = validate_video(path, 'voice')
    ffmpeg = check_video_audio_dependency()
    with tempfile.TemporaryDirectory(prefix='tecnomed_video_audio_') as directory:
        output = Path(directory) / 'video_audio.wav'
        command = [ffmpeg, '-nostdin', '-v', 'error', '-threads', '1', '-filter_threads', '1',
                   '-i', str(source), '-vn', '-ac', '1',
                   '-ar', '16000', '-t', str(MAX_VIDEO_SECONDS), '-f', 'wav', '-y', str(output)]
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            last_status = time.perf_counter()
            while process.poll() is None:
                if stop.wait(0.05):
                    process.terminate()
                    try:
                        process.wait(2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise InterruptedError()
                if time.perf_counter() - last_status >= 1:
                    _voice_status(connection, 'audio file loading', 'Извлечение аудиодорожки видеофайла.')
                    last_status = time.perf_counter()
            if process.returncode:
                raise ValueError('Не удалось извлечь аудиодорожку из видео. Проверьте кодек и наличие звука.')
            if not output.is_file() or output.stat().st_size <= 44:
                raise ValueError('В видеофайле не найдена аудиодорожка.')
            # Decode/extract first: loading librosa/Praat can reserve substantial
            # memory, while FFmpeg also needs working memory for video codecs.
            _voice_status(connection, 'voice dependencies', 'Подготовка библиотек анализа голоса.')
            load_voice_file_dependencies()
            audio, rate, metadata = load_audio_file(output)
            _voice_status(connection, 'voice analysis', 'Анализ звуковой дорожки видео.')
            result = analyze_voice(audio, rate)
            result.update(source='video', file_name=source.name,
                          segment_start_sec=metadata.get('segment_start_sec', 0.0),
                          source_duration_sec=duration)
            return result
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def _voice_status(connection, phase, message):
    latest = {'state': 'PROCESSING', 'elapsed': 0, 'level': 0, 'message': message}
    connection.send((latest, [], (None, None, None, 3.0), phase,
                     time.perf_counter(), [], False, '', 'stopped'))


def _video_process(connection, stop, duration, preparation, path, mode):
    if mode == 'voice':
        try:
            _voice_status(connection, 'voice dependencies', 'Извлечение звуковой дорожки видеофайла.')
            result = _extract_voice(connection, stop, path)
            if stop.is_set():
                latest, error, reason = None, '', 'stopped'
            else:
                latest, error, reason = ({'state': 'COMPLETE', 'elapsed': 5, 'level': 0,
                                          'message': 'Акустический анализ видео завершён.', 'result': result},
                                         '', 'deadline')
            connection.send((latest, [], (None, None, None, 3), 'voice analysis', time.perf_counter(),
                              [], True, error, reason))
        except Exception as exc:
            connection.send((None, [], (None, None, None, 3), 'audio file loading', time.perf_counter(),
                             [], True, f'{type(exc).__name__}: {exc}', 'error'))
        finally:
            connection.close()
        return
    _run_video_capture(connection, stop, path, mode)
    connection.close()


class VideoCaptureWorker(ProcessCaptureWorker):
    """Video file source supervised exactly like the existing live captures."""

    def __init__(self, path, mode, parent=None):
        if mode not in {'tremor', 'facial', 'rppg', 'voice'}:
            raise ValueError(f'Unknown video analysis mode: {mode}')
        super().__init__(duration=MAX_VIDEO_SECONDS, parent=parent,
                         process_target=_video_process, process_args=(str(path), mode))
