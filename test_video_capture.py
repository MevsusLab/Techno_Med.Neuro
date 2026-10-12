import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import subprocess
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication

from app import AACPage, FacialPage, MainWindow, RPPGPage, TremorPage, VoicePage
from capture_worker import ProcessCaptureWorker
from rppg_engine import RPPGEngine
from test_facial_engine import symmetric_face
from test_rppg_capture import roi_face
from test_voice import voice_signal
from tremor_engine import TremorEngine
from video_capture import (VideoCaptureWorker, _extract_voice, _run_video_capture,
                           source_timestamp, validate_video)


def mocked_supervised_video_child(connection, stop, duration, preparation, path, mode):
    detector = Mock()
    detector.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[])
    with patch('video_capture.ensure_model'), patch('video_capture.HandLandmarkerOptions'), \
         patch('video_capture.BaseOptions'), patch('video_capture.HandLandmarker.create_from_options', return_value=detector):
        _run_video_capture(connection, stop, path, mode)


class CollectingConnection:
    def __init__(self):
        self.packets = 0
        self.samples = []
        self.latest = None
        self.final = None

    def send(self, packet):
        self.packets += 1
        latest, samples, bounds, phase, started, updates, done, error, reason = packet
        self.samples.extend(samples)
        if latest is not None:
            self.latest = latest
        if done:
            self.final = packet


def fake_video_landmarks(points, scale=.55):
    return [SimpleNamespace(x=float(point[0] * scale / 320),
                            y=float(point[1] * scale / 240),
                            z=float(point[2] * scale / 320)) for point in points]


class VideoCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])
        cls.temp = tempfile.TemporaryDirectory(prefix='ndtp_video_test_')
        cls.root = Path(cls.temp.name)
        cls.video = cls.root / 'all_modules.avi'
        cls._write_video(cls.video, 30, 30)
        cls.audio_video = cls.root / 'voice_track.avi'
        cls._write_video(cls.audio_video, 6, 15, size=(160, 120))
        cls.silent_video = cls.audio_video
        import soundfile as sf
        wav = cls.root / 'voice.wav'
        sf.write(wav, voice_signal(), 16000)
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-i', str(cls.audio_video),
                        '-i', str(wav), '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'pcm_s16le',
                        '-shortest', '-y', str(cls.root / 'with_audio.avi')], check=True, timeout=30)
        cls.audio_video = cls.root / 'with_audio.avi'

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def _write_video(path, duration, fps, size=(320, 240)):
        width, height = size
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
        if not writer.isOpened():
            raise RuntimeError('OpenCV MJPG encoder unavailable in test environment')
        y, x = np.indices((height, width))
        checker = ((x // 3 + y // 3) % 2 * 20).astype(np.uint8)
        for frame in range(int(duration * fps)):
            image = np.empty((height, width, 3), np.uint8)
            image[:, :, 0] = 80 + checker
            image[:, :, 1] = 110 + checker
            image[:, :, 2] = 135 + checker
            writer.write(image)
        writer.release()

    def test_file_validation_formats_durations_fps_and_size(self):
        path, fps, frames, duration = validate_video(self.video, 'facial')
        self.assertEqual(path, self.video.resolve())
        self.assertAlmostEqual(fps, 30, delta=.1)
        self.assertEqual(frames, 900)
        self.assertAlmostEqual(duration, 30, delta=.1)
        short = self.root / 'short.avi'
        self._write_video(short, 4, 30, (160, 120))
        with self.assertRaisesRegex(ValueError, 'не менее 15'):
            validate_video(short, 'facial')
        with self.assertRaisesRegex(ValueError, 'Выберите видеофайл'):
            validate_video(self.root / 'missing.mp4', 'tremor')
        with self.assertRaisesRegex(ValueError, 'MP4'):
            validate_video(self.root / 'all_modules.txt', 'tremor')

    def test_nonmonotonic_or_missing_container_pts_uses_real_frame_clock(self):
        period = 1 / 30
        times = [source_timestamp(ms, index, period, previous)
                 for ms, index, previous in ((0, 0, -1), (0, 1, 1e-6),
                                             (0, 2, 1 / 30 + 1e-6),
                                             (200, 3, 2 / 30 + 1e-6))]
        self.assertTrue(np.all(np.diff(times) > 0))
        self.assertAlmostEqual(times[1] - times[0], period)
        self.assertAlmostEqual(times[2] - times[1], period)
        self.assertAlmostEqual(times[3], .2)

    def test_tremor_video_uses_existing_landmark_engine_and_original_timestamps(self):
        detector = Mock()
        tip = SimpleNamespace(x=.5, y=.5)
        landmarks = [SimpleNamespace(x=.5, y=.5) for _ in range(21)]
        landmarks[8] = tip
        detector.detect_for_video.side_effect = lambda image, ms: SimpleNamespace(hand_landmarks=[landmarks])
        connection = CollectingConnection()
        with patch('video_capture.ensure_model'), patch('video_capture.HandLandmarkerOptions'), \
             patch('video_capture.BaseOptions'), patch('video_capture.HandLandmarker.create_from_options', return_value=detector):
            _run_video_capture(connection, Mock(is_set=Mock(return_value=False)), str(self.video), 'tremor')
        self.assertTrue(connection.final[6])
        self.assertEqual(connection.final[8], 'deadline')
        self.assertEqual(len(connection.samples), 450)
        self.assertAlmostEqual(connection.samples[-1][0], 14.966, delta=.08)
        self.assertEqual(connection.latest[0].shape, (240, 320, 3))
        engine = TremorEngine()
        for timestamp, x, y in connection.samples:
            engine.add_sample(x, y, timestamp)
        self.assertGreaterEqual(engine.count, 400)
        detector.close.assert_called_once()

    def test_supervised_video_worker_uses_isolated_child_and_completes(self):
        import multiprocessing
        before = {child.pid for child in multiprocessing.active_children()}
        worker = ProcessCaptureWorker(duration=600, process_target=mocked_supervised_video_child,
                                      process_args=(str(self.video), 'tremor'))
        worker.start()
        self.assertTrue(worker.wait(30000), 'Video-file worker did not finish.')
        self.assertEqual(worker.error, '')
        self.assertEqual(worker.reason, 'deadline')
        latest, samples = worker.take_pending()
        self.assertGreaterEqual(len(samples), 400)
        self.assertEqual(latest[0].shape, (240, 320, 3))
        self.assertEqual({child.pid for child in multiprocessing.active_children()}, before)

    def test_facial_video_splits_three_five_second_expression_stages(self):
        from video_capture import _face_model
        model = Mock()

        def infer(image, timestamp_ms):
            t = timestamp_ms / 1000
            points = symmetric_face()
            if 5 <= t < 10:
                points[[61, 78], 0] -= 24
                points[[291, 308], 0] += 24
                points[[61, 78, 291, 308], 1] -= 16
            elif 10 <= t < 15:
                points[[70, 105, 300, 334], 1] -= 20
            return SimpleNamespace(face_landmarks=[fake_video_landmarks(points)],
                                   facial_transformation_matrixes=[np.eye(4)])

        model.detect_for_video.side_effect = infer
        connection = CollectingConnection()
        with patch('video_capture._face_model', return_value=model):
            _run_video_capture(connection, Mock(is_set=Mock(return_value=False)), str(self.video), 'facial')
        latest, _, *_ = connection.final
        self.assertEqual(connection.final[8], 'deadline')
        self.assertEqual(latest['stage_bounds'], {'rest': (2.0, 5.0), 'smile': (7.0, 10.0), 'brows': (12.0, 15.0)})
        self.assertEqual(latest['counts'], {'rest': 90, 'smile': 90, 'brows': 90})
        self.assertEqual(latest['rejected'], {'rest': 0, 'smile': 0, 'brows': 0})
        self.assertEqual(len(connection.samples), 270)
        self.assertEqual({sample['stage'] for sample in connection.samples}, {'rest', 'smile', 'brows'})
        self.assertLess(max(sample['timestamp'] for sample in connection.samples if sample['stage'] == 'rest'), 5)
        self.assertTrue(latest['image'].shape[0] <= 240)

    def test_rppg_video_streams_timed_skin_samples_and_quality_bounds(self):
        model = Mock()
        points = roi_face()
        model.detect_for_video.return_value = SimpleNamespace(
            face_landmarks=[fake_video_landmarks(points)], facial_transformation_matrixes=[np.eye(4)])
        connection = CollectingConnection()
        with patch('video_capture._face_model', return_value=model):
            _run_video_capture(connection, Mock(is_set=Mock(return_value=False)), str(self.video), 'rppg')
        self.assertEqual(connection.final[8], 'deadline')
        self.assertEqual(connection.final[2][:2], (0.0, 30.0))
        self.assertEqual(len(connection.samples), 900)
        self.assertTrue(np.all(np.diff([sample['timestamp'] for sample in connection.samples]) > 0))
        self.assertEqual(connection.samples[0]['rgb'].shape, (3, 3))
        self.assertEqual(connection.latest['frames'], 900)
        model.close.assert_called_once()

    def test_voice_video_extracts_audio_and_runs_existing_voice_analysis(self):
        connection = CollectingConnection()
        result = _extract_voice(connection, threading.Event(), str(self.audio_video))
        self.assertEqual(result['source'], 'video')
        self.assertEqual(result['file_name'], self.audio_video.name)
        self.assertAlmostEqual(result['f0'], 180, delta=1)
        self.assertAlmostEqual(result['source_duration_sec'], 5, delta=.1)
        self.assertAlmostEqual(result['segment_start_sec'], 0, delta=.05)

    def test_video_capture_worker_supervises_video_audio_extraction_and_analysis(self):
        import multiprocessing
        before = {child.pid for child in multiprocessing.active_children()}
        worker = VideoCaptureWorker(str(self.audio_video), 'voice')
        worker.start()
        self.assertTrue(worker.wait(60000), 'Video voice worker did not finish.')
        self.assertEqual(worker.error, '')
        self.assertEqual(worker.reason, 'deadline')
        latest, samples = worker.take_pending()
        self.assertEqual(samples, [])
        self.assertEqual(latest['state'], 'COMPLETE')
        self.assertEqual(latest['result']['source'], 'video')
        self.assertEqual(latest['result']['file_name'], self.audio_video.name)
        self.assertAlmostEqual(latest['result']['f0'], 180, delta=1)
        self.assertEqual({child.pid for child in multiprocessing.active_children()}, before)

    def test_voice_video_without_audio_track_reports_specific_error(self):
        connection = CollectingConnection()
        with self.assertRaises(ValueError) as caught:
            _extract_voice(connection, threading.Event(), str(self.silent_video))
        self.assertTrue('видео' in str(caught.exception).lower()
                        or 'аудио' in str(caught.exception).lower())

    def test_video_source_name_is_included_in_analysis_reports(self):
        from protocol_panel import FacialProtocolPanel, RPPGProtocolPanel, VoiceProtocolPanel
        tremor = TremorPage()
        face = FacialProtocolPanel()
        pulse = RPPGProtocolPanel()
        voice = VoiceProtocolPanel()
        try:
            tremor_result = {'dominant_freq': 5., 'rms_px': 1., 'concentration': 50., 'fs': 30.,
                             'coverage': .95, 'max_gap_sec': .04, 'max_freq': 12., 'source': 'video',
                             'source_file': 'sample.avi'}
            tremor.protocol_panel.set_results(tremor_result)
            self.assertIn('sample.avi', tremor.protocol_panel.report)
            facial_stage = {'samples': 60, 'rate': 30., 'width_px': 150.}
            facial = {'asymmetry_index': 5., 'smile_imbalance': 5., 'brow_imbalance': 5.,
                      'rest': facial_stage, 'smile': facial_stage, 'brows': facial_stage}
            facial.update(source='video', source_file='sample.avi')
            face.set_results(facial)
            self.assertIn('sample.avi', face.report)
            result = {'bpm': 72., 'concentration': 80., 'coverage': .95,
                      'roi_bpm': [72., 72., 72.], 'used_rois': [0, 1, 2],
                      'samples': 900, 'rate': 30., 'duration': 30., 'max_gap': .034,
                      'resolution_bpm': 2., 'warnings': [], 'time': np.arange(20),
                      'pulse': np.sin(np.arange(20)), 'roi_signals': np.zeros((20, 3)),
                      'frequencies': np.arange(10), 'spectrum': np.arange(10),
                      'motion_span': [0.] * 6, 'source': 'video', 'source_file': 'sample.avi'}
            result.update(source='video', source_file='sample.avi')
            pulse.set_results(result)
            self.assertIn('sample.avi', pulse.report)
            voice_result = {'jitter_local': .5, 'shimmer_local': 2., 'voiced_fraction': .99,
                            'level_dbfs': -18., 'warnings': [], 'f0': 180., 'jitter_rap': .3,
                            'shimmer_apq5': 1., 'hnr': 20., 'sample_rate': 16000, 'duration': 5.,
                            'clipping': 0., 'mfcc_mean': [0.] * 13, 'source': 'video',
                            'file_name': 'sample.avi', 'segment_start_sec': 3., 'source_duration_sec': 8.}
            voice.set_results(voice_result)
            self.assertIn('Видео: sample.avi', voice.report)
            self.assertIn('3,00 с', voice.report)
        finally:
            tremor.close()
            face.close()
            pulse.close()
            voice.close()

    def test_page_upload_buttons_bind_to_video_workers_and_modes(self):
        for page, start, button, mode in ((TremorPage(), '_start_source', 'btn_video', 'tremor'),
                                          (FacialPage(), '_start_source', 'upload_video', 'facial'),
                                          (RPPGPage(), '_start_source', 'upload_video', 'rppg')):
            try:
                fake = Mock()
                with patch('app.VideoCaptureWorker', return_value=fake) as factory:
                    getattr(page, start)(str(self.video))
                factory.assert_called_once_with(str(self.video), mode, page)
                self.assertEqual(fake.start.call_count, 1)
                self.assertFalse(getattr(page, button).isEnabled())
                self.assertEqual(page.source_kind if hasattr(page, 'source_kind') else page._source_kind, 'video')
                self.assertEqual(page.source_path if hasattr(page, 'source_path') else page._source_path, str(self.video))
                page._stop(analyze=False) if mode == 'tremor' else page._stop()
            finally:
                page.cleanup()
                page.close()

        voice = VoicePage()
        try:
            fake = Mock()
            with patch('app.VideoCaptureWorker', return_value=fake) as factory:
                voice._begin(factory, str(self.audio_video), 'voice')
            factory.assert_called_once_with(str(self.audio_video), 'voice', parent=voice)
            fake.start.assert_called_once()
            self.assertFalse(voice.upload.isEnabled())
        finally:
            voice.cleanup()
            voice.close()


if __name__ == '__main__':
    unittest.main()
