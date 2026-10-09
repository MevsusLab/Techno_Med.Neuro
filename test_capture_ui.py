import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
import app
import capture_worker


class FakeWorker(QObject):
    status = pyqtSignal(str)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, duration, parent=None):
        super().__init__(parent)
        self.error = ""
        self.reason = "deadline"
        self.start = Mock()
        self.deleteLater = Mock()
        self.interrupted = False
        self.requestInterruption = Mock(side_effect=self._interrupt)
        self.bounds = (None, None, None, 10.0)
        self.timeout = 10.0
        self.samples = []
        self.latest = None

    def _interrupt(self):
        self.interrupted = True

    def isInterruptionRequested(self):
        return self.interrupted

    def snapshot(self):
        return (*self.bounds[:3], self.timeout)

    def take_pending(self):
        latest, samples = self.latest, self.samples
        self.latest = None
        self.samples = []
        return latest, samples


class CaptureUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])
        cls.qt.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.factory_patch = patch("app.CaptureWorker", side_effect=FakeWorker)
        self.factory = self.factory_patch.start()
        self.clock_patch = patch("app.time.perf_counter", return_value=100.0)
        self.clock = self.clock_patch.start()
        self.status = app.StatusBar()
        self.page = app.TremorPage(self.status)
        self.page.engine.analyze = Mock(return_value=None)

    def tearDown(self):
        self.page.cleanup()
        if self.page.worker is not None:
            self.page.worker.finished.emit()
        self.page.close()
        self.page.deleteLater()
        self.status.deleteLater()
        self.qt.processEvents()
        self.clock_patch.stop()
        self.factory_patch.stop()

    def start_capture(self):
        self.page.btn_start.click()
        self.page.timer.stop()
        return self.page.worker

    def acquire(self, worker, start=100.0, end=100.2):
        worker.bounds = (start, end, None)
        worker.samples = [(start, None, None), (end, 12.0, 24.0)]
        worker.latest = (np.zeros((12, 16, 3), dtype=np.uint8), end, 12.0, 24.0)
        self.clock.return_value = end
        self.page._on_frame()

    def test_start_and_stop_clicks_ignore_signal_boolean(self):
        worker = self.start_capture()
        worker.start.assert_called_once_with()
        self.assertEqual(self.page.page_badge.text(), "STARTING")
        self.assertFalse(self.page.recording)
        self.assertIsNone(self.page.record_start)
        self.acquire(worker)
        self.clock.return_value = 105.0
        self.page.btn_stop.click()
        self.assertTrue(self.page._analyze_on_finish)
        self.assertFalse(self.page.btn_start.isEnabled())
        self.page.engine.analyze.assert_not_called()
        worker.finished.emit()
        self.page.engine.analyze.assert_called_once_with(record_start=100.0, record_end=105.0)
        self.assertTrue(self.page.btn_start.isEnabled())

    def test_setup_time_is_not_recording_time(self):
        worker = self.start_capture()
        self.clock.return_value = 200.0
        worker.status.emit("CAM READY")
        self.page._on_frame()
        self.assertIsNone(self.page.record_start)
        self.assertEqual(self.page.progress.value(), 0)
        self.assertFalse(worker.interrupted)
        self.acquire(worker, 200.0, 200.2)
        self.assertTrue(self.page.recording)
        self.assertEqual(self.page.record_start, 200.0)
        self.assertLess(self.page.progress.value(), 2)

    def test_stop_during_setup_cancels_without_analysis(self):
        worker = self.start_capture()
        self.page.btn_stop.click()
        self.assertTrue(worker.interrupted)
        self.assertFalse(self.page.btn_start.isEnabled())
        worker.finished.emit()
        self.page.engine.analyze.assert_not_called()
        self.assertEqual(self.page.page_badge.text(), "IDLE")
        self.assertTrue(self.page.btn_start.isEnabled())

    def test_reset_discards_pending_samples_and_clears_all_metrics(self):
        worker = self.start_capture()
        self.acquire(worker)
        for metric in [self.page.m_freq, self.page.m_amp, self.page.m_class, self.page.m_conf]:
            metric.set_value("old")
        self.status.update_badge("DSP", "COMPLETE", app.GREEN)
        self.page.btn_reset.click()
        worker.samples = [(100.3, 5.0, 6.0)]
        worker.status.emit("CAM READY")
        worker.error = "Late backend error"
        worker.failed.emit(worker.error)
        self.page._on_frame()
        self.assertFalse(self.page.btn_start.isEnabled())
        worker.finished.emit()
        self.page.engine.analyze.assert_not_called()
        self.assertEqual(self.page.engine.count, 0)
        self.assertIsNone(self.page.record_start)
        self.assertIsNone(self.page.record_end)
        self.assertEqual(self.page.progress.value(), 0)
        self.assertEqual(self.page.page_badge.text(), "IDLE")
        self.assertEqual(self.page.analysis_badge.text(), "WAITING")
        self.assertEqual(self.status.badges["DSP"].text(), "STANDBY")
        for metric in [self.page.m_freq, self.page.m_amp, self.page.m_class, self.page.m_conf]:
            self.assertEqual(metric.val.text(), "---")

    def test_capture_failure_disables_restart_until_finished(self):
        worker = self.start_capture()
        self.acquire(worker)
        worker.error = "Camera disconnected"
        worker.failed.emit(worker.error)
        self.assertTrue(worker.interrupted)
        self.assertFalse(self.page.btn_start.isEnabled())
        self.page._start()
        self.factory.assert_called_once()
        worker.samples = [(100.3, 5.0, 6.0)]
        worker.finished.emit()
        self.page.engine.analyze.assert_not_called()
        self.assertEqual(self.page.engine.count, 0)
        self.assertEqual(self.page.analysis_badge.text(), "CAPTURE FAILED")
        self.assertTrue(self.page.btn_start.isEnabled())
        self.start_capture()
        self.assertEqual(self.factory.call_count, 2)

    def test_finished_drains_all_scalar_samples_and_releases_worker(self):
        worker = self.start_capture()
        self.acquire(worker)
        worker.bounds = (100.0, 114.95, None)
        worker.samples = [(114.9, 1.0, 2.0), (114.95, None, None)]
        self.clock.return_value = 115.0
        finished = Mock()
        self.page.capture_finished.connect(finished)
        worker.finished.emit()
        self.assertEqual(self.page.engine.timestamps, [100.2, 114.9])
        self.page.engine.analyze.assert_called_once_with(record_start=100.0, record_end=114.95)
        worker.deleteLater.assert_called_once_with()
        finished.assert_called_once_with()
        self.assertIsNone(self.page.worker)
        self.assertFalse(self.page.timer.isActive())
        self.assertFalse(self.page.recording)
        self.assertFalse(self.page.btn_stop.isEnabled())
        self.assertEqual(self.status.badges["CAM"].text(), "DISCONNECTED")
        self.assertEqual(self.status.badges["MEDIAPIPE"].text(), "NOT LOADED")

    def test_manual_stop_excludes_frames_acquired_after_stop(self):
        worker = self.start_capture()
        self.acquire(worker)
        self.clock.return_value = 105.0
        self.page.btn_stop.click()
        worker.samples = [(104.9, 1.0, 2.0), (105.1, 3.0, 4.0)]
        worker.bounds = (100.0, 105.1, None)
        worker.finished.emit()
        self.assertEqual(self.page.engine.timestamps, [100.2, 104.9])
        self.page.engine.analyze.assert_called_once_with(record_start=100.0, record_end=105.0)

    def test_deadline_watchdog_uses_last_acquired_frame_not_timer_time(self):
        worker = self.start_capture()
        self.acquire(worker)
        worker.bounds = (100.0, 114.95, None)
        self.clock.return_value = 115.05
        self.page._on_frame()
        self.assertTrue(worker.interrupted)
        self.assertIsNone(self.page._stop_time)
        self.assertFalse(self.page.btn_start.isEnabled())
        worker.finished.emit()
        self.page.engine.analyze.assert_called_once_with(record_start=100.0, record_end=114.95)

    def test_deadline_interruption_race_still_analyzes_completed_capture(self):
        worker = self.start_capture()
        self.acquire(worker)
        worker.bounds = (100.0, 114.95, None)
        worker.reason = "stopped"
        self.clock.return_value = 115.05
        self.page._on_frame()
        worker.finished.emit()
        self.page.engine.analyze.assert_called_once_with(record_start=100.0, record_end=114.95)
        self.assertIsNone(self.page.worker)

    def test_analysis_exception_does_not_escape_completion_or_disable_restart(self):
        worker = self.start_capture()
        self.acquire(worker)
        self.page.engine.analyze.side_effect = TypeError("unexpected keyword argument 'record_start'")
        finished = Mock()
        self.page.capture_finished.connect(finished)
        with patch("app.traceback.print_exc") as logged:
            worker.finished.emit()
        logged.assert_called_once()
        finished.assert_called_once()
        self.assertEqual(self.page.page_badge.text(), "ANALYSIS ERROR")
        self.assertIn("record_start", self.page.analysis_detail.text())
        self.assertTrue(self.page.btn_start.isEnabled())
        self.assertFalse(self.page.timer.isActive())
        self.assertIsNone(self.page.worker)

    def test_repeated_completed_captures_use_real_engine_contract(self):
        self.page.engine = app.TremorEngine()
        for trial in range(5):
            with self.subTest(trial=trial):
                start = 100.0 + trial * 30
                self.clock.return_value = start
                worker = self.start_capture()
                t = start + np.arange(450) / 30
                worker.bounds = (t[0], t[-1], None)
                worker.samples = [(ti, float(10 * np.sin(2 * np.pi * 5 * ti)), 0.0) for ti in t]
                self.clock.return_value = start + 15.0
                worker.finished.emit()
                self.assertEqual(self.page.analysis_badge.text(), "COMPLETE")
                self.assertEqual(self.page.engine.count, 450)
                self.assertAlmostEqual(float(self.page.m_freq.val.text()), 5, delta=0.2)
                self.assertTrue(self.page.btn_start.isEnabled())
                self.assertIsNone(self.page.worker)

    def test_final_frame_processing_exception_still_releases_session(self):
        worker = self.start_capture()
        self.acquire(worker)
        finished = Mock()
        self.page.capture_finished.connect(finished)
        with patch.object(self.page, "_on_frame", side_effect=ValueError("bad frame")), patch("app.traceback.print_exc"):
            worker.finished.emit()
        finished.assert_called_once()
        worker.deleteLater.assert_called_once()
        self.assertIsNone(self.page.worker)
        self.assertTrue(self.page.btn_start.isEnabled())

    def test_failure_in_finished_state_also_prevents_analysis(self):
        worker = self.start_capture()
        self.acquire(worker)
        worker.error = "Resource cleanup failed"
        worker.finished.emit()
        self.page.engine.analyze.assert_not_called()
        self.assertEqual(self.page.analysis_badge.text(), "CAPTURE FAILED")

    def test_watchdog_is_responsive_while_camera_read_blocks(self):
        entered = threading.Event()
        unblock = threading.Event()
        camera = Mock()
        landmarker = Mock()
        def blocked_read():
            entered.set()
            unblock.wait(3)
            return True, np.zeros((12, 16, 3), dtype=np.uint8)
        camera.read.side_effect = blocked_read
        worker = capture_worker.CaptureWorker(parent=self.page)
        worker.preparation_duration = 0
        self.factory.side_effect = None
        self.factory.return_value = worker
        def detect(image, timestamp):
            worker.requestInterruption()
            return SimpleNamespace(hand_landmarks=[])
        landmarker.detect_for_video.side_effect = detect
        with patch("capture_worker.ensure_model"), patch("capture_worker.cv2.VideoCapture", return_value=camera), patch("capture_worker.HandLandmarker.create_from_options", return_value=landmarker):
            try:
                self.start_capture()
                self.assertTrue(entered.wait(1))
                self.clock.return_value = 101.1
                self.page._on_frame()
                self.assertEqual(self.page.analysis_badge.text(), "WAITING")
                self.assertFalse(worker.isInterruptionRequested())
                self.assertEqual(worker.snapshot()[3], capture_worker.CAMERA_STARTUP_TIMEOUT)
                self.assertFalse(self.page.btn_start.isEnabled())
                heartbeat = Mock()
                QTimer.singleShot(0, heartbeat)
                QTest.qWait(20)
                heartbeat.assert_called_once_with()
                camera.release.assert_not_called()
            finally:
                unblock.set()
                self.assertTrue(worker.wait(2000))
                QTest.qWait(20)
        camera.release.assert_called_once_with()
        landmarker.close.assert_called_once_with()
        self.assertIsNone(self.page.worker)
        self.assertEqual(worker.error, "")
        self.assertEqual(self.page.analysis_badge.text(), "INCOMPLETE")
        self.page.engine.analyze.assert_not_called()

    def test_close_is_deferred_until_worker_finishes(self):
        window = app.MainWindow()
        page = window.tremor_page
        page.engine.analyze = Mock(return_value=None)
        window.show()
        page.btn_start.click()
        page.timer.stop()
        worker = page.worker
        try:
            event = QCloseEvent()
            window.closeEvent(event)
            self.assertFalse(event.isAccepted())
            self.assertTrue(worker.interrupted)
            self.assertIs(page.worker, worker)
            self.assertTrue(window.isVisible())
            worker.finished.emit()
            self.qt.processEvents()
            self.assertFalse(window.isVisible())
            self.assertIsNone(page.worker)
            worker.deleteLater.assert_called_once_with()
            page.engine.analyze.assert_not_called()
        finally:
            if page.worker is not None:
                page.worker.finished.emit()
            window.close()
            window.deleteLater()

    def test_neutral_results_accept_missing_peak_and_raw_y(self):
        t = np.linspace(0, 1, 20)
        self.page.engine.analyze.return_value = {
            "dominant_freq": None, "rms_px": 0.2, "concentration": 12.0,
            "max_freq": 12.0, "time": t, "raw_y": t, "filtered_y": t, "filtered_x": t,
            "psd_freqs": t * 12, "psd": t + 0.1,
            "samples": 450, "fs": 30.0, "coverage": 1.0, "resolution_hz": 0.12,
            "max_gap_sec": 1 / 30, "above_band": False,
        }
        self.page.record_start = 100.0
        self.page.record_end = 115.0
        self.page._run_analysis()
        self.assertEqual(self.page.m_freq.val.text(), "---")
        self.assertEqual(self.page.m_amp.val.text(), "0.20")
        self.assertEqual(self.page.m_class.val.text(), "NO CLEAR PEAK")
        self.assertEqual(self.status.badges["DSP"].text(), "COMPLETE")

    def test_real_engine_results_render_after_capture(self):
        self.page.engine = app.TremorEngine()
        worker = self.start_capture()
        t = 100 + np.arange(450) / 30
        worker.bounds = (t[0], t[-1], None)
        worker.samples = [(ti, float(10 * np.sin(2 * np.pi * 5 * ti)), 0.0) for ti in t]
        self.clock.return_value = 115.0
        worker.finished.emit()
        self.assertEqual(self.page.analysis_badge.text(), "COMPLETE")
        self.assertAlmostEqual(float(self.page.m_freq.val.text()), 5, delta=0.2)
        self.assertIn("Покрытие: 100%", self.page.info_label.text())

    def test_real_engine_rejects_missing_end_tracking(self):
        self.page.engine = app.TremorEngine()
        worker = self.start_capture()
        t = 100 + np.arange(450) / 30
        worker.bounds = (t[0], t[-1], None)
        worker.samples = [(ti, 10.0 if ti < 114 else None, 0.0 if ti < 114 else None) for ti in t]
        self.clock.return_value = 115.0
        worker.finished.emit()
        self.assertEqual(self.page.analysis_badge.text(), "INVALID CAPTURE")
        self.assertIn("в начале или конце", self.page.analysis_detail.text())
        self.assertIn("Наибольший интервал:", self.page.info_label.text())

    def test_short_webcam_gaps_render_limited_measurement(self):
        self.page.engine = app.TremorEngine()
        worker = self.start_capture()
        t = 100 + np.arange(497) / 33.1
        keep = np.ones(len(t), dtype=bool)
        keep[20::11] = False
        t = t[keep]
        t[100] += 0.021
        worker.bounds = (t[0], t[-1], None)
        worker.samples = [(ti, float(10 * np.sin(2 * np.pi * 4 * ti)), 0.0) for ti in t]
        self.clock.return_value = 115.1
        worker.finished.emit()
        self.assertEqual(self.page.analysis_badge.text(), "LIMITED BAND")
        self.assertEqual(self.status.badges['DSP'].text(), "LIMITED")
        self.assertIn("Более быстрые движения не оценивались", self.page.analysis_detail.text())
        self.assertAlmostEqual(float(self.page.m_freq.val.text()), 4, delta=0.2)


class CaptureWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.worker = capture_worker.CaptureWorker(preparation_duration=0)
        self.camera = Mock()
        self.landmarker = Mock()
        self.cancelled = False
        self.now = 100.0
        self.patches = [
            patch("capture_worker.ensure_model"),
            patch("capture_worker.cv2.VideoCapture", return_value=self.camera),
            patch("capture_worker.HandLandmarker.create_from_options", return_value=self.landmarker),
            patch("capture_worker.time.perf_counter", side_effect=lambda: self.now),
            patch.object(self.worker, "isInterruptionRequested", side_effect=lambda: self.cancelled),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.worker.deleteLater()

    def test_timestamps_are_acquisition_times_and_only_latest_frame_is_kept(self):
        reads = [100.1, 100.2, 100.3]
        landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
        def read():
            self.now = reads.pop(0)
            return True, np.zeros((12, 16, 3), dtype=np.uint8)
        def detect(image, timestamp):
            self.now += 0.05
            if not reads:
                self.cancelled = True
            return SimpleNamespace(hand_landmarks=[landmarks])
        self.camera.read.side_effect = read
        self.landmarker.detect_for_video.side_effect = detect
        self.worker.run()
        latest, samples = self.worker.take_pending()
        self.assertEqual(samples, [(100.1, 8.0, 6.0), (100.2, 8.0, 6.0), (100.3, 8.0, 6.0)])
        self.assertEqual(latest[1], 100.3)
        self.assertEqual(self.worker.snapshot(), (100.1, 100.3, None, capture_worker.CAMERA_STALL_TIMEOUT))
        self.assertEqual(self.worker.take_pending(), (None, []))
        self.camera.release.assert_called_once_with()
        self.landmarker.close.assert_called_once_with()

    def test_worker_stops_at_deadline_and_excludes_late_frame(self):
        self.landmarker.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[])
        reads = [100.1, 100.5, 100.7]
        self.worker.duration = 0.5
        def read():
            self.now = reads.pop(0)
            return True, np.zeros((12, 16, 3), dtype=np.uint8)
        self.camera.read.side_effect = read
        self.worker.run()
        latest, samples = self.worker.take_pending()
        self.assertEqual(samples, [(100.1, None, None), (100.5, None, None)])
        self.assertEqual(latest[1], 100.5)
        self.assertEqual(self.worker.reason, "deadline")
        self.assertEqual(self.worker.snapshot(), (100.1, 100.5, None, capture_worker.CAMERA_STALL_TIMEOUT))

    def test_failed_reads_time_out_without_any_valid_frame(self):
        def read():
            self.now += 2.5
            return False, None
        self.camera.read.side_effect = read
        with patch.object(self.worker, "msleep"):
            self.worker.run()
        self.assertEqual(self.worker.reason, "error")
        self.assertIn("Camera stopped", self.worker.error)
        self.assertEqual(self.worker.take_pending(), (None, []))
        self.camera.release.assert_called_once_with()
        self.landmarker.close.assert_called_once_with()

    def test_slow_first_frame_is_allowed_during_camera_startup(self):
        self.worker.preparation_duration = 0
        calls = iter([103.0, 103.1])
        self.landmarker.detect_for_video.side_effect = lambda image, timestamp: (
            setattr(self, "cancelled", True) or SimpleNamespace(hand_landmarks=[])
        )
        self.camera.read.side_effect = lambda: (
            setattr(self, "now", next(calls)) or True,
            np.zeros((12, 16, 3), dtype=np.uint8),
        )
        self.worker.run()
        self.assertEqual(self.worker.error, "")
        start, end, _, timeout = self.worker.snapshot()
        self.assertEqual(start, 103.0)
        self.assertEqual(end, 103.0)
        self.assertEqual(timeout, capture_worker.CAMERA_STALL_TIMEOUT)
        self.camera.release.assert_called_once_with()
        self.landmarker.close.assert_called_once_with()

    def test_first_frame_after_startup_deadline_is_rejected(self):
        self.now = 100.0
        def delayed_read():
            self.now = 111.0
            return True, np.zeros((12, 16, 3), dtype=np.uint8)
        self.camera.read.side_effect = delayed_read
        self.worker.run()
        self.assertIn("10-second first-frame timeout", self.worker.error)
        self.landmarker.detect_for_video.assert_not_called()

    def test_failed_first_frame_uses_longer_startup_timeout(self):
        waits = iter([102.0, 104.0, 106.0, 108.0, 110.1])
        self.camera.read.side_effect = lambda: (setattr(self, "now", next(waits)) or False, None)
        with patch.object(self.worker, "msleep"):
            self.worker.run()
        self.assertIn("Camera stopped delivering frames", self.worker.error)
        self.assertEqual(self.worker.reason, "error")

    def test_inference_failure_releases_both_resources(self):
        self.camera.read.return_value = True, np.zeros((12, 16, 3), dtype=np.uint8)
        self.landmarker.detect_for_video.side_effect = RuntimeError("Inference failed")
        self.worker.run()
        self.assertEqual(self.worker.error, "Inference failed")
        self.camera.release.assert_called_once_with()
        self.landmarker.close.assert_called_once_with()

    def test_cleanup_attempts_model_close_even_if_camera_release_fails(self):
        self.camera.isOpened.return_value = False
        self.camera.release.side_effect = RuntimeError("Release failed")
        self.worker.run()
        self.landmarker.close.assert_called_once_with()
        self.assertIn("Camera could not", self.worker.error)

    def test_sample_buffer_overflow_fails_instead_of_dropping_scalars(self):
        self.worker._sample_limit = 2
        self.camera.read.return_value = True, np.zeros((12, 16, 3), dtype=np.uint8)
        self.landmarker.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[])
        self.worker.run()
        latest, samples = self.worker.take_pending()
        self.assertEqual(len(samples), 2)
        self.assertIsNotNone(latest)
        self.assertEqual(self.worker.error, "Capture sample limit exceeded")

    def test_cancellation_before_setup_does_not_open_camera(self):
        self.cancelled = True
        self.worker.run()
        self.camera.isOpened.assert_not_called()
        self.landmarker.close.assert_not_called()
        self.assertEqual(self.worker.error, "")

    def test_preparation_frames_do_not_enter_recording(self):
        self.worker.preparation_duration = 2
        reads = iter(100 + np.arange(150) / 30)
        landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
        def read():
            self.now = float(next(reads))
            return True, np.zeros((12, 16, 3), dtype=np.uint8)
        def detect(image, timestamp):
            if self.now >= 103.5:
                self.cancelled = True
            return SimpleNamespace(hand_landmarks=[landmarks] if self.now >= 101 else [])
        self.camera.read.side_effect = read
        self.landmarker.detect_for_video.side_effect = detect
        self.worker.run()
        start, end, _, _ = self.worker.snapshot()
        self.assertAlmostEqual(start, 103, places=5)
        latest, samples = self.worker.take_pending()
        self.assertTrue(all(t >= start for t, _, _ in samples))
        self.assertEqual(samples[0][0], start)
        self.assertEqual(latest[1], end)

    def test_preparation_with_no_hand_times_out(self):
        self.worker.preparation_duration = 2
        def read():
            self.now += 0.1
            return True, np.zeros((12, 16, 3), dtype=np.uint8)
        self.camera.read.side_effect = read
        self.landmarker.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[])
        self.worker.run()
        self.assertIn("Preparation timed out", self.worker.error)
        self.assertIsNone(self.worker.snapshot()[0])
        _, samples = self.worker.take_pending()
        self.assertEqual(samples, [])
        self.camera.release.assert_called_once_with()

    def test_preparation_waits_for_hand_to_stop_moving(self):
        self.worker.preparation_duration = 2
        reads = iter(100 + np.arange(180) / 30)
        def read():
            self.now = float(next(reads))
            return True, np.zeros((12, 100, 3), dtype=np.uint8)
        def detect(image, timestamp):
            x = (0.1 if int((self.now - 100) * 30) % 2 else 0.7) if self.now < 102 else 0.5
            if self.now >= 105:
                self.cancelled = True
            return SimpleNamespace(hand_landmarks=[[SimpleNamespace(x=x, y=0.5) for _ in range(21)]])
        self.camera.read.side_effect = read
        self.landmarker.detect_for_video.side_effect = detect
        self.worker.run()
        start, _, _, _ = self.worker.snapshot()
        self.assertGreaterEqual(start, 104)
        self.assertLess(start, 105)


class ModelDownloadTests(unittest.TestCase):
    def test_existing_user_model_is_never_downloaded_or_deleted(self):
        with patch("capture_worker.os.path.isfile", return_value=True), patch("capture_worker.urllib.request.urlopen") as request, patch("capture_worker.os.unlink") as unlink:
            capture_worker.ensure_model(lambda: False)
        request.assert_not_called()
        unlink.assert_not_called()

    def test_download_has_pinned_url_timeout_and_atomic_rename(self):
        output = MagicMock()
        output.name = "download.part"
        output.tell.return_value = 4
        response = MagicMock()
        response.read1.side_effect = [b"data", b""]
        with patch("capture_worker.os.path.isfile", return_value=False), patch("capture_worker.os.path.exists", return_value=False), patch("capture_worker.os.makedirs"), patch("capture_worker.tempfile.NamedTemporaryFile") as temporary, patch("capture_worker.urllib.request.urlopen") as request, patch("capture_worker.os.rename") as rename, patch("capture_worker.os.unlink") as unlink:
            temporary.return_value.__enter__.return_value = output
            request.return_value.__enter__.return_value = response
            capture_worker.ensure_model(lambda: False)
        self.assertIn("/1/", capture_worker.HAND_MODEL_URL)
        request.assert_called_once_with(capture_worker.HAND_MODEL_URL, timeout=10)
        output.write.assert_called_once_with(b"data")
        rename.assert_called_once_with("download.part", capture_worker.HAND_MODEL_PATH)
        unlink.assert_not_called()

    def test_download_failure_removes_only_temporary_file(self):
        output = MagicMock()
        output.name = "download.part"
        with patch("capture_worker.os.path.isfile", return_value=False), patch("capture_worker.os.makedirs"), patch("capture_worker.tempfile.NamedTemporaryFile") as temporary, patch("capture_worker.urllib.request.urlopen", side_effect=TimeoutError("Network timeout")), patch("capture_worker.os.rename") as rename, patch("capture_worker.os.unlink") as unlink:
            temporary.return_value.__enter__.return_value = output
            with self.assertRaises(TimeoutError):
                capture_worker.ensure_model(lambda: False)
        rename.assert_not_called()
        unlink.assert_called_once_with("download.part")


if __name__ == "__main__":
    unittest.main()
