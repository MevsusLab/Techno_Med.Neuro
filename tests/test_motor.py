import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import multiprocessing
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication, QLabel

from app import MotorPage, MainWindow
from motor_engine import analyze_tapping, finger_separation, motor_band
from motor_capture import MotorCaptureWorker, _motor_process
from motor_protocol import MotorProtocolPanel, MOTOR_DISCLAIMER, motor_protocol_text


def tapping_samples(frequency=3.2, duration=15, decay=0, rate=60, offset=0):
    t = np.arange(0, duration, 1 / rate)
    d = .05 + (1 + decay * t / duration) * (1 - np.cos(2 * np.pi * frequency * t)) / 2
    return [{'timestamp': float(x + offset), 'distance': float(y), 'palm_scale': 100., 'hand': 'Right'}
            for x, y in zip(t, d)]


class MotorEngineTests(unittest.TestCase):
    def test_normalized_distance_preserves_translation_rotation_and_scale(self):
        points = np.full((21, 2), 100.)
        points[9] = [100, 180]
        points[4], points[8] = [80, 120], [144, 120]
        distance, _ = finger_separation(points)
        self.assertAlmostEqual(distance, .8)
        rotation = np.array([[0, -1], [1, 0]])
        transformed = points @ rotation * 2 + [700, 300]
        self.assertAlmostEqual(finger_separation(transformed)[0], distance)
        with self.assertRaises(ValueError):
            finger_separation(points * .1)

    def test_known_tap_rate_counts_complete_cycles_and_stable_amplitude(self):
        for rate in (25, 30, 60):
            with self.subTest(rate=rate):
                r = analyze_tapping(tapping_samples(rate=rate), 0, 15)
                self.assertAlmostEqual(r['tap_frequency'], 3.2, delta=.05)
                self.assertAlmostEqual(r['amplitude_decrement'], 0, delta=2)
                self.assertAlmostEqual(r['total_taps'], 48, delta=1)
                self.assertLess(r['rhythm_cv'], 10)
                self.assertEqual(motor_band(r), 'stable')

    def test_falling_peak_regression_measures_decrement(self):
        r = analyze_tapping(tapping_samples(decay=-.45), 0, 15)
        self.assertLess(r['amplitude_slope'], 0)
        self.assertAlmostEqual(r['amplitude_decrement'], -41, delta=4)
        self.assertEqual(motor_band(r), 'marked')

    def test_slow_taps_are_valid_and_not_tracking_failure(self):
        r = analyze_tapping(tapping_samples(frequency=1.1), 0, 15)
        self.assertAlmostEqual(r['tap_frequency'], 1.1, delta=.02)
        self.assertEqual(motor_band(r), 'marked')

    def test_irregular_intervals_are_measured(self):
        samples = tapping_samples()
        centers = np.array([.5, 1.0, 1.2, 2.0, 2.3, 3., 3.2, 4., 4.3, 5., 5.2, 6.,
                            6.3, 7., 7.2, 8., 8.3, 9., 9.2, 10., 10.3, 11., 11.2, 12., 12.3, 13., 14.])
        for s in samples:
            s['distance'] = float(.05 + np.max(np.exp(-((s['timestamp'] - centers) / .06) ** 2)))
        r = analyze_tapping(samples, 0, 15)
        self.assertGreater(r['rhythm_cv'], 35)
        self.assertEqual(motor_band(r), 'marked')

    def test_absolute_capture_time_has_no_effect(self):
        a = analyze_tapping(tapping_samples(), 0, 15)
        b = analyze_tapping(tapping_samples(offset=1234567), 1234567, 1234582)
        self.assertAlmostEqual(a['tap_frequency'], b['tap_frequency'], places=6)
        # Floating-point precision at large absolute timestamps is finite;
        # sub-millipercent variation is far below the report's 0.1% precision.
        self.assertAlmostEqual(a['amplitude_decrement'], b['amplitude_decrement'], delta=.001)

    def test_incomplete_cycles_and_flat_signal_rejected(self):
        for baseline in (.6, 0):
            samples = tapping_samples()
            for s in samples:
                s['distance'] = baseline + (.2 * s['distance'] if baseline else 0)
            with self.subTest(baseline=baseline), self.assertRaisesRegex(ValueError, 'полных циклов'):
                analyze_tapping(samples, 0, 15)

    def test_tracking_gaps_camera_motion_and_hand_switch_rejected(self):
        for problem in ('gap', 'scale', 'hand'):
            samples = tapping_samples()
            if problem == 'gap':
                for s in samples[120:150]:
                    s['distance'] = None
            elif problem == 'scale':
                for s in samples[len(samples)//2:]:
                    s['palm_scale'] = 180
            else:
                samples[-1]['hand'] = 'Left'
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                analyze_tapping(samples, 0, 15)

    def test_low_fps_short_capture_and_duplicate_times_rejected(self):
        for samples, end in ((tapping_samples(rate=20), 15), (tapping_samples(duration=8), 8),
                             (tapping_samples() + tapping_samples()[-1:], 15)):
            with self.assertRaises(ValueError):
                analyze_tapping(samples, 0, end)

    def test_missing_single_frame_is_flagged_not_counted_as_a_tap(self):
        clean = analyze_tapping(tapping_samples(), 0, 15)
        samples = tapping_samples()
        for i in range(200, 260, 3):
            samples[i]['distance'] = None
        r = analyze_tapping(samples, 0, 15)
        self.assertTrue(r['warnings'])
        self.assertAlmostEqual(r['total_taps'], clean['total_taps'], delta=1)

    def test_threshold_boundaries_and_severe_precedence(self):
        cases = [(3, -14.9, 20, 'stable'), (2.8, 0, 20, 'stable'), (3, -15, 0, 'borderline'),
                 (3, -25, 0, 'borderline'), (2, 0, 0, 'borderline'), (3, 0, 20.1, 'borderline'),
                 (3, 0, 35, 'borderline'), (3, 0, 35.1, 'marked'), (2.5, -30, 0, 'marked'),
                 (1.9, -20, 0, 'marked')]
        for f, dec, cv, band in cases:
            self.assertEqual(motor_band({'tap_frequency': f, 'amplitude_decrement': dec, 'rhythm_cv': cv}), band)


class FakeMotorWorker(QObject):
    finished = pyqtSignal()
    failed = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, **kwargs):
        super().__init__()
        self.error, self.reason = '', 'deadline'
        self.start = Mock()
        self.requestInterruption = Mock()
        self.deleteLater = Mock()
        self.snapshot = Mock(return_value=(0, 15, None, 2))
        self.take_pending = Mock(return_value=(None, []))


class MotorUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.page = MotorPage()
        self.page.resize(1200, 750)
        self.page.show()
        self.qt.processEvents()

    def tearDown(self):
        self.page.cleanup()
        if self.page.worker:
            self.page.worker.finished.emit()
        self.page.close()
        self.page.deleteLater()
        self.qt.processEvents()

    def complete(self):
        worker = FakeMotorWorker()
        worker.take_pending.return_value = (None, tapping_samples())
        self.page.worker = worker
        self.page._finished()
        self.page.protocol_panel.reveal_controller.animation.setCurrentTime(240)
        self.qt.processEvents()
        self.assertEqual(self.page.badge.text(), 'COMPLETE')
        return worker

    def test_completed_panel_is_half_height_and_closing_restores_viewport(self):
        full = self.page.video.height()
        self.complete()
        panel = self.page.protocol_panel
        self.assertAlmostEqual(self.page.video.height() / (self.page.video.height() + panel.height()), .5, delta=.02)
        self.assertLess(self.page.video.geometry().bottom(), panel.geometry().top())
        self.assertLess(self.page.capture_area.geometry().bottom(), self.page.start.geometry().top())
        panel.close_button.click()
        self.qt.processEvents()
        self.assertGreater(self.page.video.height(), full * .95)

    def test_new_capture_and_reset_discard_previous_metrics(self):
        self.complete()
        with patch('app.MotorCaptureWorker', side_effect=FakeMotorWorker):
            self.page.start.click()
        self.assertIsNone(self.page.results)
        self.assertTrue(self.page.protocol_panel.isHidden())
        self.assertFalse(self.page.start.isEnabled())
        self.page.reset.click()
        self.page.worker.finished.emit()
        self.assertEqual(self.page.badge.text(), 'IDLE')
        self.assertEqual(self.page.frequency.val.text(), '---')
        self.assertTrue(self.page.start.isEnabled())

    def test_stop_discards_late_results(self):
        with patch('app.MotorCaptureWorker', side_effect=FakeMotorWorker):
            self.page.start.click()
        self.page.worker.take_pending.return_value = (None, tapping_samples())
        self.page.stop.click()
        self.page.worker.finished.emit()
        self.assertIsNone(self.page.results)
        self.assertTrue(self.page.protocol_panel.isHidden())

    def test_upload_uses_video_worker_and_preserves_source_in_report(self):
        with patch('app.QFileDialog.getOpenFileName', return_value=('C:/sample.mp4', '')), \
                patch('app.MotorCaptureWorker', side_effect=FakeMotorWorker) as factory:
            self.page.upload_video.click()
        factory.assert_called_once_with(path='C:/sample.mp4', parent=self.page)
        self.page.worker.take_pending.return_value = (None, tapping_samples())
        self.page.worker.finished.emit()
        self.assertIn('sample.mp4', self.page.protocol_panel.report)

    def test_failed_capture_releases_worker_and_allows_restart(self):
        worker = FakeMotorWorker()
        worker.error = 'Кисть не видна'
        self.page.worker = worker
        self.page._finished()
        self.assertEqual(self.page.badge.text(), 'RETRY')
        self.assertTrue(self.page.start.isEnabled())
        self.assertTrue(self.page.protocol_panel.isHidden())
        worker.deleteLater.assert_called_once()

    def test_graph_peak_markers_trend_and_mode(self):
        self.complete()
        r = self.page.results
        self.assertEqual(len(self.page.wave.ax.collections[0].get_offsets()), r['total_taps'])
        self.assertTrue(np.allclose(self.page.wave.ax.lines[-1].get_ydata(), r['trend']))
        self.page.chart_mode.setCurrentIndex(1)
        self.assertEqual(len(self.page.wave.ax.lines), 1)

    def test_exports_have_kinematics_disclaimer_and_brief_final_action(self):
        self.complete()
        panel = self.page.protocol_panel
        with tempfile.TemporaryDirectory() as tmp:
            text, pdf = Path(tmp) / 'motor.txt', Path(tmp) / 'motor.pdf'
            with patch('protocol_panel.QFileDialog.getSaveFileName', return_value=(str(text), '')):
                panel.save_button.click()
            report = text.read_text(encoding='utf-8')
            for token in ('Частота тапов', 'Полных циклов', 'CV межтаповых', MOTOR_DISCLAIMER):
                self.assertIn(token, report)
            self.assertTrue(report.endswith(panel.action_label.text()))
            with patch('protocol_panel.QFileDialog.getSaveFileName', return_value=(str(pdf), '')):
                panel.export_button.click()
            self.assertTrue(pdf.read_bytes().startswith(b'%PDF-'))

    def test_reports_do_not_assign_disease_or_updrs_score(self):
        result = analyze_tapping(tapping_samples(), 0, 15)
        for f, dec, cv in ((3, 0, 0), (2.5, -20, 21), (1.8, -35, 40)):
            result.update(tap_frequency=f, amplitude_decrement=dec, rhythm_cv=cv)
            title, analysis, action, quality = motor_protocol_text(result)
            self.assertNotIn('болезнь Паркинсона', analysis)
            self.assertNotIn('[НОРМА]', title)
            self.assertNotIn('UPDRS POSITIVE', title)
            self.assertIn('не баллы UPDRS', quality)
            self.assertTrue(action)

    def test_navigation_and_deferred_close_include_motor_worker(self):
        window = MainWindow()
        try:
            self.assertEqual(window.nav.buttons[-1].text(), 'МОТОРИКА')
            window.switch_page(6)
            worker = FakeMotorWorker()
            window.motor_page.worker = worker
            window.switch_page(1)
            self.assertEqual(window.stack.currentIndex(), 6)
            event = QCloseEvent()
            window.closeEvent(event)
            self.assertFalse(event.isAccepted())
            worker.requestInterruption.assert_called()
            worker.finished.connect(window.motor_page._finished)
            worker.finished.emit()
            self.qt.processEvents()
        finally:
            window.motor_page.worker = None
            window.close()

    def test_minimum_window_controls_fit_and_resizing_keeps_equal_split(self):
        window = MainWindow()
        window.resize(1200, 750)
        window.show()
        window.switch_page(6)
        self.qt.processEvents()
        page = window.motor_page
        try:
            for button in (page.start, page.upload_video, page.stop, page.reset):
                self.assertLessEqual(button.geometry().right(), page.capture_area.geometry().right())
            self.assertLess(window.nav.buttons[-1].geometry().right(), window.width())
            page._render(analyze_tapping(tapping_samples(), 0, 15))
            page.protocol_panel.reveal_controller.animation.setCurrentTime(240)
            self.qt.processEvents()
            self.assertAlmostEqual(page.video.height(), page.protocol_panel.height(), delta=2)
            window.resize(1400, 860)
            self.qt.processEvents()
            self.assertAlmostEqual(page.video.height(), page.protocol_panel.height(), delta=2)
        finally:
            window.close()


class Connection:
    def __init__(self):
        self.packets = []

    def send(self, packet):
        self.packets.append(packet)

    def close(self):
        pass


class MotorCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def test_file_inference_preserves_frame_times_normalization_overlay_and_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tapping.avi'
            path.write_bytes(b'fixture')
            frame = np.full((480, 640, 3), 90, dtype=np.uint8)
            camera, model = Mock(), Mock()
            camera.isOpened.return_value = True
            camera.read.side_effect = [(True, frame.copy()) for _ in range(300)] + [(False, None)]
            camera.get.return_value = 0
            points = np.tile([.5, .5], (21, 1))
            points[0], points[9], points[4], points[8] = [.5, .6], [.5, .4], [.4, .3], [.6, .3]
            landmarks = [SimpleNamespace(x=x, y=y) for x, y in points]
            model.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[landmarks], handedness=[])
            connection = Connection()
            with patch('motor_capture.validate_video', return_value=(path, 30, 300, 10)), \
                    patch('motor_capture.cv2.VideoCapture', return_value=camera), \
                    patch('motor_capture.ensure_model'), \
                    patch('motor_capture.HandLandmarker.create_from_options', return_value=model) as create:
                _motor_process(connection, threading.Event(), 15, 2, str(path))
            samples = [s for p in connection.packets for s in p[1]]
            self.assertEqual(len(samples), 300)
            self.assertAlmostEqual(samples[-1]['timestamp'], 299/30)
            self.assertAlmostEqual(samples[0]['distance'], 128/96)
            self.assertEqual(connection.packets[-1][8], 'deadline')
            self.assertEqual(connection.packets[-1][2][:2], (0, 10))
            options = create.call_args.args[0]
            self.assertEqual(options.num_hands, 1)
            self.assertEqual(options.min_hand_detection_confidence, .6)
            self.assertEqual(options.min_tracking_confidence, .6)
            self.assertLessEqual(connection.packets[-1][0]['image'].shape[1], 320)
            self.assertGreater(np.std(connection.packets[-1][0]['image']), 0)
            camera.release.assert_called_once()
            model.close.assert_called_once()

    def test_camera_startup_failure_and_preemptive_stop_cleanup(self):
        camera, model = Mock(), Mock()
        camera.isOpened.return_value = False
        connection = Connection()
        with patch('motor_capture.cv2.VideoCapture', return_value=camera), \
                patch('motor_capture.ensure_model'), \
                patch('motor_capture.HandLandmarker.create_from_options', return_value=model):
            _motor_process(connection, threading.Event(), 15, 2)
        self.assertEqual(connection.packets[-1][8], 'error')
        self.assertIn('камеру', connection.packets[-1][7])
        model.close.assert_called_once()
        camera.release.assert_called_once()
        stopped = threading.Event()
        stopped.set()
        with patch('motor_capture.cv2.VideoCapture') as factory:
            _motor_process(Connection(), stopped, 15, 2)
        factory.assert_not_called()

    def test_camera_preparation_excluded_and_fifteen_seconds_measured(self):
        camera, model = Mock(), Mock()
        camera.isOpened.return_value = True
        now = [100.]
        frame = np.full((480, 640, 3), 90, dtype=np.uint8)
        def read():
            now[0] += 1/30
            return True, frame.copy()
        camera.read.side_effect = read
        points = np.tile([.5, .5], (21, 1))
        points[0], points[9], points[4], points[8] = [.5, .6], [.5, .4], [.4, .3], [.6, .3]
        model.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[[SimpleNamespace(x=x, y=y) for x, y in points]], handedness=[])
        connection = Connection()
        with patch('motor_capture.cv2.VideoCapture', return_value=camera), \
                patch('motor_capture.ensure_model'), \
                patch('motor_capture.HandLandmarker.create_from_options', return_value=model), \
                patch('motor_capture.time.perf_counter', side_effect=lambda: now[0]):
            _motor_process(connection, threading.Event(), 15, 2)
        final = connection.packets[-1]
        self.assertEqual(final[8], 'deadline')
        self.assertAlmostEqual(final[2][1] - final[2][0], 15)
        self.assertGreaterEqual(final[2][0], 102)
        samples = [s for p in connection.packets for s in p[1]]
        self.assertTrue(all(final[2][0] <= s['timestamp'] < final[2][1] for s in samples))
        self.assertGreaterEqual(len(samples), 450)
        camera.release.assert_called_once()
        model.close.assert_called_once()

    def test_supervised_real_video_worker_finishes_without_camera_or_child_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'blank.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (160, 120))
            self.assertTrue(writer.isOpened())
            for _ in range(300):
                writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
            writer.release()
            before = {p.pid for p in multiprocessing.active_children()}
            worker = MotorCaptureWorker(path=str(path))
            worker.start()
            self.assertTrue(worker.wait(60000), 'Motor worker did not release its child.')
            self.assertEqual(worker.error, '')
            self.assertEqual(worker.reason, 'deadline')
            latest, samples = worker.take_pending()
            self.assertEqual(len(samples), 300)
            self.assertTrue(all(s['distance'] is None for s in samples))
            self.assertEqual(latest['state'], 'COMPLETE')
            self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)


if __name__ == '__main__':
    unittest.main()
