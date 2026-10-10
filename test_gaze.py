import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PyQt6.QtWidgets import QApplication

from aac_speech import SpeechWorker, russian_voice
from app import AAC_CATEGORIES, AAC_WORDS_PER_SHEET, AACPage, MainWindow
from gaze_engine import (CALIBRATION_MIN_SAMPLES, DEADZONE_PX, DWELL_GRACE_SECONDS, DwellEngine,
                         DwellState, GazeSmoother, GazeTargeting, best_effort_sample, calibration_sample,
                         fit_gaze_classifier, gaze_features, head_motion_covariance,
                         inconsistent_targets, tracking_features)
from gaze_simulation import PageDriver, VirtualUser, face_landmarks


def gaze_face(gx=0.5, gy=0.5):
    """Flat synthetic face with exact iris ratios (both eyes share one box)."""
    points = np.zeros((478, 3))
    for iris_start, corners, lids in ((468, (33, 133), (159, 145)), (473, (362, 263), (386, 374))):
        points[corners[0], :2] = [100, 100]
        points[corners[1], :2] = [160, 100]
        points[lids[0], :2] = [130, 94]
        points[lids[1], :2] = [130, 106]
        points[iris_start:iris_start + 5, :2] = [100 + 60 * gx, 94 + 12 * gy]
    points[1, :2] = [130, 140]
    return points


def target_windows(means, rng, noise=(.004, .002, .002, .002), passes=2, samples=30):
    return [[mean + rng.normal(0, noise, (samples, len(mean))) for _ in range(passes)] for mean in means]


GRID = np.array([[.0, .0, 0, 0], [.1, .0, 0, 0], [.0, .05, 0, 0], [.1, .05, 0, 0]]) + [.5, .5, 0, .3]


class GazeFeatureTests(unittest.TestCase):
    def test_iris_ratios_return_normalized_gaze_coordinates(self):
        gaze = gaze_features(gaze_face(0.63, 0.42), (0, 0, 0))
        self.assertAlmostEqual(gaze[0], 0.63)
        self.assertAlmostEqual(gaze[1], 0.42)

    def test_opposite_eye_axes_align_without_cancelling_horizontal_motion(self):
        for x in (0.2, 0.5, 0.8):
            points = gaze_face(x, 0.6)
            points[33, 0], points[133, 0] = 100, 160
            points[362, 0], points[263, 0] = 100, 160
            self.assertTrue(np.allclose(gaze_features(points), [x, 0.6]))
            self.assertTrue(np.allclose(gaze_features(points / 640), [x, 0.6]))

    def test_iris_center_is_used_instead_of_ring_average(self):
        points = gaze_face(.7, .4)
        points[469:473, :2] = [100, 94]
        points[474:478, :2] = [160, 106]
        self.assertTrue(np.allclose(gaze_features(points), [.7, .4]))

    def test_pose_closed_eyes_and_invalid_landmarks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Поверните'):
            gaze_features(gaze_face(), (22, 0, 0))
        with self.assertRaisesRegex(ValueError, 'закрыты'):
            gaze_features(face_landmarks(blink=True))
        with self.assertRaises(ValueError):
            tracking_features(np.full((478, 2), np.nan))
        with self.assertRaises(ValueError):
            tracking_features(np.zeros((400, 2)))

    def test_tracking_features_are_translation_scale_and_roll_invariant(self):
        points = face_landmarks((6, -3), (2, 1))
        baseline = tracking_features(points)
        angle = np.deg2rad(12)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.assertTrue(np.allclose(tracking_features(points @ rotation.T * 1.6 + [80, 120]), baseline))

    def test_gaze_moves_only_iris_while_head_rotation_moves_iris_against_head(self):
        base = tracking_features(face_landmarks())
        gaze_right = tracking_features(face_landmarks(gaze=(1, 0))) - base
        gaze_down = tracking_features(face_landmarks(gaze=(0, 1))) - base
        self.assertGreater(gaze_right[0], .005)
        self.assertGreater(gaze_down[1], .005)
        self.assertTrue(np.allclose(gaze_right[2:], 0, atol=1e-6))
        self.assertTrue(np.allclose(gaze_down[2:], 0, atol=1e-6))
        head_right = tracking_features(face_landmarks(head=(1, 0))) - base
        head_down = tracking_features(face_landmarks(head=(0, 1))) - base
        # Eye counter-rotation: iris and nose proxy change with opposite signs.
        self.assertLess(head_right[0], 0)
        self.assertGreater(head_right[2], 0)
        self.assertLess(head_down[1], 0)
        self.assertGreater(head_down[3], 0)

    def test_eye_corner_feature_is_not_inverted_by_moving_eyelids(self):
        points = gaze_face(.5, .3)
        moved = points.copy()
        for iris, upper, lower in ((468, 159, 145), (473, 386, 374)):
            moved[iris, 1] -= 2
            moved[upper, 1] -= 4
            moved[lower, 1] -= 1
        self.assertGreater(gaze_features(moved)[1], gaze_features(points)[1])
        self.assertLess(tracking_features(moved)[1], tracking_features(points)[1])

    def test_ema_step_response_duplicate_gap_reset_and_reacquisition(self):
        smoother = GazeSmoother()
        self.assertTrue(np.allclose(smoother.update([.2, .4], 1), [.2, .4]))
        self.assertTrue(np.allclose(smoother.update([.6, .8], 1.03), [.3, .5]))
        self.assertTrue(np.allclose(smoother.update([0, 0], 1.03), [.3, .5]))
        self.assertIsNone(smoother.update(None, 1.06))
        self.assertTrue(np.allclose(smoother.update([.8, .2], 1.1), [.8, .2]))
        self.assertTrue(np.allclose(smoother.update([.1, .9], 1.6), [.1, .9]))


class CalibrationEngineTests(unittest.TestCase):
    def test_sample_trims_blink_outliers_but_rejects_jump_and_short_windows(self):
        rng = np.random.default_rng(5)
        values = rng.normal([.5, .52, .0, .3], [.002, .001, .001, .001], (40, 4))
        values[5] = [1, 1, 1, 1]
        center, inliers = calibration_sample(values)
        self.assertLess(np.linalg.norm(center - [.5, .52, 0, .3]), .002)
        self.assertEqual(len(inliers), 39)
        jumped = values.copy()
        jumped[20:, 0] += .06
        with self.assertRaisesRegex(ValueError, 'сместился'):
            calibration_sample(jumped)
        with self.assertRaisesRegex(ValueError, 'Недостаточно'):
            calibration_sample(values[:CALIBRATION_MIN_SAMPLES - 1])

    def test_best_effort_keeps_the_settled_end_of_a_restless_window(self):
        times = np.arange(90) / 30
        values = np.tile([.5, .5, 0, .3], (90, 1))
        values[:45, 0] = np.linspace(.3, .5, 45)
        kept = best_effort_sample(values, times)
        self.assertTrue(np.allclose(np.median(kept, axis=0), [.5, .5, 0, .3]))

    def test_head_motion_step_is_learned_and_tiny_motion_falls_back_to_prior(self):
        rng = np.random.default_rng(2)
        t = np.arange(90) / 30
        head = np.column_stack((4 * np.sin(2 * np.pi * t / 2), 3 * np.sin(2 * np.pi * t / 1.4)))
        sweep = np.column_stack((.5 - .0034 * head[:, 0], .5 - .0036 * head[:, 1],
                                 .0055 * head[:, 0], .3 + .0055 * head[:, 1])) + rng.normal(0, .001, (90, 4))
        covariance, learned = head_motion_covariance(sweep)
        self.assertTrue(learned)
        self.assertLess(np.corrcoef(sweep[:, 0], sweep[:, 2])[0, 1], -.9)
        self.assertLess(covariance[0, 2], 0)
        _, learned = head_motion_covariance(np.tile([.5, .5, 0, .3], (90, 1)) + rng.normal(0, 1e-4, (90, 4)))
        self.assertFalse(learned)
        _, learned = head_motion_covariance(None)
        self.assertFalse(learned)

    def test_classifier_enters_inside_rejects_gap_offgrid_invalid_and_holds_current(self):
        model = fit_gaze_classifier(target_windows(GRID, np.random.default_rng(1)))
        for index, mean in enumerate(GRID):
            self.assertEqual(model.classify(mean), index)
        midpoint = (GRID[0] + GRID[1]) / 2
        self.assertIsNone(model.classify(midpoint))
        near_zero = GRID[0] + .45 * (GRID[1] - GRID[0])
        self.assertIsNone(model.classify(near_zero))
        self.assertEqual(model.classify(near_zero, current=0), 0)
        self.assertEqual(model.classify(GRID[0] + .6 * (GRID[1] - GRID[0]), current=0), 1)
        self.assertIsNone(model.classify(GRID[0] + [-.5, -.5, 0, 0]))
        self.assertIsNone(model.classify([np.nan, .5, 0, .3]))
        self.assertIsNone(model.classify([.5, .5]))
        self.assertEqual(model.weak_pairs(), [])

    def test_fit_is_well_conditioned_for_identical_windows_and_reports_weak_pairs(self):
        means = GRID.copy()
        means[3] = means[2] + [.0005, 0, 0, 0]
        windows = [[np.tile(mean, (20, 1))] for mean in means]
        model = fit_gaze_classifier(windows)
        self.assertTrue(np.isfinite(model.precision).all())
        self.assertIn((2, 3), model.weak_pairs())
        with self.assertRaises(ValueError):
            fit_gaze_classifier([[np.tile(GRID[0], (20, 1))]])
        with self.assertRaises(ValueError):
            fit_gaze_classifier([[np.full((20, 4), np.nan)], [np.tile(GRID[1], (20, 1))]])

    def test_inconsistent_window_is_identified_for_refinement(self):
        windows = target_windows(GRID, np.random.default_rng(3))
        windows[1][0] = windows[3][0].copy()
        model = fit_gaze_classifier(windows)
        self.assertIn(1, inconsistent_targets(model, windows))
        self.assertNotIn(0, inconsistent_targets(model, windows))

    def test_drift_correction_follows_shift_but_never_pulls_gap_onto_button(self):
        model = fit_gaze_classifier(target_windows(GRID, np.random.default_rng(4)))
        targeting = GazeTargeting(model)
        shift = .2 * (GRID[1] - GRID[0])
        for _ in range(200):
            self.assertEqual(targeting.classify(GRID[0] + shift), 0)
        self.assertGreater(np.linalg.norm(targeting.offset), .5 * np.linalg.norm(shift))
        self.assertLessEqual(np.sqrt(targeting.offset @ model.precision @ targeting.offset), targeting.bound + 1e-9)
        fresh = GazeTargeting(model)
        midpoint = (GRID[0] + GRID[1]) / 2
        for _ in range(300):
            self.assertIsNone(fresh.classify(midpoint))
        self.assertTrue(np.allclose(fresh.offset, 0))


class DwellEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = DwellEngine()
        self.targets = {'word': (0, 0, 100, 100), 'arrow': (106, 0, 206, 100)}

    def fixation(self, point, start, frames=28):
        return [self.engine.update(point, self.targets, start + index / 30) for index in range(frames)]

    def test_exact_threshold_triggers_once_then_full_cooldown(self):
        updates = self.fixation([50, 50], 1)
        self.assertAlmostEqual(updates[15].progress, .5 / .9)
        self.assertEqual(updates[-1].state, DwellState.TRIGGERED)
        self.assertEqual(updates[-1].triggered, 'word')
        for index in range(1, 15):
            update = self.engine.update([150, 50], self.targets, 1.9 + index / 30)
            self.assertEqual(update.state, DwellState.COOLDOWN)
            self.assertEqual(update.progress, 0)
        update = self.engine.update([150, 50], self.targets, 2.4)
        self.assertEqual(update.state, DwellState.DWELLING)
        self.assertEqual(update.elapsed_ms, 0)

    def test_deadzone_entry_and_border_hysteresis(self):
        self.assertEqual(self.engine.update([DEADZONE_PX - 7, 50], self.targets, 1).state, DwellState.IDLE)
        self.assertEqual(self.engine.update([DEADZONE_PX, 50], self.targets, 1.03).target, 'word')
        update = self.engine.update([2, 50], self.targets, 1.06)
        self.assertEqual((update.target, update.elapsed_ms), ('word', 30))
        self.assertEqual(self.engine.update([-1, 50], self.targets, 1.09).state, DwellState.IDLE)
        self.assertEqual(self.engine.update([5, 50], self.targets, 1.12).state, DwellState.IDLE)

    def test_missing_frames_keep_dwell_briefly_but_unclassified_gaze_resets(self):
        self.fixation([50, 50], 1, 10)
        held = self.engine.update(None, self.targets, 1.36, missing=True)
        self.assertEqual((held.state, held.target), (DwellState.DWELLING, 'word'))
        self.assertIsNone(held.triggered)
        resumed = self.engine.update([50, 50], self.targets, 1.4)
        self.assertEqual(resumed.elapsed_ms, 400)
        lost = self.engine.update(None, self.targets, 1.4 + DWELL_GRACE_SECONDS + .05, missing=True)
        self.assertEqual(lost.state, DwellState.IDLE)
        self.fixation([50, 50], 3, 10)
        self.assertEqual(self.engine.update(None, self.targets, 3.36).state, DwellState.IDLE)

    def test_gaps_target_changes_duplicates_and_invalid_points_never_complete(self):
        self.fixation([50, 50], 1, 20)
        update = self.engine.update([50, 50], self.targets, 2)
        self.assertEqual(update.elapsed_ms, 0)
        update = self.engine.update([150, 50], self.targets, 2.03)
        self.assertEqual((update.target, update.elapsed_ms), ('arrow', 0))
        self.assertIsNone(self.engine.update([150, 50], self.targets, 2.03).triggered)
        self.assertEqual(self.engine.update([np.nan, 50], self.targets, 2.06).state, DwellState.IDLE)


class AACPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.page = AACPage()
        self.page.resize(1400, 860)
        self.page.show()
        self.qt.processEvents()

    def tearDown(self):
        self.page.cleanup()
        self.page.close()
        self.page.deleteLater()
        self.qt.processEvents()

    def calibrated(self, seed=0, **user):
        driver = PageDriver(self.page, VirtualUser(seed, **user))
        self.page._begin_calibration_state()
        self.page.frames_seen = True
        driver.calibrate()
        self.assertEqual(self.page.badge.text(), 'ACTIVE')
        return driver

    def selections(self, driver, point, seconds=2.0):
        with patch.object(self.page, '_activate_target') as activate:
            self.page.dwell_engine.reset()
            driver.look(point, seconds)
            return [call.args[0] for call in activate.call_args_list]

    def test_six_four_word_sheets_preserve_all_original_24_words_exactly_once(self):
        expected = {'ПОМОЩЬ', 'ВОДА', 'БОЛЬ', 'ДА', 'НЕТ', 'ТУАЛЕТ', 'ПЕРЕВЕРНУТЬ',
                    'ПОДНЯТЬ ГОЛОВУ', 'ДУШНО', 'ХОЛОДНО', 'ЖАРКО', 'ЧЕШЕТСЯ', 'ВРАЧ',
                    'ТАБЛЕТКА', 'ЕДА', 'СВЕТ', 'ОЧКИ', 'УСТАЛ', 'СПАСИБО', 'ПОЖАЛУЙСТА',
                    'СЕМЬЯ', 'ТЕЛЕФОН', 'НЕ ПОНИМАЮ', 'ХОРОШО'}
        words = [word for _, sheet in AAC_CATEGORIES for word in sheet]
        self.assertEqual(len(AAC_CATEGORIES), 6)
        self.assertTrue(all(len(sheet) == AAC_WORDS_PER_SHEET == 4 for _, sheet in AAC_CATEGORIES))
        self.assertEqual(len(words), 24)
        self.assertEqual(set(words), expected)
        self.assertEqual(AAC_CATEGORIES[0][1], ['ПОМОЩЬ', 'ВОДА', 'БОЛЬ', 'ТУАЛЕТ'])
        self.assertEqual(len(self.page.card_buttons), 4)
        self.assertEqual(len(self.page._calibration_buttons()), 6)

    def test_four_cards_fill_equal_two_by_two_positions_and_are_wider(self):
        grid = self.page.matrix.layout()
        self.assertEqual(grid.count(), 4)
        for index, button in enumerate(self.page.card_buttons):
            self.assertIs(grid.itemAtPosition(index // 2, index % 2).widget(), button)
        widths = [button.width() for button in self.page.card_buttons]
        heights = [button.height() for button in self.page.card_buttons]
        self.assertLessEqual(max(widths) - min(widths), 1)
        self.assertLessEqual(max(heights) - min(heights), 1)
        self.assertGreater(min(widths), self.page.matrix.width() * .45)
        self.assertGreater(min(heights), self.page.matrix.height() * .45)

    def test_word_grid_uses_over_sixty_percent_and_large_gaze_targets(self):
        usable_height = self.page.height() - self.page.guidance.height() - 100
        self.assertGreater(self.page.grid_panel.height(), usable_height * 0.60)
        self.assertGreaterEqual(min(button.height() for button in self.page.card_buttons), 76)
        self.assertGreaterEqual(self.page.previous_button.height(), 58)
        self.assertEqual(self.page.category_label.text(), 'КАТЕГОРИЯ [1 / 6]: ЭКСТРЕННАЯ ПОМОЩЬ / ПОТРЕБНОСТИ')

    def test_manual_controls_append_speak_clear_and_wrap_categories(self):
        with patch.object(self.page, '_speak') as speak:
            self.page.card_buttons[2].click()
        self.assertEqual(self.page.output.toPlainText(), 'БОЛЬ ')
        speak.assert_called_once_with('БОЛЬ')
        with patch.object(self.page.speech, 'speak') as speak:
            self.page.btn_speak.click()
            speak.assert_called_once_with('БОЛЬ')
        self.page.btn_clear.click()
        self.assertEqual(self.page.output.toPlainText(), '')
        self.page.previous_button.click()
        self.assertEqual(self.page.category_index, 5)
        self.page.next_button.click()
        self.assertEqual(self.page.category_index, 0)

    def test_manual_navigation_reaches_every_word_and_returns_to_first_sheet(self):
        expected = []
        widgets = list(self.page.card_buttons)
        with patch.object(self.page, '_speak') as speak:
            for index, (_, words) in enumerate(AAC_CATEGORIES):
                self.assertEqual(self.page.category_index, index)
                self.assertEqual([button.text() for button in widgets], words)
                self.assertIn(f'[{index + 1} / 6]', self.page.category_label.text())
                for button in widgets:
                    button.click()
                expected.extend(words)
                self.page.next_button.click()
        self.assertEqual(self.page.category_index, 0)
        self.assertEqual(self.page.card_buttons, widgets)
        self.assertEqual(self.page.output.toPlainText(), ''.join(word + ' ' for word in expected))
        self.assertEqual([call.args[0] for call in speak.call_args_list], expected)
        for index in reversed(range(len(AAC_CATEGORIES))):
            self.page.previous_button.click()
            self.assertEqual(self.page.category_index, index)

    def test_gaze_selects_all_24_words_and_both_arrows_without_recalibration(self):
        driver = self.calibrated(13)
        model = self.page.calibration_model
        expected = []

        def look_until(point, changed):
            deadline = driver.user.time + 3.0
            while not changed() and driver.user.time < deadline:
                driver.frame(point)
            self.assertTrue(changed(), self.page.guidance.text())

        with patch.object(self.page, '_speak') as speak:
            for index, (_, words) in enumerate(AAC_CATEGORIES):
                self.assertEqual(self.page.category_index, index)
                for position, word in enumerate(words):
                    before = self.page.output.toPlainText()
                    look_until(driver.centers()[position + 2],
                               lambda: self.page.output.toPlainText() != before)
                    self.assertEqual(self.page.output.toPlainText(), before + word + ' ')
                    expected.append(word)
                look_until(driver.centers()[1], lambda: self.page.category_index != index)
                self.assertIs(self.page.calibration_model, model)
                self.assertFalse(self.page.calibration_active)
                self.assertEqual(self.page.category_index, (index + 1) % 6)
                # Remaining on an arrow during cooldown cannot skip a sheet.
                driver.look(driver.centers()[1], .4)
                self.assertEqual(self.page.category_index, (index + 1) % 6)
            self.assertEqual(self.page.output.toPlainText(), ''.join(word + ' ' for word in expected))
            self.assertEqual([call.args[0] for call in speak.call_args_list], expected)
            look_until(driver.centers()[0], lambda: self.page.category_index == 5)
            self.assertEqual(self.page.category_index, 5)
            self.assertIs(self.page.calibration_model, model)

    def test_manual_page_change_clears_partial_dwell_and_requires_a_fresh_fixation(self):
        self.calibrated(14)
        model = self.page.calibration_model
        features = model.model.means[4].copy()  # Bottom-left card, not its current word.
        self.page.dwell_engine.reset()
        for frame in range(13):
            self.page._update_selection(features, 50 + frame / 30)
        self.assertGreater(self.page.card_buttons[2].progress, .4)
        self.page.next_button.click()
        self.assertEqual(self.page.card_buttons[2].text(), 'НЕ ПОНИМАЮ')
        self.assertIsNone(self.page.current_target)
        self.assertTrue(all(button.progress == 0 for button in self.page.card_buttons))
        self.assertIs(self.page.calibration_model, model)
        with patch.object(self.page, '_speak') as speak:
            for frame in range(14, 41):
                self.page._update_selection(features, 50 + frame / 30)
            self.assertEqual(self.page.output.toPlainText(), '')
            speak.assert_not_called()
            for frame in range(41, 56):
                self.page._update_selection(features, 50 + frame / 30)
            self.assertEqual(self.page.output.toPlainText(), 'НЕ ПОНИМАЮ ')
            speak.assert_called_once_with('НЕ ПОНИМАЮ')

    def test_calibration_warning_names_current_sheet_words_after_navigation(self):
        self.calibrated(15)
        with patch.object(self.page.calibration_model, 'weak_pairs', return_value=[(2, 3)]):
            self.page._refresh_calibration_warning()
            self.assertIn('«ПОМОЩЬ» и «ВОДА»', self.page.calibration_warning)
            self.page.next_button.click()
            self.assertIn('«ДА» и «НЕТ»', self.page.calibration_warning)
            self.assertNotIn('ПОМОЩЬ', self.page.calibration_warning)

    def test_calibration_never_errors_and_all_six_targets_select(self):
        driver = self.calibrated(0)
        self.assertEqual(driver.badges, ['CALIBRATING', 'ACTIVE'])
        stages = ' '.join(driver.prompts)
        for word in ('КАЛИБРОВКА 1/6', 'ДВИЖЕНИЕ ГОЛОВЫ', 'ПРОВЕРКА 6/6'):
            self.assertIn(word, stages)
        self.assertTrue(self.page.calibration_model.model.learned_head_motion)
        self.assertTrue(self.page.btn_calibrate.isEnabled())
        driver.user.drift(1.5)
        buttons = self.page._calibration_buttons()
        self.assertEqual(self.page.calibration_model.model.means.shape, (6, 4))
        for index in (2, 4, 0, 1, 3, 5):
            with self.subTest(button=buttons[index].text() or index):
                self.assertEqual(self.selections(driver, driver.centers()[index]), [buttons[index]])

    def test_eye_only_and_head_pointing_users_select_all_four_card_positions(self):
        for seed, head_gain in ((1, 0.0), (2, 0.8)):
            with self.subTest(head_gain=head_gain):
                self.page._begin_calibration_state()
                driver = PageDriver(self.page, VirtualUser(seed, head_gain=head_gain))
                self.page.frames_seen = True
                driver.calibrate()
                buttons = self.page._calibration_buttons()
                for index in range(2, len(buttons)):
                    self.assertEqual(self.selections(driver, driver.centers()[index]), [buttons[index]])

    def test_noisy_low_fps_blinking_calibration_completes_without_pause_or_error(self):
        self.page._begin_calibration_state()
        self.page.frames_seen = True
        driver = PageDriver(self.page, VirtualUser(3, noise=0.45, fps=20.0))
        duration = driver.calibrate(blink_every=30)
        self.assertLess(duration, 60)
        self.assertEqual(driver.badges, ['CALIBRATING', 'ACTIVE'])
        self.assertFalse(any('ERROR' in badge or 'PAUSED' in badge for badge in driver.badges))
        buttons = self.page._calibration_buttons()
        self.assertEqual(self.selections(driver, driver.centers()[3], 3.0)[:1], [buttons[3]])

    def test_contaminated_capture_window_is_refined_then_selects_correctly(self):
        self.page._begin_calibration_state()
        self.page.frames_seen = True
        driver = PageDriver(self.page, VirtualUser(4))
        driver.calibrate(misdirect={('capture', 3): 5})
        self.assertIn('УТОЧНЕНИЕ', ' '.join(driver.prompts))
        buttons = self.page._calibration_buttons()
        self.assertEqual(self.selections(driver, driver.centers()[3]), [buttons[3]])
        self.assertEqual(self.selections(driver, driver.centers()[5]), [buttons[5]])

    def test_restless_gaze_still_progresses_through_calibration(self):
        self.page._begin_calibration_state()
        self.page.frames_seen = True
        user = VirtualUser(5)
        original = user.sample

        def restless(point, **kwargs):
            # Alternate small saccades for the first seconds of every button.
            offset = .05 if int(user.time * 4) % 2 else -.05
            started = self.page.calibration_started
            if started is not None and user.time - started < 3.0:
                point = point + [offset, 0]
            return original(point, **kwargs)

        user.sample = restless
        driver = PageDriver(self.page, user)
        self.assertLess(driver.calibrate(), 90)
        self.assertEqual(self.page.badge.text(), 'ACTIVE')

    def test_gaze_between_buttons_or_off_grid_does_not_select(self):
        driver = self.calibrated(6)
        centers = driver.centers()
        self.assertEqual(self.selections(driver, np.array([.3, .15]), 3.0), [])
        self.assertEqual(self.selections(driver, np.array([.5, -.1]), 3.0), [])
        self.assertLessEqual(len(self.selections(driver, (centers[3] + centers[5]) / 2, 3.0)), 1)

    def test_face_missing_and_startup_states_explain_without_collecting(self):
        worker = Mock(error='', reason='stopped')
        worker.take_pending.return_value = (None, [])
        with patch('app.GazeCaptureWorker', return_value=worker):
            self.page.btn_tracking.click()
        self.addCleanup(lambda: setattr(self.page, 'worker', None))
        self.assertIn('Запуск камеры', self.page.guidance.text())
        self.assertEqual(self.page.badge.text(), 'CALIBRATING')
        self.assertFalse(self.page.btn_calibrate.isEnabled())
        user = VirtualUser(7)
        image = np.zeros((240, 320, 3), np.uint8)
        sample = user.sample(np.array([.5, .5]), face=False)
        worker.take_pending.return_value = ({'image': image, **sample}, [sample])
        self.page._poll()
        self.assertIn('КАЛИБРОВКА 1/6', self.page.guidance.text())
        self.assertIn('Держите лицо в кадре', self.page.guidance.text())
        self.assertIsNone(self.page.calibration_started)
        driver = PageDriver(self.page, user)
        driver.look(driver.centers()[0], 2.2)
        self.assertEqual(self.page.calibration_queue[0], 1)

    def test_blink_during_dwell_keeps_progress_and_long_loss_resets(self):
        driver = self.calibrated(8)
        point = driver.centers()[4]
        with patch.object(self.page, '_activate_target') as activate:
            self.page.dwell_engine.reset()
            driver.look(point, 0.5)
            progress = self.page.card_buttons[2].progress
            self.assertGreater(progress, .2)
            driver.look(point, 0.13, blink=True)
            self.assertGreaterEqual(self.page.card_buttons[2].progress, progress)
            driver.look(point, 0.6)
            activate.assert_called_once_with(self.page.card_buttons[2])
            driver.look(point, 1.0)
            before = self.page.card_buttons[2].progress
            self.assertGreater(before, 0)
            driver.look(point, 0.5, face=False)
            self.assertEqual(self.page.card_buttons[2].progress, 0)
            self.assertIsNone(self.page.current_target)

    def test_poll_uses_child_timestamps_only_and_stall_resets_dwell(self):
        driver = self.calibrated(9)
        worker = Mock()
        self.page.worker = worker
        self.addCleanup(lambda: setattr(self.page, 'worker', None))
        image = np.zeros((240, 320, 3), np.uint8)
        driver.user.time = 5e6  # child clock unrelated to the GUI clock
        point = driver.centers()[5]
        for _ in range(12):
            sample = driver.user.sample(point)
            worker.take_pending.return_value = ({'image': image, **sample}, [sample])
            self.page._poll()
        self.assertIs(self.page.current_target, self.page.card_buttons[3])
        self.assertEqual(self.page.gaze_x.val.text(), f'{sample["gaze"][0]:.3f}')
        self.assertGreater(int(self.page.dwell_display.val.text()), 0)
        worker.take_pending.return_value = (None, [])
        with patch('app.time.perf_counter', return_value=time.perf_counter() + 1.0):
            self.page._poll()
        self.assertIsNone(self.page.current_target)
        self.assertEqual(self.page.gaze_x.val.text(), '---')

    def test_resize_and_category_change_during_calibration_keep_completed_buttons(self):
        self.page._begin_calibration_state()
        self.page.frames_seen = True
        driver = PageDriver(self.page, VirtualUser(10))
        driver.look(driver.centers()[0], 2.2)
        self.assertEqual(len(self.page.calibration_windows[0]), 1)
        self.page.resize(1500, 950)
        self.qt.processEvents()
        self.page.next_button.click()
        self.assertEqual(len(self.page.calibration_windows[0]), 1)
        self.assertEqual(self.page.calibration_queue[0], 1)
        self.assertIsNone(self.page.calibration_started)
        driver.calibrate()
        self.assertEqual(self.page.badge.text(), 'ACTIVE')

    def test_recalibration_replaces_model_and_controls_follow_state(self):
        driver = self.calibrated(11)
        first = self.page.calibration_model
        worker = Mock()
        self.page.worker = worker
        self.addCleanup(lambda: setattr(self.page, 'worker', None))
        self.page.btn_calibrate.click()
        self.assertTrue(self.page.calibration_active)
        self.assertFalse(self.page.btn_calibrate.isEnabled())
        self.assertEqual(self.selections(driver, driver.centers()[3], 1.5), [])
        driver.calibrate()
        self.assertIsNot(self.page.calibration_model, first)
        self.assertTrue(self.page.btn_calibrate.isEnabled())

    def test_trigger_flash_and_progress_do_not_move_or_restyle_widgets(self):
        driver = self.calibrated(12)
        buttons = self.page._calibration_buttons()
        before = [(button.geometry(), button.font().toString(), button.styleSheet()) for button in buttons]
        with patch.object(self.page, '_speak'):
            self.page.dwell_engine.reset()
            driver.look(driver.centers()[2], 1.2)
        self.assertEqual(self.page.output.toPlainText(), 'ПОМОЩЬ ')
        self.assertGreater(buttons[2].flash_until, 0)
        after = [(button.geometry(), button.font().toString(), button.styleSheet()) for button in buttons]
        self.assertEqual(before, after)

    def test_start_stop_error_and_restart_wait_for_resource_release(self):
        worker = Mock(error='', reason='stopped')
        worker.take_pending.return_value = (None, [])
        with patch('app.GazeCaptureWorker', return_value=worker):
            self.page.btn_tracking.click()
        worker.start.assert_called_once()
        self.assertTrue(self.page.calibration_active)
        self.page.btn_stop_tracking.click()
        worker.requestInterruption.assert_called_once()
        self.assertFalse(self.page.timer.isActive())
        self.assertFalse(self.page.btn_tracking.isEnabled())
        self.assertFalse(self.page.calibration_active)
        self.page._tracking_finished()
        self.assertIsNone(self.page.worker)
        self.assertTrue(self.page.btn_tracking.isEnabled())
        self.assertEqual(self.page.badge.text(), 'IDLE')
        self.page.worker = worker
        self.page._capture_failed('Ошибка камеры')
        self.page._tracking_finished()
        self.assertIn('Ошибка камеры', self.page.guidance.text())
        self.assertEqual(self.page.badge.text(), 'TRACKING ERROR')

    def test_camera_navigation_and_window_close_include_aac(self):
        window = MainWindow()
        worker = Mock(error='', reason='stopped')
        window.aac_page.worker = worker
        window.switch_page(4)
        window.switch_page(1)
        self.assertEqual(window.stack.currentIndex(), 4)
        event = Mock()
        window.closeEvent(event)
        event.ignore.assert_called_once()
        worker.requestInterruption.assert_called_once()
        window.aac_page._tracking_finished()
        self.qt.processEvents()
        self.assertIsNone(window.aac_page.worker)
        window.close()


class FakeSpeechEngine:
    def __init__(self):
        self.said = []
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.stopped = False
        self.busy = False

    def record(self, name):
        self.calls.append((name, threading.get_ident()))

    def getProperty(self, name):
        self.record(name)
        return [SimpleNamespace(id='ru', name='Russian', languages=[b'\x05ru-RU'])]

    def setProperty(self, name, value):
        self.record(name)
        self.voice = value

    def connect(self, *args):
        self.record('connect')
        self.callback = args[1]

    def startLoop(self, *args):
        self.record('startLoop')

    def say(self, text):
        self.record('say')
        self.said.append(text)
        self.busy = True
        self.started.set()

    def iterate(self):
        self.record('iterate')
        if self.release.is_set() or self.stopped:
            self.busy = False

    def isBusy(self):
        return self.busy

    def stop(self):
        self.record('stop')
        self.stopped = True

    def endLoop(self):
        self.record('endLoop')


class SpeechWorkerTests(unittest.TestCase):
    def test_russian_language_metadata_and_missing_voice(self):
        for language in (b'\x05ru', 'ru-RU', 'ru_RU', '419'):
            self.assertEqual(russian_voice([SimpleNamespace(id='voice', languages=[language])]), 'voice')
        with self.assertRaisesRegex(RuntimeError, 'русский голос'):
            russian_voice([SimpleNamespace(id='en', name='English', languages=['en-US'])])

    def test_serialized_daemon_speech_never_runs_on_caller_and_can_be_cancelled(self):
        engine = FakeSpeechEngine()
        owner = threading.get_ident()
        worker = SpeechWorker(lambda: engine)
        try:
            worker.speak('ПОМОЩЬ')
            self.assertTrue(engine.started.wait(2))
            self.assertTrue(worker.thread.daemon)
            worker.speak('ВОДА')
            self.assertEqual(engine.said, ['ПОМОЩЬ'])
            worker.cancel()
            worker.close()
            worker.thread.join(2)
            self.assertFalse(worker.thread.is_alive())
            self.assertTrue(engine.stopped)
            self.assertTrue(all(thread != owner for _, thread in engine.calls))
            self.assertEqual(len({thread for _, thread in engine.calls}), 1)
            self.assertEqual(engine.voice, 'ru')
            self.assertEqual(worker.take_errors(), [])
        finally:
            engine.release.set()
            worker.close()
            worker.thread.join(2)

    def test_missing_voice_is_reported_without_gui_thread_exception(self):
        engine = FakeSpeechEngine()
        engine.getProperty = lambda name: []
        worker = SpeechWorker(lambda: engine)
        worker.speak('ПОМОЩЬ')
        worker.thread.join(2)
        self.assertFalse(worker.thread.is_alive())
        self.assertIn('русский голос', worker.take_errors()[0])
        self.assertEqual(engine.said, [])
        worker.close()

    def test_driver_error_callback_is_reported_with_pyttsx3_keyword_arguments(self):
        engine = FakeSpeechEngine()
        worker = SpeechWorker(lambda: engine)
        try:
            worker.speak('ПОМОЩЬ')
            self.assertTrue(engine.started.wait(2))
            engine.callback(name='utterance', exception=RuntimeError('driver error'))
            self.assertIn('driver error', worker.take_errors()[0])
        finally:
            worker.close()
            worker.thread.join(2)


if __name__ == '__main__':
    unittest.main()
