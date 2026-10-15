import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import time
import unittest
from unittest.mock import patch

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QCoreApplication, QEvent

from app import AACPage, AAC_CATEGORIES, AAC_WORDS_PER_SHEET
from gaze_engine import DwellState
from phrase_graph import PhraseGraph, ROOT_WORDS, _EDGES


VOCABULARY = {word for _, words in AAC_CATEGORIES for word in words}


class PhraseGraphTests(unittest.TestCase):
    def setUp(self):
        self.graph = PhraseGraph()

    def test_all_24_words_have_valid_meaningful_weighted_transitions(self):
        self.assertEqual(len(VOCABULARY), 24)
        self.assertEqual(set(self.graph.all_words()), VOCABULARY)
        self.assertEqual(set(_EDGES), VOCABULARY)
        for word in VOCABULARY:
            edges = self.graph.edges_from(word)
            self.assertGreaterEqual(len(edges), 6)
            self.assertEqual(len({target for target, _ in edges}), len(edges))
            for target, weight in edges:
                self.assertIn(target, VOCABULARY)
                self.assertNotEqual(target, word)
                self.assertGreaterEqual(weight, .5)
                self.assertLessEqual(weight, 1.)
            predictions = self.graph.predict(word)
            self.assertEqual(len(predictions), 4)
            self.assertEqual(len(set(predictions)), 4)
            self.assertNotIn(word, predictions)
            self.assertEqual(predictions, [target for target, _ in sorted(edges, key=lambda e: -e[1])[:4]])

    def test_care_chains_have_expected_priorities(self):
        self.assertEqual(self.graph.predict('БОЛЬ'), ['ТАБЛЕТКА', 'ВРАЧ', 'ПОМОЩЬ', 'ДА'])
        self.assertEqual(self.graph.predict('ТАБЛЕТКА'), ['ВОДА', 'СПАСИБО', 'ВРАЧ', 'ДА'])
        self.assertEqual(self.graph.predict('ПЕРЕВЕРНУТЬ'), ['ПОДНЯТЬ ГОЛОВУ', 'СПАСИБО', 'БОЛЬ', 'ХОРОШО'])

    def test_short_adjacency_falls_back_without_duplicates_or_selected_word(self):
        self.graph.adjacency['ПОМОЩЬ'] = [('ВОДА', 1.), ('ПОМОЩЬ', .9), ('ВОДА', .8)]
        self.assertEqual(self.graph.predict('ПОМОЩЬ'), ['ВОДА', 'БОЛЬ', 'ДА', 'НЕТ'])
        self.graph.adjacency['БОЛЬ'] = []
        self.assertEqual(self.graph.predict('БОЛЬ'), ['ПОМОЩЬ', 'ВОДА', 'ДА', 'НЕТ'])
        self.assertEqual(self.graph.predict('UNKNOWN'), ROOT_WORDS[:4])

    def test_count_limits_and_return_values_do_not_mutate_graph(self):
        self.assertEqual(self.graph.predict('БОЛЬ', 0), [])
        self.assertEqual(len(self.graph.predict('БОЛЬ', 6)), 6)
        with self.assertRaises(ValueError):
            self.graph.predict('БОЛЬ', -1)
        words = self.graph.predict('БОЛЬ')
        words.clear()
        self.assertEqual(len(self.graph.predict('БОЛЬ')), 4)


class PredictiveAACPageTests(unittest.TestCase):
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
        self.page = None
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.qt.processEvents()

    def words(self):
        return [button.text() for button in self.page.card_buttons]

    def test_start_uses_original_four_word_sheet_and_no_predictive_context(self):
        self.assertEqual(len(self.page.card_buttons), AAC_WORDS_PER_SHEET)
        self.assertEqual(self.words(), AAC_CATEGORIES[0][1])
        self.assertFalse(self.page.is_predictive)
        self.assertIsNone(self.page.predictive_context)

    def test_selection_speaks_displayed_word_and_subsequent_prediction_not_category_word(self):
        with patch.object(self.page, '_speak') as speech:
            self.page.card_buttons[2].click()
            self.assertEqual(self.words(), ['ТАБЛЕТКА', 'ВРАЧ', 'ПОМОЩЬ', 'ДА'])
            self.page.card_buttons[0].click()
            self.assertEqual(self.page.output.toPlainText(), 'БОЛЬ ТАБЛЕТКА ')
            self.assertEqual([call.args[0] for call in speech.call_args_list], ['БОЛЬ', 'ТАБЛЕТКА'])
        self.assertTrue(self.page.is_predictive)
        self.assertEqual(self.page.predictive_context, 'ТАБЛЕТКА')
        self.assertEqual(self.words(), self.page.phrase_graph.predict('ТАБЛЕТКА'))
        self.assertEqual(self.page.category_label.text(), 'ПРЕДИКТИВНЫЙ КОНТЕКСТ: [ ТАБЛЕТКА ]')
        self.assertEqual(self.page.previous_button.text(), '[ ⟲ В КОРЕНЬ ]')

    def test_root_arrow_restores_same_category_without_clearing_buffer(self):
        self.page._set_category(4)
        with patch.object(self.page, '_speak'):
            self.page.card_buttons[1].click()
            before = self.page.output.toPlainText()
            self.page._activate_target(self.page.previous_button)
        self.assertFalse(self.page.is_predictive)
        self.assertIsNone(self.page.predictive_context)
        self.assertEqual(self.page.category_index, 4)
        self.assertEqual(self.words(), AAC_CATEGORIES[4][1])
        self.assertEqual(self.page.output.toPlainText(), before)
        self.assertEqual(self.page.previous_button.text(), '< ПРЕДЫДУЩАЯ КАТЕГОРИЯ')
        self.assertEqual(self.page.category_label.text(), 'КАТЕГОРИЯ [5 / 6]: МЕДИЦИНА / БЫТ')

    def test_right_arrow_exits_prediction_and_browses_all_standard_categories(self):
        with patch.object(self.page, '_speak'):
            self.page.card_buttons[0].click()
        self.page.next_button.click()
        self.assertFalse(self.page.is_predictive)
        self.assertEqual(self.page.category_index, 1)
        self.assertEqual(self.words(), AAC_CATEGORIES[1][1])
        for _ in range(5):
            self.page.next_button.click()
        self.assertEqual(self.page.category_index, 0)

    def test_clear_empties_output_cancels_queued_speech_and_restores_current_sheet(self):
        self.page._set_category(3)
        with patch.object(self.page, '_speak'):
            self.page.card_buttons[1].click()
        with patch.object(self.page.speech, 'cancel') as cancel:
            self.page.btn_clear.click()
        cancel.assert_called_once()
        self.assertEqual(self.page.output.toPlainText(), '')
        self.assertFalse(self.page.is_predictive)
        self.assertEqual(self.words(), AAC_CATEGORIES[3][1])

    def test_predictions_keep_calibration_model_and_clear_dwell_without_shortening_cooldown(self):
        sentinel = object()
        self.page.calibration_model = sentinel
        with patch.object(self.page, '_refresh_calibration_warning'), patch.object(self.page, '_speak'):
            self.page.dwell_engine.target = self.page.card_buttons[2]
            self.page.dwell_engine.started = 50
            self.page.dwell_engine.last_timestamp = 50.8
            self.page.dwell_engine.cooldown_until = 51.5
            self.page.card_buttons[2].set_dwell(.8, True)
            self.page.card_buttons[2].click()
        self.assertIs(self.page.calibration_model, sentinel)
        self.assertEqual(self.page.dwell_engine.cooldown_until, 51.5)
        self.assertIsNone(self.page.current_target)
        self.assertTrue(all(button.progress == 0 for button in self.page.card_buttons))
        self.assertEqual(self.page.dwell_engine.state, DwellState.COOLDOWN)

    def test_gaze_predicted_word_and_root_recovery_observe_900ms_dwell_and_500ms_cooldown(self):
        from gaze_engine import fit_gaze_classifier, GazeTargeting
        from test_gaze import target_windows
        import numpy as np
        means = np.array([[.25, .6, 0, .3], [.75, .6, 0, .3], [.4, .4, 0, .3],
                          [.6, .4, 0, .3], [.4, .5, 0, .3], [.6, .5, 0, .3]])
        self.page.calibration_model = GazeTargeting(fit_gaze_classifier(target_windows(means, np.random.default_rng(5))))
        with patch.object(self.page, '_speak') as speech:
            for frame in range(28):
                self.page._update_selection(means[4], 10 + frame/30)
            self.assertEqual(self.page.output.toPlainText(), 'БОЛЬ ')
            for frame in range(28, 42):
                self.page._update_selection(means[2], 10 + frame/30)
            self.assertEqual(self.page.output.toPlainText(), 'БОЛЬ ')
            for frame in range(42, 71):
                self.page._update_selection(means[2], 10 + frame/30)
            self.assertEqual(self.page.output.toPlainText(), 'БОЛЬ ТАБЛЕТКА ')
            start = 13
            for frame in range(28):
                self.page._update_selection(means[0], start + frame/30)
            self.assertFalse(self.page.is_predictive)
            self.assertEqual(self.words(), AAC_CATEGORIES[0][1])
            self.assertEqual(speech.call_count, 2)

    def test_calibration_restores_standard_labels_and_manual_click_does_not_relabel_targets(self):
        with patch.object(self.page, '_speak'):
            self.page.card_buttons[2].click()
            self.page._begin_calibration_state()
            self.assertFalse(self.page.is_predictive)
            self.assertEqual(self.words(), AAC_CATEGORIES[0][1])
            self.page.card_buttons[2].click()
        self.assertFalse(self.page.is_predictive)
        self.assertEqual(self.words(), AAC_CATEGORIES[0][1])
        self.assertEqual(len(self.page._calibration_buttons()), 6)

    def test_all_word_contexts_preserve_geometry_typography_and_style(self):
        widgets = self.page._calibration_buttons() + [self.page.matrix, self.page.category_label]
        before = [(widget.geometry(), widget.font().toString(), widget.styleSheet()) for widget in widgets]
        with patch.object(self.page, '_speak'):
            for category, (_, words) in enumerate(AAC_CATEGORIES):
                for index, word in enumerate(words):
                    self.page._set_category(category)
                    self.page._select_word(index)
                    self.qt.processEvents()
                    self.assertEqual(self.page.predictive_context, word)
                    self.assertEqual(before, [(widget.geometry(), widget.font().toString(), widget.styleSheet()) for widget in widgets])

    def test_prediction_and_label_update_performance(self):
        durations = []
        with patch.object(self.page, '_speak'):
            for _ in range(200):
                self.page.output.clear()
                started = time.perf_counter_ns()
                self.page._show_predictions('БОЛЬ')
                durations.append((time.perf_counter_ns() - started)/1e6)
        ordered = sorted(durations)
        p95 = ordered[int(len(ordered) * .95)]
        self.assertLess(p95, 2., f'Prediction/UI update p95: {p95:.3f} ms')


if __name__ == '__main__':
    unittest.main()
