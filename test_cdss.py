import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from copy import deepcopy
import tempfile
import threading
import time
from pathlib import Path
import unittest
from unittest.mock import patch

import networkx as nx
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtTest import QSignalSpy, QTest
from PyQt6.QtWidgets import QApplication

from app import MainWindow
from cdss_dashboard import CDSS_DISCLAIMER, SCORE_NOTE, summary_text
from session_diagnostics import (SessionDiagnostics, RELATIONS, HYPOTHESES,
                                 standardize, build_synthesis, ontology)


RESULTS = {
    'tremor': {'dominant_freq': 5., 'concentration': 80., 'rms_px': 1.},
    'facial': {'asymmetry_index': 14., 'smile_imbalance': 25., 'brow_imbalance': 8.},
    'voice': {'f0': 180., 'jitter_local': 1.2, 'shimmer_local': 4., 'hnr': 19.},
    'motor': {'tap_frequency': 2.1, 'amplitude_decrement': -30., 'rhythm_cv': 25.},
    'rppg': {'bpm': 110., 'concentration': 80., 'roi_bpm': [110., 111., 110.5],
             'used_rois': [0, 1, 2], 'coverage': .98, 'warnings': []},
}


def snapshot(results=RESULTS):
    return {key: standardize(key, value) for key, value in results.items()}


def wait_until(predicate, timeout=3000):
    deadline = time.perf_counter() + timeout/1000
    while not predicate() and time.perf_counter() < deadline:
        QTest.qWait(10)
    if not predicate():
        raise AssertionError('Asynchronous update timed out')


class InferenceTests(unittest.TestCase):
    def test_baseline_is_bipartite_dag_and_signed_relations_are_preserved(self):
        graph = ontology()
        self.assertIsInstance(graph, nx.DiGraph)
        self.assertTrue(nx.is_directed_acyclic_graph(graph))
        for source, target, relation, weight in RELATIONS:
            self.assertEqual(graph.nodes[source]['bipartite'], 0)
            self.assertEqual(graph.nodes[target]['bipartite'], 1)
            self.assertEqual(graph[source][target]['weight'], weight)
            self.assertEqual(graph[source][target]['type'], relation)
        self.assertFalse(any(d['active'] for _, d in graph.nodes(data=True) if d['category'] == 'biomarker'))

    def test_empty_session_does_not_invent_normality_or_probabilities(self):
        result = build_synthesis({})
        self.assertEqual(result['status'], 'empty')
        self.assertEqual(result['graph'].number_of_nodes(), 0)
        self.assertEqual(result['ranking'], [])
        self.assertEqual(len(result['missing']), 5)

    def test_multimodal_evidence_accumulates_and_exclusion_penalizes(self):
        result = build_synthesis(snapshot())
        self.assertEqual(result['graph'].nodes['SYNDROME_PARKINSONIAN']['score'], 100)
        self.assertEqual(result['graph'].nodes['SYNDROME_ESSENTIAL_TREMOR']['score'], 0)
        self.assertIn('MOTOR_CADENCE_LOW', result['active'])
        self.assertEqual(result['graph'].out_degree('MOTOR_CADENCE_LOW'), 0)
        action_results = deepcopy(RESULTS)
        action_results['tremor']['dominant_freq'] = 8
        result = build_synthesis(snapshot(action_results))
        score = result['graph'].nodes['SYNDROME_ESSENTIAL_TREMOR']['score']
        self.assertAlmostEqual(score, 100 * (.88 - .75)/.88)
        candidate = next(r for r in result['ranking'] if r['id'] == 'SYNDROME_ESSENTIAL_TREMOR')
        self.assertEqual(candidate['opposing'], ['MOTOR_DECREMENT_HIGH'])

    def test_single_module_does_not_receive_multimodal_full_support(self):
        result = build_synthesis(snapshot({'tremor': RESULTS['tremor']}))
        self.assertAlmostEqual(result['ranking'][0]['score'], 100 * .85/2.4)
        self.assertEqual(result['ranking'][0]['modules'], ['tremor'])

    def test_threshold_boundaries_and_tremor_quality(self):
        for frequency, concentration, rms, key in (
            (4, 50, .2, 'TREMOR_REST_4_6HZ'), (6.5, 80, 1, 'TREMOR_ACTION_6_12HZ'),
            (12, 80, 1, 'TREMOR_ACTION_6_12HZ'), (12.1, 80, 1, None),
            (5, 49.9, 1, None), (5, 80, .19, None), (None, 80, 1, None)):
            data = {'dominant_freq': frequency, 'concentration': concentration, 'rms_px': rms}
            result = build_synthesis(snapshot({'tremor': data}))
            self.assertEqual(set(result['active']), {key} if key else set())
        for decrement, frequency, expected in ((-20, 2.5, set()), (-20.1, 2.5, {'MOTOR_DECREMENT_HIGH'}),
                                                (0, 2.49, {'MOTOR_CADENCE_LOW'})):
            data = {'amplitude_decrement': decrement, 'tap_frequency': frequency, 'rhythm_cv': 0}
            self.assertEqual(set(build_synthesis(snapshot({'motor': data}))['active']), expected)

    def test_face_and_voice_thresholds_do_not_assign_stroke_or_paresis_diagnosis(self):
        for smile, brow, expected in ((18, 0, None), (18.1, 11.9, 'FACE_LOWER_PARESIS'),
                                       (18.1, 12, 'FACE_COMPLETE_PARESIS')):
            data = {'asymmetry_index': 5, 'smile_imbalance': smile, 'brow_imbalance': brow}
            result = build_synthesis(snapshot({'facial': data}))
            self.assertEqual(set(result['active']), {expected} if expected else set())
            if expected:
                self.assertIn('экстренной', result['action'])
                self.assertIn('не является FAST+', result['action'])
        for jitter, shimmer, active in ((1.04, 3.81, False), (1.041, 1, True), (.5, 3.811, True)):
            data = dict(RESULTS['voice'], jitter_local=jitter, shimmer_local=shimmer)
            self.assertEqual(bool(build_synthesis(snapshot({'voice': data}))['active']), active)

    def test_optical_quality_and_baseline_stability_gate_hr_evidence(self):
        for bpm, expected in ((59.9, 'BRADYCARDIA_RESTING'), (60, None), (95, None), (95.1, 'TACHYCARDIA_RESTING')):
            data = dict(RESULTS['rppg'], bpm=bpm)
            result = build_synthesis(snapshot({'rppg': data}))
            self.assertEqual(set(result['active']), {expected} if expected else set())
        for changes in ({'warnings': ['Head drift']}, {'concentration': 59}, {'coverage': .5},
                        {'used_rois': [0]}, {'roi_bpm': [100, 120, 130]}):
            result = build_synthesis(snapshot({'rppg': dict(RESULTS['rppg'], **changes)}))
            self.assertFalse(result['active'])
            self.assertTrue(result['limited'])

    def test_normal_values_do_not_establish_no_pathology(self):
        data = deepcopy(RESULTS)
        data['tremor']['dominant_freq'] = 3
        data['facial']['smile_imbalance'] = 5
        data['voice'].update(jitter_local=.5, shimmer_local=1)
        data['motor'].update(tap_frequency=3, amplitude_decrement=0)
        data['rppg']['bpm'] = 72
        result = build_synthesis(snapshot(data))
        self.assertEqual(result['status'], 'no_flags')
        self.assertEqual(result['ranking'], [])
        self.assertEqual(result['graph'].nodes['SYNDROME_PHYSIOLOGICAL']['score'], 0)

    def test_standardized_store_copies_scalars_only_and_rejects_invalid_values(self):
        data = dict(RESULTS['voice'], source='file', file_name='vowel.wav', waveform=[1] * 10000)
        assessment = standardize('voice', data)
        self.assertNotIn('waveform', assessment)
        self.assertEqual(assessment['units']['f0'], 'Гц')
        self.assertEqual(assessment['source_file'], 'vowel.wav')
        for value in (float('nan'), -1):
            with self.assertRaises(ValueError):
                standardize('voice', dict(data, jitter_local=value))


class SessionAndUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.resize(1200, 750)
        self.window.show()
        self.session = self.window.session
        self.cdss = self.window.dashboard_page.cdss
        self.qt.processEvents()

    def tearDown(self):
        self.window.close()
        if self.session.worker.isRunning():
            self.session.worker.wait(3000)
        self.qt.processEvents()
        self.window.close()
        self.window.deleteLater()
        self.qt.processEvents()

    def publish_all(self):
        for module, data in RESULTS.items():
            self.session.publish(module, data)
        wait_until(lambda: self.session.result is not None)

    def test_all_modules_share_store_and_report_completion_events_publish(self):
        pages = {'tremor': self.window.tremor_page, 'facial': self.window.facial_page,
                 'voice': self.window.voice_page, 'motor': self.window.motor_page, 'rppg': self.window.rppg_page}
        for module, page in pages.items():
            self.assertIs(page.session, self.session)
            page.protocol_panel.assessment_completed.emit(RESULTS[module])
        wait_until(lambda: self.session.result is not None)
        self.assertEqual(set(self.session.assessments), set(RESULTS))
        self.assertTrue(self.cdss.export.isEnabled())
        self.assertGreater(self.cdss.score.value(), 0)

    def test_actual_report_completion_and_clear_signals_update_shared_store(self):
        from motor_engine import analyze_tapping
        from test_motor import tapping_samples
        tremor = dict(RESULTS['tremor'], fs=30., coverage=.98, max_gap_sec=.034, max_freq=12.)
        stage = {'samples': 60, 'rate': 30., 'width_px': 150.}
        facial = dict(RESULTS['facial'], rest=stage, smile=stage, brows=stage)
        voice = dict(RESULTS['voice'], jitter_rap=.2, shimmer_apq5=1., sample_rate=16000,
                     duration=5., clipping=0., mfcc_mean=[0.] * 13, voiced_fraction=.99,
                     level_dbfs=-18., warnings=[])
        motor = analyze_tapping(tapping_samples(), 0, 15)
        rppg = dict(RESULTS['rppg'], samples=900, rate=30., duration=30., max_gap=.034, resolution_bpm=2.)
        for page, result in ((self.window.tremor_page, tremor), (self.window.facial_page, facial),
                             (self.window.voice_page, voice), (self.window.motor_page, motor),
                             (self.window.rppg_page, rppg)):
            page.protocol_panel.set_results(result)
        wait_until(lambda: self.session.result is not None)
        self.assertEqual(set(self.session.assessments), set(RESULTS))
        for page in (self.window.tremor_page, self.window.facial_page, self.window.voice_page,
                     self.window.motor_page, self.window.rppg_page):
            page.protocol_panel.clear()
        wait_until(lambda: self.session.result is not None)
        self.assertFalse(self.session.assessments)
        self.assertEqual(self.session.result['status'], 'empty')

    def test_reset_prunes_result_but_closing_report_keeps_assessment(self):
        self.publish_all()
        panel = self.window.motor_page.protocol_panel
        panel.hide()
        self.assertIn('motor', self.session.assessments)
        self.window.motor_page.reset.click()
        wait_until(lambda: self.session.result is not None)
        self.assertNotIn('motor', self.session.assessments)
        self.assertNotIn('MOTOR_DECREMENT_HIGH', self.session.result['graph'])
        self.assertFalse(self.session.result['graph'].has_edge('MOTOR_DECREMENT_HIGH', 'SYNDROME_ESSENTIAL_TREMOR'))

    def test_new_assessment_replaces_old_nodes_and_missing_input_is_not_negative_evidence(self):
        self.session.publish('tremor', RESULTS['tremor'])
        wait_until(lambda: self.session.result is not None)
        self.session.publish('tremor', dict(RESULTS['tremor'], dominant_freq=8))
        wait_until(lambda: self.session.result is not None)
        self.assertNotIn('TREMOR_REST_4_6HZ', self.session.result['graph'])
        self.assertIn('TREMOR_ACTION_6_12HZ', self.session.result['graph'])
        self.assertNotIn('SYNDROME_PHYSIOLOGICAL', self.session.result['graph'])
        self.session.clear('tremor')
        wait_until(lambda: self.session.result is not None)
        self.assertEqual(self.session.result['status'], 'empty')
        self.assertFalse(self.cdss.export.isEnabled())

    def test_invalid_result_does_not_escape_qt_slot(self):
        spy = QSignalSpy(self.session.failed)
        self.session.publish('motor', {'tap_frequency': float('nan')})
        self.assertEqual(len(spy), 1)
        self.assertNotIn('motor', self.session.assessments)

    def test_worker_computation_is_off_gui_and_outdated_result_cannot_overwrite_new_revision(self):
        entered, release = threading.Event(), threading.Event()
        identities = []
        original = build_synthesis
        def slow(snapshot, revision):
            identities.append(threading.get_ident())
            if revision == 1:
                entered.set()
                release.wait(2)
            return original(snapshot, revision)
        with patch('session_diagnostics.build_synthesis', side_effect=slow):
            self.session.publish('motor', RESULTS['motor'])
            self.assertTrue(entered.wait(1))
            fired = []
            QTimer.singleShot(0, lambda: fired.append(True))
            self.qt.processEvents()
            self.assertTrue(fired)
            self.session.clear('motor')
            self.assertFalse(self.cdss.export.isEnabled())
            release.set()
            wait_until(lambda: self.session.result is not None)
        self.assertTrue(all(i != threading.get_ident() for i in identities))
        self.assertEqual(self.session.result['revision'], 2)
        self.assertEqual(self.session.result['status'], 'empty')

    def test_graph_empty_state_active_labels_and_minimum_geometry(self):
        self.assertEqual(self.cdss.height(), 260)
        self.assertEqual(len(self.cdss.graph_view.nodes), 15)
        self.assertIn('Ожидание', self.cdss.primary.text())
        self.publish_all()
        self.assertIn('Индекс поддержки', self.cdss.score.format())
        self.assertIn('не вероятность', self.cdss.note.text())
        self.assertIn('гипотеза', self.cdss.primary.text().lower())
        self.assertIn('Вклады правил', self.cdss.evidence.text())
        self.assertLess(self.cdss.geometry().bottom(), self.window.dashboard_page.catalog.geometry().top())
        self.assertLessEqual(self.cdss.export.mapTo(self.cdss, self.cdss.export.rect().bottomRight()).x(), self.cdss.width())

    def test_summary_pdf_contains_session_metrics_evidence_contradictions_and_disclaimer(self):
        self.publish_all()
        report = summary_text(self.session.result)
        for token in ('f_dom', 'jitter', 'rhythm_cv', 'baseline_stability', 'EXCLUDES', CDSS_DISCLAIMER, SCORE_NOTE):
            self.assertIn(token, report)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'summary.pdf'
            with patch('cdss_dashboard.QFileDialog.getSaveFileName', return_value=(str(path), '')):
                self.cdss.export.click()
            self.assertTrue(path.read_bytes().startswith(b'%PDF-'))


if __name__ == '__main__':
    unittest.main()
