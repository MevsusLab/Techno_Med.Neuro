import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
import tempfile
import time
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from session_diagnostics import build_synthesis, standardize
from patient_graph import (TemporalPatientGraph, _compute_delta, _compute_trend,
                           _jaccard_weighted, _REFERENCE_PROFILES)


RESULTS = {
    'tremor': {'dominant_freq': 5., 'concentration': 80., 'rms_px': 1.},
    'facial': {'asymmetry_index': 14., 'smile_imbalance': 25., 'brow_imbalance': 8.},
    'voice': {'f0': 180., 'jitter_local': 1.2, 'shimmer_local': 4., 'hnr': 19.},
    'motor': {'tap_frequency': 2.1, 'amplitude_decrement': -30., 'rhythm_cv': 25.},
    'rppg': {'bpm': 110., 'concentration': 80., 'roi_bpm': [110., 111., 110.5],
             'used_rois': [0, 1, 2], 'coverage': .98, 'warnings': []},
}

MILD_RESULTS = {
    'tremor': {'dominant_freq': 3., 'concentration': 40., 'rms_px': 0.1},
    'motor': {'tap_frequency': 3.5, 'amplitude_decrement': -5., 'rhythm_cv': 10.},
}


def snapshot(results=RESULTS):
    return {key: standardize(key, value) for key, value in results.items()}


class DeltaTrendTests(unittest.TestCase):
    def test_delta_computes_numeric_differences(self):
        prev = {'tremor': {'f_dom': 5.0, 'concentration': 80.0}}
        curr = {'tremor': {'f_dom': 4.5, 'concentration': 85.0}}
        delta = _compute_delta(prev, curr)
        self.assertAlmostEqual(delta['tremor.f_dom'], -0.5)
        self.assertAlmostEqual(delta['tremor.concentration'], 5.0)

    def test_delta_ignores_non_numeric_and_missing_keys(self):
        prev = {'tremor': {'f_dom': 5.0, 'label': 'test'}}
        curr = {'tremor': {'f_dom': 4.0, 'new_key': 3.0}}
        delta = _compute_delta(prev, curr)
        self.assertIn('tremor.f_dom', delta)
        self.assertNotIn('tremor.label', delta)
        self.assertNotIn('tremor.new_key', delta)

    def test_delta_handles_empty_metrics(self):
        self.assertEqual(_compute_delta({}, {}), {})
        self.assertEqual(_compute_delta({'a': {'x': 1}}, {}), {})

    def test_trend_improvement(self):
        self.assertEqual(_compute_trend({'S': 80.0}, {'S': 70.0}), 'УЛУЧШЕНИЕ')

    def test_trend_stable(self):
        self.assertEqual(_compute_trend({'S': 80.0}, {'S': 82.0}), 'СТАБИЛЬНО')

    def test_trend_worsening(self):
        self.assertEqual(_compute_trend({'S': 50.0}, {'S': 60.0}), 'УХУДШЕНИЕ')

    def test_trend_empty_syndromes(self):
        self.assertEqual(_compute_trend({}, {}), 'СТАБИЛЬНО')
        self.assertEqual(_compute_trend({'S': 50.0}, {}), 'УЛУЧШЕНИЕ')
        self.assertEqual(_compute_trend({}, {'S': 50.0}), 'УХУДШЕНИЕ')


class JaccardTests(unittest.TestCase):
    def test_identical_sets(self):
        self.assertAlmostEqual(_jaccard_weighted({'A', 'B'}, {'A', 'B'}), 100.0)

    def test_disjoint_sets(self):
        self.assertAlmostEqual(_jaccard_weighted({'A'}, {'B'}), 0.0)

    def test_partial_overlap(self):
        sim = _jaccard_weighted({'A', 'B', 'C'}, {'B', 'C', 'D'})
        self.assertAlmostEqual(sim, 2 / 4 * 100)

    def test_empty_sets(self):
        self.assertAlmostEqual(_jaccard_weighted(set(), set()), 0.0)

    def test_weighted_jaccard(self):
        scores_a = {'A': 80.0, 'B': 60.0}
        scores_b = {'A': 90.0, 'C': 50.0}
        sim = _jaccard_weighted({'A', 'B'}, {'A', 'C'}, scores_a, scores_b)
        expected = min(80, 90) / (max(80, 90) + max(60, 0) + max(0, 50)) * 100
        self.assertAlmostEqual(sim, expected, places=1)


class PatientGraphCoreTests(unittest.TestCase):
    def setUp(self):
        self.graph = TemporalPatientGraph(path=None)

    def test_reference_profiles_seeded_on_init(self):
        for ref in _REFERENCE_PROFILES:
            self.assertIn(ref['id'], self.graph.graph)
            data = self.graph.graph.nodes[ref['id']]
            self.assertTrue(data.get('reference'))
            self.assertEqual(data['node_type'], 'SESSION')
            for bio in ref['biomarkers']:
                self.assertTrue(self.graph.graph.has_edge(ref['id'], bio))

    def test_create_patient_assigns_unique_ids(self):
        pid1 = self.graph.create_patient()
        pid2 = self.graph.create_patient('Тестовый пациент')
        self.assertNotEqual(pid1, pid2)
        self.assertTrue(pid1.startswith('patient:'))
        self.assertEqual(self.graph.graph.nodes[pid2]['label'], 'Тестовый пациент')

    def test_list_patients(self):
        self.graph.create_patient('Пациент A')
        self.graph.create_patient('Пациент B')
        patients = self.graph.list_patients()
        self.assertEqual(len(patients), 2)
        labels = [p['label'] for p in patients]
        self.assertIn('Пациент A', labels)
        self.assertIn('Пациент B', labels)

    def test_commit_session_creates_nodes_and_edges(self):
        pid = self.graph.create_patient()
        snap = snapshot()
        sid = self.graph.commit_session(pid, snap)
        self.assertTrue(sid.startswith('session:'))
        self.assertTrue(self.graph.graph.has_edge(pid, sid))
        data = self.graph.graph.nodes[sid]
        self.assertEqual(data['node_type'], 'SESSION')
        self.assertEqual(data['session_index'], 1)
        self.assertIn('tremor', data['metrics'])
        self.assertTrue(len(data['active_biomarkers']) > 0)

    def test_commit_session_rejects_invalid_patient(self):
        with self.assertRaises(ValueError):
            self.graph.commit_session('nonexistent', snapshot())

    def test_commit_session_rejects_empty_assessments(self):
        pid = self.graph.create_patient()
        with self.assertRaises(ValueError):
            self.graph.commit_session(pid, {})

    def test_multi_session_creates_next_session_edge(self):
        pid = self.graph.create_patient()
        snap1 = snapshot()
        sid1 = self.graph.commit_session(pid, snap1)
        snap2 = snapshot(MILD_RESULTS)
        sid2 = self.graph.commit_session(pid, snap2)
        self.assertTrue(self.graph.graph.has_edge(sid1, sid2))
        edge = self.graph.graph.get_edge_data(sid1, sid2)
        self.assertEqual(edge['type'], 'NEXT_SESSION')
        self.assertIn('delta', edge)
        self.assertIn('trend', edge)
        self.assertIn(edge['trend'], ('УЛУЧШЕНИЕ', 'СТАБИЛЬНО', 'УХУДШЕНИЕ'))

    def test_next_session_trend_improvement_when_pathology_decreases(self):
        pid = self.graph.create_patient()
        self.graph.commit_session(pid, snapshot())
        sid2 = self.graph.commit_session(pid, snapshot(MILD_RESULTS))
        sessions = self.graph.patient_sessions(pid)
        edge = self.graph.graph.get_edge_data(sessions[0]['id'], sid2)
        self.assertEqual(edge['trend'], 'УЛУЧШЕНИЕ')

    def test_session_exhibits_and_indicates_edges(self):
        pid = self.graph.create_patient()
        sid = self.graph.commit_session(pid, snapshot())
        data = self.graph.graph.nodes[sid]
        for bio in data['active_biomarkers']:
            self.assertTrue(self.graph.graph.has_edge(sid, bio))
            self.assertEqual(self.graph.graph.get_edge_data(sid, bio)['type'], 'EXHIBITS')
        for syn_id, score in data['syndrome_scores'].items():
            if score > 0:
                self.assertTrue(self.graph.graph.has_edge(sid, syn_id))
                self.assertEqual(self.graph.graph.get_edge_data(sid, syn_id)['type'], 'INDICATES')


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.graph = TemporalPatientGraph(path=None)
        self.pid = self.graph.create_patient()

    def test_empty_trajectory(self):
        traj = self.graph.get_patient_trajectory(self.pid)
        self.assertEqual(traj['sessions'], [])
        self.assertEqual(traj['overall_trend'], 'НЕТ ДАННЫХ')

    def test_single_session_trajectory(self):
        self.graph.commit_session(self.pid, snapshot())
        traj = self.graph.get_patient_trajectory(self.pid)
        self.assertEqual(len(traj['sessions']), 1)
        self.assertEqual(traj['overall_trend'], 'ПЕРВИЧНАЯ СЕССИЯ')

    def test_multi_session_trajectory(self):
        self.graph.commit_session(self.pid, snapshot())
        self.graph.commit_session(self.pid, snapshot(MILD_RESULTS))
        self.graph.commit_session(self.pid, snapshot(MILD_RESULTS))
        traj = self.graph.get_patient_trajectory(self.pid)
        self.assertEqual(len(traj['sessions']), 3)
        self.assertEqual(len(traj['deltas']), 2)
        self.assertIn(traj['overall_trend'], ('УЛУЧШЕНИЕ', 'СТАБИЛЬНО', 'УХУДШЕНИЕ'))


class ClinicalTwinTests(unittest.TestCase):
    def setUp(self):
        self.graph = TemporalPatientGraph(path=None)

    def test_find_twins_against_reference_profiles(self):
        twins = self.graph.find_clinical_twins(
            ['TREMOR_REST_4_6HZ', 'MOTOR_DECREMENT_HIGH'],
            {'SYNDROME_PARKINSONIAN': 100.0})
        self.assertTrue(len(twins) > 0)
        self.assertTrue(twins[0]['reference'])
        self.assertIn('Паркинсонический', twins[0]['label'])
        self.assertGreater(twins[0]['similarity'], 50.0)

    def test_find_twins_face_paresis_matches_fast_reference(self):
        twins = self.graph.find_clinical_twins(
            ['FACE_LOWER_PARESIS'],
            {'SYNDROME_CENTRAL_STROKE': 95.0})
        matching = [t for t in twins if 'FAST' in t['label']]
        self.assertTrue(len(matching) > 0)

    def test_find_twins_excludes_own_patient(self):
        pid = self.graph.create_patient()
        self.graph.commit_session(pid, snapshot())
        twins = self.graph.find_clinical_twins(
            ['TREMOR_REST_4_6HZ', 'MOTOR_DECREMENT_HIGH'],
            {'SYNDROME_PARKINSONIAN': 100.0},
            exclude_patient_id=pid)
        own_sessions = [t for t in twins if t['patient_id'] == pid]
        self.assertEqual(len(own_sessions), 0)

    def test_find_twins_empty_biomarkers(self):
        twins = self.graph.find_clinical_twins([], {})
        self.assertEqual(len(twins), 0)

    def test_find_twins_from_historical_session(self):
        pid1 = self.graph.create_patient('Пациент А')
        self.graph.commit_session(pid1, snapshot())
        pid2 = self.graph.create_patient('Пациент Б')
        twins = self.graph.find_clinical_twins(
            ['TREMOR_REST_4_6HZ', 'MOTOR_DECREMENT_HIGH', 'VOICE_JITTER_ELEVATED'],
            {'SYNDROME_PARKINSONIAN': 100.0},
            exclude_patient_id=pid2)
        patient_a_matches = [t for t in twins if t['patient_id'] == pid1]
        self.assertTrue(len(patient_a_matches) > 0)


class PersistenceTests(unittest.TestCase):
    def test_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'test_graph.json')
            graph1 = TemporalPatientGraph(path=path)
            pid = graph1.create_patient('Пациент Тест')
            graph1.commit_session(pid, snapshot())
            graph1.save()
            self.assertTrue(os.path.isfile(path))
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.assertIn('nodes', data)
            graph2 = TemporalPatientGraph(path=path)
            patients = graph2.list_patients()
            self.assertEqual(len(patients), 1)
            self.assertEqual(patients[0]['label'], 'Пациент Тест')
            sessions = graph2.patient_sessions(pid)
            self.assertEqual(len(sessions), 1)

    def test_no_path_does_not_crash_save(self):
        graph = TemporalPatientGraph(path=None)
        graph.create_patient()
        graph.save()

    def test_references_preserved_after_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'ref_test.json')
            graph1 = TemporalPatientGraph(path=path)
            graph1.save()
            graph2 = TemporalPatientGraph(path=path)
            for ref in _REFERENCE_PROFILES:
                self.assertIn(ref['id'], graph2.graph)


class HistoryGraphDataTests(unittest.TestCase):
    def test_history_graph_data_structure(self):
        graph = TemporalPatientGraph(path=None)
        pid = graph.create_patient()
        graph.commit_session(pid, snapshot())
        graph.commit_session(pid, snapshot(MILD_RESULTS))
        data = graph.history_graph_data(pid)
        self.assertIn('nodes', data)
        self.assertIn('edges', data)
        types = {n['type'] for n in data['nodes']}
        self.assertIn('PATIENT', types)
        self.assertIn('SESSION', types)
        edge_types = {e['type'] for e in data['edges']}
        self.assertIn('HAS_SESSION', edge_types)
        self.assertIn('NEXT_SESSION', edge_types)
        next_edges = [e for e in data['edges'] if e['type'] == 'NEXT_SESSION']
        self.assertTrue(all(e['trend'] in ('УЛУЧШЕНИЕ', 'СТАБИЛЬНО', 'УХУДШЕНИЕ') for e in next_edges))


class UIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def test_cdss_dashboard_mode_toggle(self):
        from app import MainWindow
        window = MainWindow()
        window.resize(1200, 750)
        window.show()
        self.qt.processEvents()
        cdss = window.dashboard_page.cdss
        self.assertTrue(hasattr(cdss, 'mode_button'))
        self.assertIn('ТЕКУЩИЙ СРЕЗ', cdss.mode_button.text())
        cdss.mode_button.click()
        self.qt.processEvents()
        self.assertIn('ГРАФ ИСТОРИИ', cdss.mode_button.text())
        cdss.mode_button.click()
        self.qt.processEvents()
        self.assertIn('ТЕКУЩИЙ СРЕЗ', cdss.mode_button.text())
        window.close()
        if window.session.worker.isRunning():
            window.session.worker.wait(3000)
        self.qt.processEvents()
        window.deleteLater()
        self.qt.processEvents()

    def test_patient_selector_and_new_patient(self):
        from app import MainWindow
        window = MainWindow()
        window.resize(1200, 750)
        window.show()
        self.qt.processEvents()
        cdss = window.dashboard_page.cdss
        self.assertTrue(hasattr(cdss, 'patient_combo'))
        initial_count = cdss.patient_combo.count()
        self.assertGreaterEqual(initial_count, 1)
        cdss.new_patient_button.click()
        self.qt.processEvents()
        self.assertEqual(cdss.patient_combo.count(), initial_count + 1)
        window.close()
        if window.session.worker.isRunning():
            window.session.worker.wait(3000)
        self.qt.processEvents()
        window.deleteLater()
        self.qt.processEvents()

    def test_save_session_button_creates_graph_session(self):
        from app import MainWindow
        window = MainWindow()
        window.resize(1200, 750)
        window.show()
        session = window.session
        cdss = window.dashboard_page.cdss
        for module, data in RESULTS.items():
            session.publish(module, data)
        deadline = time.perf_counter() + 3
        while session.result is None and time.perf_counter() < deadline:
            QTest.qWait(10)
        self.assertIsNotNone(session.result)
        self.qt.processEvents()
        self.assertTrue(cdss.save_session_button.isEnabled())
        initial_session_count = cdss.patient_graph.session_count()
        cdss.save_session_button.click()
        self.qt.processEvents()
        self.assertEqual(cdss.patient_graph.session_count(), initial_session_count + 1)
        window.close()
        if session.worker.isRunning():
            session.worker.wait(3000)
        self.qt.processEvents()
        window.deleteLater()
        self.qt.processEvents()

    def test_existing_cdss_graph_view_unchanged_in_current_mode(self):
        from app import MainWindow
        window = MainWindow()
        window.resize(1200, 750)
        window.show()
        self.qt.processEvents()
        cdss = window.dashboard_page.cdss
        self.assertEqual(len(cdss.graph_view.nodes), 15)
        self.assertIn('Ожидание', cdss.primary.text())
        window.close()
        if window.session.worker.isRunning():
            window.session.worker.wait(3000)
        self.qt.processEvents()
        window.deleteLater()
        self.qt.processEvents()

    def test_persistent_storage_across_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'restart_test.json')
            g1 = TemporalPatientGraph(path=path)
            pid = g1.add_patient(first_name='Алексей', last_name='Смирнов', password='secret_pwd_123')
            self.assertTrue(g1.has_password(pid))
            self.assertTrue(g1.verify_password(pid, 'secret_pwd_123'))
            self.assertFalse(g1.verify_password(pid, 'wrong_pwd'))
            sid = g1.commit_session(pid, snapshot())
            g1.save()

            g2 = TemporalPatientGraph(path=path)
            patients = g2.list_patients()
            self.assertEqual(len(patients), 1)
            self.assertEqual(patients[0]['id'], pid)
            self.assertEqual(patients[0]['label'], 'Смирнов Алексей')
            self.assertTrue(g2.has_password(pid))
            self.assertTrue(g2.verify_password(pid, 'secret_pwd_123'))
            self.assertFalse(g2.verify_password(pid, 'wrong_pwd'))
            sessions = g2.patient_sessions(pid)
            self.assertEqual(len(sessions), 1)
            summary = g2.get_patient_card_subgraph_summary(pid)
            self.assertEqual(summary['first_name'], 'Алексей')
            self.assertEqual(summary['last_name'], 'Смирнов')
            self.assertTrue(summary['is_password_protected'])
            self.assertEqual(len(summary['sessions']), 1)
            self.assertTrue(len(summary['all_diagnoses']) > 0)

    def test_password_and_card_state_transitions(self):
        from patient_cards_drawer import PatientGraphCard
        graph = TemporalPatientGraph(path=None)
        pid = graph.add_patient(first_name='Елена', last_name='Ковалева', password='card_password_99')
        card = PatientGraphCard(pid, graph)
        self.assertFalse(card.is_unlocked)
        self.assertTrue(hasattr(card, 'unlock_button'))

        unlocked = card.unlock('wrong_password')
        self.assertFalse(unlocked)
        self.assertFalse(card.is_unlocked)

        unlocked = card.unlock('card_password_99')
        self.assertTrue(unlocked)
        self.assertTrue(card.is_unlocked)
        self.assertTrue(hasattr(card, 'mini_graph'))
        self.assertTrue(hasattr(card, 'lock_button'))

        card.lock()
        self.assertFalse(card.is_unlocked)

    def test_patient_drawer_toggle_via_button(self):
        from app import MainWindow
        window = MainWindow()
        window.resize(1200, 750)
        window.show()
        self.qt.processEvents()
        self.assertTrue(hasattr(window, 'patient_drawer_button'))
        self.assertTrue(hasattr(window, 'patient_drawer'))
        drawer = window.patient_drawer
        self.assertFalse(drawer.is_open)

        window.patient_drawer_button.click()
        self.qt.processEvents()
        self.assertTrue(drawer.is_open)
        self.assertTrue(window.patient_drawer_button.isChecked())

        window.patient_drawer_button.click()
        self.qt.processEvents()
        self.assertFalse(drawer.is_open)
        self.assertFalse(window.patient_drawer_button.isChecked())

        window.close()
        if window.session.worker.isRunning():
            window.session.worker.wait(3000)
        self.qt.processEvents()
        window.deleteLater()
        self.qt.processEvents()

    def test_session_reset_on_new_or_switched_patient(self):
        from app import MainWindow
        window = MainWindow()
        window.resize(1200, 750)
        window.show()
        session = window.session
        cdss = window.dashboard_page.cdss

        for module, data in RESULTS.items():
            session.publish(module, data)
        deadline = time.perf_counter() + 3
        while session.result is None and time.perf_counter() < deadline:
            QTest.qWait(10)
        self.assertIsNotNone(session.result)
        self.assertTrue(len(session.assessments) > 0)
        self.assertNotEqual(session.result['status'], 'empty')

        cdss.new_patient_button.click()
        self.qt.processEvents()
        deadline = time.perf_counter() + 3
        while (session.result is None or session.result['status'] != 'empty') and time.perf_counter() < deadline:
            QTest.qWait(10)
        self.assertEqual(len(session.assessments), 0)
        self.assertEqual(session.result['status'], 'empty')
        self.assertIn('Ожидание', cdss.primary.text())

        window.close()
        if session.worker.isRunning():
            session.worker.wait(3000)
        self.qt.processEvents()
        window.deleteLater()
        self.qt.processEvents()


if __name__ == '__main__':
    unittest.main()

