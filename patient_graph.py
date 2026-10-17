# граф связей истории пацыента и сессий

import hashlib
import json
import math
import os
from pathlib import Path
import uuid
from copy import deepcopy
from datetime import datetime

import networkx as nx

from session_diagnostics import BIOMARKERS, HYPOTHESES, build_synthesis


project_root = Path(__file__).resolve().parent
default_storage_path = str(project_root / "patient_history_graph.json")
_DEFAULT_PATH = object()

# алиасы для совместимости
PROJECT_ROOT = project_root
DEFAULT_STORAGE_PATH = default_storage_path


_REFERENCE_PROFILES = [
    {
        'id': 'ref:parkinsonian',
        'label': 'Эталон #1: Паркинсонический профиль',
        'biomarkers': ['TREMOR_REST_4_6HZ', 'MOTOR_DECREMENT_HIGH', 'MOTOR_CADENCE_LOW', 'VOICE_JITTER_ELEVATED'],
        'syndromes': {'SYNDROME_PARKINSONIAN': 100.0, 'SYNDROME_ESSENTIAL_TREMOR': 0.0},
        'metrics': {
            'tremor': {'f_dom': 5.0, 'concentration': 85.0, 'rms': 1.2},
            'motor': {'tap_frequency': 2.0, 'decrement': -35.0, 'rhythm_cv': 28.0},
            'voice': {'f0': 165.0, 'jitter': 1.8, 'shimmer': 5.2, 'hnr': 16.0},
        },
    },
    {
        'id': 'ref:face_fast',
        'label': 'Эталон #2: Лицевой парез FAST+',
        'biomarkers': ['FACE_LOWER_PARESIS', 'VOICE_JITTER_ELEVATED'],
        'syndromes': {'SYNDROME_CENTRAL_STROKE': 95.0, 'SYNDROME_BELLS_PALSY': 0.0},
        'metrics': {
            'facial': {'asi': 22.0, 'smile': 32.0, 'brow': 6.0},
            'voice': {'f0': 175.0, 'jitter': 1.3, 'shimmer': 4.2, 'hnr': 18.0},
        },
    },
    {
        'id': 'ref:essential_tremor',
        'label': 'Эталон #3: Эссенциальный тремор',
        'biomarkers': ['TREMOR_ACTION_6_12HZ'],
        'syndromes': {'SYNDROME_ESSENTIAL_TREMOR': 88.0, 'SYNDROME_PARKINSONIAN': 0.0},
        'metrics': {
            'tremor': {'f_dom': 8.5, 'concentration': 75.0, 'rms': 0.8},
            'motor': {'tap_frequency': 3.5, 'decrement': -5.0, 'rhythm_cv': 12.0},
        },
    },
]


def _session_biomarker_set(graph, session_id):
    result = set()
    for _, target, data in graph.out_edges(session_id, data=True):
        if data.get('type') == 'EXHIBITS':
            result.add(target)
    return result


def _session_syndrome_scores(graph, session_id):
    result = {}
    for _, target, data in graph.out_edges(session_id, data=True):
        if data.get('type') == 'INDICATES':
            result[target] = data.get('score', 0.0)
    return result


def _compute_delta(prev_metrics, curr_metrics):
    deltas = {}
    all_modules = set(list(prev_metrics.keys()) + list(curr_metrics.keys()))
    for module in all_modules:
        pm = prev_metrics.get(module, {})
        cm = curr_metrics.get(module, {})
        for key in set(list(pm.keys()) + list(cm.keys())):
            pv, cv = pm.get(key), cm.get(key)
            if isinstance(pv, (int, float)) and isinstance(cv, (int, float)):
                if math.isfinite(pv) and math.isfinite(cv):
                    deltas[f'{module}.{key}'] = round(cv - pv, 4)
    return deltas


def _compute_trend(prev_syndromes, curr_syndromes):
    prev_max = max(prev_syndromes.values()) if prev_syndromes else 0.0
    curr_max = max(curr_syndromes.values()) if curr_syndromes else 0.0
    diff = curr_max - prev_max
    if diff < -5.0:
        return 'УЛУЧШЕНИЕ'
    elif diff > 5.0:
        return 'УХУДШЕНИЕ'
    return 'СТАБИЛЬНО'


def _jaccard_weighted(set_a, set_b, scores_a=None, scores_b=None):
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    intersection = set_a & set_b
    if not union:
        return 0.0
    if scores_a and scores_b:
        numerator = 0.0
        denominator = 0.0
        for item in union:
            sa = scores_a.get(item, 0.0) if item in set_a else 0.0
            sb = scores_b.get(item, 0.0) if item in set_b else 0.0
            numerator += min(sa, sb)
            denominator += max(sa, sb)
        return (numerator / denominator * 100) if denominator > 0 else 0.0
    return len(intersection) / len(union) * 100


class TemporalPatientGraph:
    def __init__(self, path=_DEFAULT_PATH, in_memory=False):
        if in_memory or path is None:
            self.path = None
        elif path is _DEFAULT_PATH:
            self.path = default_storage_path
        else:
            self.path = str(path)
        self.graph = nx.DiGraph()
        self._seed_references()
        if self.path and os.path.isfile(self.path):
            self._load()

    def _seed_references(self):
        for ref in _REFERENCE_PROFILES:
            rid = ref['id']
            if rid in self.graph:
                continue
            self.graph.add_node(rid, node_type='SESSION', label=ref['label'],
                                reference=True, patient_id='ref',
                                timestamp=datetime(2024, 1, 1).isoformat(),
                                session_index=0, metrics=deepcopy(ref['metrics']),
                                active_biomarkers=list(ref['biomarkers']),
                                syndrome_scores=deepcopy(ref['syndromes']))
            for bio_id in ref['biomarkers']:
                if bio_id not in self.graph:
                    label = BIOMARKERS.get(bio_id, (bio_id,))[0]
                    self.graph.add_node(bio_id, node_type='BIOMARKER', label=label)
                self.graph.add_edge(rid, bio_id, type='EXHIBITS')
            for syn_id, score in ref['syndromes'].items():
                if score > 0:
                    if syn_id not in self.graph:
                        label = HYPOTHESES.get(syn_id, syn_id)
                        self.graph.add_node(syn_id, node_type='SYNDROME', label=label)
                    self.graph.add_edge(rid, syn_id, type='INDICATES', score=score)

    def _load(self):
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            loaded = nx.node_link_graph(data, directed=True)
            for node, attrs in loaded.nodes(data=True):
                if node not in self.graph:
                    self.graph.add_node(node, **attrs)
                else:
                    self.graph.nodes[node].update(attrs)
            for u, v, attrs in loaded.edges(data=True):
                if not self.graph.has_edge(u, v):
                    self.graph.add_edge(u, v, **attrs)
                else:
                    self.graph.edges[u, v].update(attrs)
        except Exception:
            pass

    def save(self):
        if not self.path:
            return
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def add_patient(self, label=None, first_name="", last_name="", password=None):
        existing = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'PATIENT']
        index = len(existing) + 1
        patient_id = f'patient:{uuid.uuid4().hex[:8]}'
        if first_name or last_name:
            name_label = f"{last_name} {first_name}".strip()
            label = name_label if name_label else label
        if label is None:
            label = f'Пациент #{index:02d}'
        password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest() if password else None
        self.graph.add_node(patient_id, node_type='PATIENT', label=label,
                            first_name=first_name, last_name=last_name,
                            password_hash=password_hash,
                            created_at=datetime.now().isoformat(), metadata={},
                            diagnoses_summary=[])
        self.save()
        return patient_id

    create_patient = add_patient

    def has_password(self, patient_id):
        if patient_id not in self.graph:
            return False
        return bool(self.graph.nodes[patient_id].get('password_hash'))

    def verify_password(self, patient_id, raw_password):
        if patient_id not in self.graph:
            return False
        pwd_hash = self.graph.nodes[patient_id].get('password_hash')
        if not pwd_hash:
            return True
        if not raw_password:
            return False
        return hashlib.sha256(raw_password.encode('utf-8')).hexdigest() == pwd_hash

    def set_password(self, patient_id, password):
        if patient_id in self.graph:
            pwd_hash = hashlib.sha256(password.encode('utf-8')).hexdigest() if password else None
            self.graph.nodes[patient_id]['password_hash'] = pwd_hash
            self.save()

    def list_patients(self):
        result = []
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'PATIENT':
                result.append({'id': node, 'label': data.get('label', node),
                               'first_name': data.get('first_name', ''),
                               'last_name': data.get('last_name', ''),
                               'has_password': bool(data.get('password_hash')),
                               'created_at': data.get('created_at', '')})
        result.sort(key=lambda p: p.get('created_at', ''))
        return result

    def patient_sessions(self, patient_id):
        sessions = []
        for _, target, data in self.graph.out_edges(patient_id, data=True):
            if data.get('type') == 'HAS_SESSION':
                attrs = self.graph.nodes[target]
                sessions.append({'id': target, **attrs})
        sessions.sort(key=lambda s: s.get('session_index', 0))
        return sessions

    def _last_session_id(self, patient_id):
        sessions = self.patient_sessions(patient_id)
        return sessions[-1]['id'] if sessions else None

    def commit_session(self, patient_id, assessments, synthesis_result=None):
        if patient_id not in self.graph or self.graph.nodes[patient_id].get('node_type') != 'PATIENT':
            raise ValueError(f'Пациент {patient_id} не найден в графе.')
        if not assessments:
            raise ValueError('Нет данных для сохранения в сессию.')
        sessions = self.patient_sessions(patient_id)
        session_index = len(sessions) + 1
        session_id = f'session:{uuid.uuid4().hex[:8]}'
        timestamp = datetime.now().isoformat()

        metrics = {}
        for module, assessment in assessments.items():
            metrics[module] = deepcopy(assessment.get('metrics', {}))

        if synthesis_result is None:
            synthesis_result = build_synthesis(assessments)

        active_biomarkers = list(synthesis_result.get('active', {}).keys())
        syndrome_scores = {}
        for item in synthesis_result.get('ranking', []):
            syndrome_scores[item['id']] = item['score']

        self.graph.add_node(session_id, node_type='SESSION', patient_id=patient_id,
                            timestamp=timestamp, session_index=session_index,
                            metrics=metrics, active_biomarkers=active_biomarkers,
                            syndrome_scores=syndrome_scores, reference=False)

        self.graph.add_edge(patient_id, session_id, type='HAS_SESSION')

        for bio_id in active_biomarkers:
            if bio_id not in self.graph:
                label = BIOMARKERS.get(bio_id, (bio_id,))[0]
                self.graph.add_node(bio_id, node_type='BIOMARKER', label=label)
            self.graph.add_edge(session_id, bio_id, type='EXHIBITS')
        for syn_id, score in syndrome_scores.items():
            if score > 0:
                if syn_id not in self.graph:
                    label = HYPOTHESES.get(syn_id, syn_id)
                    self.graph.add_node(syn_id, node_type='SYNDROME', label=label)
                self.graph.add_edge(session_id, syn_id, type='INDICATES', score=score)

        prev_id = sessions[-1]['id'] if sessions else None
        if prev_id is not None:
            prev_data = self.graph.nodes[prev_id]
            prev_metrics = prev_data.get('metrics', {})
            prev_syndromes = prev_data.get('syndrome_scores', {})
            delta = _compute_delta(prev_metrics, metrics)
            trend = _compute_trend(prev_syndromes, syndrome_scores)
            dt_sec = 0.0
            try:
                t_prev = datetime.fromisoformat(prev_data.get('timestamp', timestamp))
                t_curr = datetime.fromisoformat(timestamp)
                dt_sec = (t_curr - t_prev).total_seconds()
            except (ValueError, TypeError):
                pass
            self.graph.add_edge(prev_id, session_id, type='NEXT_SESSION',
                                dt_sec=dt_sec, delta=delta, trend=trend,
                                delta_summary=self._format_delta_summary(delta, trend))

        diagnoses = []
        for item in synthesis_result.get('ranking', []):
            if item.get('score', 0) > 0:
                diagnoses.append(f"{item['label']} ({item['score']:.0f}%)")
        if diagnoses:
            self.graph.nodes[patient_id]['diagnoses_summary'] = diagnoses

        self.save()
        return session_id

    record_session = commit_session

    def _format_delta_summary(self, delta, trend):
        parts = []
        for key, value in sorted(delta.items()):
            if abs(value) > 0.01:
                sign = '+' if value > 0 else ''
                parts.append(f'{key}: {sign}{value}')
        summary = trend
        if parts:
            summary += ' (' + ', '.join(parts[:5]) + ')'
        return summary

    def get_patient_trajectory(self, patient_id):
        sessions = self.patient_sessions(patient_id)
        if not sessions:
            return {'sessions': [], 'deltas': [], 'overall_trend': 'НЕТ ДАННЫХ'}
        deltas = []
        for i in range(1, len(sessions)):
            prev_id = sessions[i - 1]['id']
            curr_id = sessions[i]['id']
            edge_data = self.graph.get_edge_data(prev_id, curr_id)
            if edge_data and edge_data.get('type') == 'NEXT_SESSION':
                deltas.append({
                    'from': prev_id, 'to': curr_id,
                    'delta': edge_data.get('delta', {}),
                    'trend': edge_data.get('trend', 'СТАБИЛЬНО'),
                    'dt_sec': edge_data.get('dt_sec', 0),
                    'delta_summary': edge_data.get('delta_summary', ''),
                })
        if not deltas:
            overall = 'ПЕРВИЧНАЯ СЕССИЯ'
        else:
            trends = [d['trend'] for d in deltas]
            improve = trends.count('УЛУЧШЕНИЕ')
            worsen = trends.count('УХУДШЕНИЕ')
            if improve > worsen:
                overall = 'УЛУЧШЕНИЕ'
            elif worsen > improve:
                overall = 'УХУДШЕНИЕ'
            else:
                overall = 'СТАБИЛЬНО'
        return {'sessions': sessions, 'deltas': deltas, 'overall_trend': overall}

    def get_patient_card_subgraph_summary(self, patient_id):
        if patient_id not in self.graph:
            return None
        p_data = self.graph.nodes[patient_id]
        first_name = p_data.get('first_name', '')
        last_name = p_data.get('last_name', '')
        label = p_data.get('label', patient_id)
        created_at = p_data.get('created_at', '')
        is_protected = bool(p_data.get('password_hash'))

        sessions = self.patient_sessions(patient_id)
        enriched_sessions = []
        for i, s in enumerate(sessions):
            sid = s['id']
            trend = 'ПЕРВИЧНАЯ СЕССИЯ' if i == 0 else 'СТАБИЛЬНО'
            delta_summary = ''
            if i > 0:
                prev_id = sessions[i - 1]['id']
                edge = self.graph.get_edge_data(prev_id, sid)
                if edge:
                    trend = edge.get('trend', 'СТАБИЛЬНО')
                    delta_summary = edge.get('delta_summary', '')
            enriched_sessions.append({
                'id': sid,
                'session_index': s.get('session_index', i + 1),
                'timestamp': s.get('timestamp', ''),
                'metrics': s.get('metrics', {}),
                'active_biomarkers': s.get('active_biomarkers', []),
                'syndrome_scores': s.get('syndrome_scores', {}),
                'trend': trend,
                'delta_summary': delta_summary,
            })

        diagnoses_map = {}
        biomarkers_set = set()
        for s in sessions:
            for bio in s.get('active_biomarkers', []):
                biomarkers_set.add(bio)
            for syn_id, score in s.get('syndrome_scores', {}).items():
                if score > 0:
                    if syn_id not in diagnoses_map or score > diagnoses_map[syn_id]['score']:
                        label_syn = HYPOTHESES.get(syn_id, syn_id)
                        diagnoses_map[syn_id] = {'id': syn_id, 'label': label_syn, 'score': score}

        all_diagnoses = sorted(diagnoses_map.values(), key=lambda d: -d['score'])
        all_biomarkers = []
        for bio_id in sorted(biomarkers_set):
            label_bio = BIOMARKERS.get(bio_id, (bio_id,))[0]
            all_biomarkers.append({'id': bio_id, 'label': label_bio})

        graph_topology = self.history_graph_data(patient_id)

        return {
            'patient_id': patient_id,
            'first_name': first_name,
            'last_name': last_name,
            'label': label,
            'created_at': created_at,
            'is_password_protected': is_protected,
            'sessions': enriched_sessions,
            'all_diagnoses': all_diagnoses,
            'all_biomarkers': all_biomarkers,
            'graph_topology': graph_topology,
        }

    def find_clinical_twins(self, biomarkers, syndrome_scores=None, exclude_patient_id=None, top_k=2):
        query_bios = set(biomarkers) if biomarkers else set()
        query_scores = {}
        if syndrome_scores:
            query_scores = {k: v for k, v in syndrome_scores.items() if v > 0}
        query_syns = set(query_scores.keys())

        candidates = []
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') != 'SESSION':
                continue
            if exclude_patient_id and data.get('patient_id') == exclude_patient_id:
                continue
            s_bios = set(data.get('active_biomarkers', []))
            s_syn_scores = data.get('syndrome_scores', {})
            s_syns = set(k for k, v in s_syn_scores.items() if v > 0)
            bio_sim = _jaccard_weighted(query_bios, s_bios)
            syn_sim = _jaccard_weighted(query_syns, s_syns, query_scores, s_syn_scores)
            combined = bio_sim * 0.4 + syn_sim * 0.6 if query_scores else bio_sim
            if combined > 0:
                candidates.append({
                    'session_id': node,
                    'label': data.get('label', node),
                    'patient_id': data.get('patient_id', ''),
                    'reference': data.get('reference', False),
                    'similarity': round(combined, 1),
                    'bio_similarity': round(bio_sim, 1),
                    'syn_similarity': round(syn_sim, 1),
                })
        candidates.sort(key=lambda c: -c['similarity'])
        return candidates[:top_k]

    def session_count(self):
        return sum(1 for _, d in self.graph.nodes(data=True) if d.get('node_type') == 'SESSION' and not d.get('reference'))

    def history_graph_data(self, patient_id):
        sessions = self.patient_sessions(patient_id)
        nodes = []
        edges = []
        patient_data = self.graph.nodes.get(patient_id, {})
        nodes.append({'id': patient_id, 'type': 'PATIENT',
                       'label': patient_data.get('label', patient_id)})
        for session in sessions:
            sid = session['id']
            nodes.append({
                'id': sid, 'type': 'SESSION',
                'label': f'Сессия #{session.get("session_index", "?")}',
                'timestamp': session.get('timestamp', ''),
                'biomarkers': session.get('active_biomarkers', []),
                'syndrome_scores': session.get('syndrome_scores', {}),
            })
            edges.append({'from': patient_id, 'to': sid, 'type': 'HAS_SESSION', 'trend': None})
        for i in range(1, len(sessions)):
            prev_id = sessions[i - 1]['id']
            curr_id = sessions[i]['id']
            edge_data = self.graph.get_edge_data(prev_id, curr_id)
            trend = edge_data.get('trend', 'СТАБИЛЬНО') if edge_data else 'СТАБИЛЬНО'
            edges.append({'from': prev_id, 'to': curr_id, 'type': 'NEXT_SESSION', 'trend': trend})
        bio_set = set()
        syn_set = set()
        for session in sessions:
            for bio in session.get('active_biomarkers', []):
                if bio not in bio_set:
                    bio_set.add(bio)
                    label = BIOMARKERS.get(bio, (bio,))[0]
                    nodes.append({'id': bio, 'type': 'BIOMARKER', 'label': label})
                edges.append({'from': session['id'], 'to': bio, 'type': 'EXHIBITS', 'trend': None})
            for syn_id, score in session.get('syndrome_scores', {}).items():
                if score > 0 and syn_id not in syn_set:
                    syn_set.add(syn_id)
                    label = HYPOTHESES.get(syn_id, syn_id)
                    nodes.append({'id': syn_id, 'type': 'SYNDROME', 'label': label})
                if score > 0:
                    edges.append({'from': session['id'], 'to': syn_id, 'type': 'INDICATES', 'trend': None})
        return {'nodes': nodes, 'edges': edges}
