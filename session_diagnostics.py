"""Small, immutable session snapshots and off-GUI research evidence inference.

Weights are user-supplied ontology heuristics, not calibrated likelihoods.
The output is an evidence index and cannot establish a clinical diagnosis.
"""

from copy import deepcopy
from datetime import datetime
import math
import threading

import networkx as nx
from PyQt6.QtCore import QObject, QThread, pyqtSignal


MODULE_NAMES = {'tremor': 'ТРЕМОР', 'facial': 'НЕЙРОМИМИКА', 'voice': 'ГОЛОС',
                'motor': 'МОТОРИКА', 'rppg': 'ПУЛЬС (rPPG)'}
BIOMARKERS = {
    'TREMOR_REST_4_6HZ': ('Пик движения 4,0–6,5 Гц', 'tremor'),
    'TREMOR_ACTION_6_12HZ': ('Пик движения 6,5–12 Гц', 'tremor'),
    'MOTOR_DECREMENT_HIGH': ('Декремент амплитуды <−20%', 'motor'),
    'MOTOR_CADENCE_LOW': ('Темп таппинга <2,5 Гц', 'motor'),
    'VOICE_JITTER_ELEVATED': ('Повышены джиттер / шиммер', 'voice'),
    'FACE_LOWER_PARESIS': ('Дисбаланс улыбки, брови <12%', 'facial'),
    'FACE_COMPLETE_PARESIS': ('Дисбаланс улыбки и бровей', 'facial'),
    'TACHYCARDIA_RESTING': ('Камерная ЧСС >95 уд/мин', 'rppg'),
    'BRADYCARDIA_RESTING': ('Камерная ЧСС <60 уд/мин', 'rppg'),
}
HYPOTHESES = {
    'SYNDROME_PARKINSONIAN': 'Экстрапирамидный / паркинсонический паттерн',
    'SYNDROME_ESSENTIAL_TREMOR': 'Постуральный тремор: дифференциальная оценка',
    'SYNDROME_CENTRAL_STROKE': 'Центральная причина асимметрии: проверка',
    'SYNDROME_BELLS_PALSY': 'Периферическая причина асимметрии: проверка',
    'SYNDROME_AUTONOMIC_DYSREG': 'Изменение ЧСС: нагрузка / регуляция',
    'SYNDROME_PHYSIOLOGICAL': 'Без сработавших порогов — не исключение патологии',
}
RELATIONS = (
    ('TREMOR_REST_4_6HZ', 'SYNDROME_PARKINSONIAN', 'SUGGESTS', .85),
    ('MOTOR_DECREMENT_HIGH', 'SYNDROME_PARKINSONIAN', 'SUGGESTS', .90),
    ('VOICE_JITTER_ELEVATED', 'SYNDROME_PARKINSONIAN', 'CORRELATES', .65),
    ('TREMOR_ACTION_6_12HZ', 'SYNDROME_ESSENTIAL_TREMOR', 'SUGGESTS', .88),
    ('MOTOR_DECREMENT_HIGH', 'SYNDROME_ESSENTIAL_TREMOR', 'EXCLUDES', -.75),
    ('FACE_LOWER_PARESIS', 'SYNDROME_CENTRAL_STROKE', 'STRONGLY_SUGGESTS', .95),
    ('VOICE_JITTER_ELEVATED', 'SYNDROME_CENTRAL_STROKE', 'CORRELATES', .60),
    ('FACE_COMPLETE_PARESIS', 'SYNDROME_BELLS_PALSY', 'STRONGLY_SUGGESTS', .92),
    ('TACHYCARDIA_RESTING', 'SYNDROME_AUTONOMIC_DYSREG', 'SUGGESTS', .70),
)
SCHEMAS = {
    'tremor': {'f_dom': 'dominant_freq', 'concentration': 'concentration', 'rms': 'rms_px'},
    'facial': {'asi': 'asymmetry_index', 'smile': 'smile_imbalance', 'brow': 'brow_imbalance'},
    'voice': {'f0': 'f0', 'jitter': 'jitter_local', 'shimmer': 'shimmer_local', 'hnr': 'hnr'},
    'motor': {'tap_frequency': 'tap_frequency', 'decrement': 'amplitude_decrement', 'rhythm_cv': 'rhythm_cv'},
    'rppg': {'bpm': 'bpm', 'concentration': 'concentration'},
}
UNITS = {
    'tremor': {'f_dom': 'Гц', 'concentration': '%', 'rms': 'пикс.'},
    'facial': {'asi': '%', 'smile': '%', 'brow': '%'},
    'voice': {'f0': 'Гц', 'jitter': '%', 'shimmer': '%', 'hnr': 'дБ'},
    'motor': {'tap_frequency': 'Гц', 'decrement': '%', 'rhythm_cv': '%'},
    'rppg': {'bpm': 'уд/мин', 'concentration': '%', 'roi_spread': 'уд/мин', 'coverage': 'доля'},
}


def standardize(module, results):
    metrics = {}
    for name, source in SCHEMAS[module].items():
        value = results[source]
        if value is None and module == 'tremor' and name == 'f_dom':
            metrics[name] = None
        else:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError('Некорректное значение биомаркера: ' + source)
            if name not in ('decrement', 'hnr') and number < 0:
                raise ValueError('Отрицательное значение биомаркера: ' + source)
            if name in ('concentration', 'asi', 'smile', 'brow') and number > 100:
                raise ValueError('Процент вне диапазона 0–100: ' + source)
            metrics[name] = number
    if module == 'rppg':
        rois = [float(v) for v in results.get('roi_bpm', [])]
        used = results.get('used_rois', [])
        selected = [rois[i] for i in used if isinstance(i, int) and 0 <= i < len(rois)]
        spread = max(selected) - min(selected) if len(selected) >= 2 else None
        coverage = float(results.get('coverage', 0))
        stable = (spread is not None and math.isfinite(spread) and spread <= 5
                  and coverage >= .85 and metrics['concentration'] >= 60
                  and not results.get('warnings'))
        metrics.update(baseline_stability='stable' if stable else 'limited',
                       roi_spread=spread, coverage=coverage)
    return {
        'module': module, 'metrics': metrics,
        'units': UNITS[module].copy(),
        'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'source': str(results.get('source', 'camera' if module != 'voice' else 'microphone')),
        'source_file': str(results.get('source_file', results.get('file_name', ''))),
        'warnings': [str(w) for w in results.get('warnings', [])],
    }


def ontology():
    graph = nx.DiGraph()
    for key, (label, module) in BIOMARKERS.items():
        graph.add_node(key, category='biomarker', bipartite=0, label=label, module=module, active=False)
    for key, label in HYPOTHESES.items():
        graph.add_node(key, category='hypothesis', bipartite=1, label=label, score=0.)
    for source, target, relation, weight in RELATIONS:
        graph.add_edge(source, target, type=relation, weight=weight, active=False)
    return graph


def build_synthesis(snapshot, revision=0):
    baseline = ontology()
    graph = nx.DiGraph()
    active = {}
    limited = []
    for module, assessment in snapshot.items():
        m = assessment['metrics']
        keys = []
        if module == 'tremor':
            f = m['f_dom']
            if f is not None and m['concentration'] >= 50 and m['rms'] >= .2:
                if 4 <= f < 6.5:
                    keys = ['TREMOR_REST_4_6HZ']
                elif 6.5 <= f <= 12:
                    keys = ['TREMOR_ACTION_6_12HZ']
            else:
                limited.append('ТРЕМОР: нет выраженного спектрального пика; это не отрицательный диагноз.')
        elif module == 'facial':
            if m['smile'] > 18:
                keys = ['FACE_LOWER_PARESIS' if m['brow'] < 12 else 'FACE_COMPLETE_PARESIS']
        elif module == 'voice':
            if m['jitter'] > 1.04 or m['shimmer'] > 3.81:
                keys = ['VOICE_JITTER_ELEVATED']
        elif module == 'motor':
            if m['decrement'] < -20:
                keys.append('MOTOR_DECREMENT_HIGH')
            if m['tap_frequency'] < 2.5:
                keys.append('MOTOR_CADENCE_LOW')
        elif module == 'rppg':
            if m['baseline_stability'] != 'stable':
                limited.append('ПУЛЬС: качество/стабильность ограничены; ЧСС не участвует в сопоставлении.')
            elif m['bpm'] > 95:
                keys = ['TACHYCARDIA_RESTING']
            elif m['bpm'] < 60:
                keys = ['BRADYCARDIA_RESTING']
        for key in keys:
            attrs = dict(baseline.nodes[key], active=True, metrics=deepcopy(m),
                         recorded_at=assessment['recorded_at'], source=assessment['source'])
            graph.add_node(key, **attrs)
            active[key] = attrs
    for source, target, relation, weight in RELATIONS:
        if source in active:
            graph.add_node(target, **baseline.nodes[target])
            graph.add_edge(source, target, type=relation, weight=weight, active=True)
    ranking = []
    for target in HYPOTHESES:
        if target not in graph:
            continue
        edges = list(graph.in_edges(target, data=True))
        supporting = [source for source, _, attrs in edges if attrs['weight'] > 0]
        opposing = [source for source, _, attrs in edges if attrs['weight'] < 0]
        denominator = sum(weight for _, hypothesis, _, weight in RELATIONS if hypothesis == target and weight > 0)
        score = 100 * max(0., min(1., sum(attrs['weight'] for _, _, attrs in edges) / denominator))
        graph.nodes[target]['score'] = score
        if supporting:
            ranking.append({'id': target, 'label': HYPOTHESES[target], 'score': score,
                            'supporting': supporting, 'opposing': opposing,
                            'modules': sorted({BIOMARKERS[k][1] for k in supporting})})
    ranking.sort(key=lambda item: (-item['score'], item['id']))
    missing = [MODULE_NAMES[k] for k in MODULE_NAMES if k not in snapshot]
    status = ('empty' if not snapshot else 'flagged' if active else 'limited' if limited
              else 'no_flags')
    # No edge can prove "no pathology"; never manufacture a confidence for it.
    if status == 'no_flags':
        graph.add_node('SYNDROME_PHYSIOLOGICAL', **baseline.nodes['SYNDROME_PHYSIOLOGICAL'])
    action = 'Выполните доступные пробы в сопоставимых условиях и сохраните результаты для обсуждения с врачом.'
    if any(key.startswith('FACE_') for key in active):
        action = ('При внезапной асимметрии лица, слабости руки или нарушении речи немедленно обратитесь '
                  'за экстренной помощью. Камерный тест не оценивает начало симптомов и не является FAST+. '
                  'При стойкой асимметрии покажите запись неврологу.')
    elif any(key.startswith('MOTOR_') or key.startswith('TREMOR_') for key in active):
        action = 'При стойком дрожании или замедлении движений покажите неврологу записи обеих рук и сводный отчёт.'
    elif 'VOICE_JITTER_ELEVATED' in active:
        action = 'Повторите фонацию в тихой комнате; стойкие изменения голоса обсудите с врачом или логопедом.'
    elif any(key.endswith('_RESTING') for key in active):
        action = 'Проверьте пульс прямым измерением в покое; при стойких отклонениях обсудите результат с врачом.'
    return {'revision': revision, 'graph': graph, 'ontology': baseline, 'ranking': ranking,
            'active': active, 'assessments': deepcopy(snapshot), 'missing': missing,
            'limited': limited, 'status': status, 'action': action}


class InferenceWorker(QThread):
    ready = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.condition = threading.Condition()
        self.pending = None
        self.stopping = False

    def submit(self, revision, snapshot):
        with self.condition:
            self.pending = (revision, snapshot)
            self.condition.notify()

    def shutdown(self):
        with self.condition:
            self.stopping = True
            self.pending = None
            self.condition.notify()

    def run(self):
        while True:
            with self.condition:
                while self.pending is None and not self.stopping:
                    self.condition.wait()
                if self.stopping:
                    return
                revision, snapshot = self.pending
                self.pending = None
            try:
                self.ready.emit(revision, build_synthesis(snapshot, revision))
            except Exception as exc:
                self.failed.emit(revision, str(exc))


class SessionDiagnostics(QObject):
    """One window-owned state store; newest module result replaces its predecessor."""
    updated = pyqtSignal(object)
    pending = pyqtSignal(int)
    failed = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.assessments = {}
        self.revision = 0
        self.result = None
        self.closing = False
        self.worker = InferenceWorker(self)
        self.worker.ready.connect(self._accept)
        self.worker.failed.connect(self._failure)
        self.worker.finished.connect(self.stopped)

    def publish(self, module, results):
        if self.closing:
            return
        try:
            assessment = standardize(module, results)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self.clear(module)
            self.failed.emit('Результат не добавлен в сессию: ' + str(exc))
            return
        self.assessments[module] = assessment
        self._schedule()

    def clear(self, module):
        if self.closing or module not in self.assessments:
            return
        del self.assessments[module]
        self._schedule()

    def reset(self):
        if self.closing:
            return
        self.assessments.clear()
        self._schedule()

    def _schedule(self):
        self.revision += 1
        self.result = None
        self.pending.emit(self.revision)
        self.worker.submit(self.revision, deepcopy(self.assessments))
        if not self.worker.isRunning():
            self.worker.start()

    def _accept(self, revision, result):
        if not self.closing and revision == self.revision:
            self.result = result
            self.updated.emit(result)

    def _failure(self, revision, message):
        if revision == self.revision and not self.closing:
            self.failed.emit(message)

    def shutdown(self):
        self.closing = True
        self.worker.shutdown()
        return not self.worker.isRunning()
