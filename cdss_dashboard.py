# панель поддержки принятия решений и граф онтологии

from html import escape
import math
from pathlib import Path

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QTextDocument
from PyQt6.QtPrintSupport import QPrinter
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QFrame, QGraphicsRectItem, QGraphicsScene,
                            QGraphicsSimpleTextItem, QGraphicsView, QHBoxLayout, QLabel,
                            QMessageBox, QProgressBar, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from session_diagnostics import BIOMARKERS, HYPOTHESES, MODULE_NAMES, ontology
from patient_graph import TemporalPatientGraph


cdss_disclaimer = (
    'ДИСКЛЕЙМЕР: Мультимодальный графовый анализ является исследовательской системой поддержки '
    'принятия решений (CDSS) и не заменяет комплексную очную диагностику врача-невролога.'
)
score_note = 'Индекс поддержки правил, не вероятность диагноза. Веса не прошли клиническую калибровку.'

# совместимость
CDSS_DISCLAIMER = cdss_disclaimer
SCORE_NOTE = score_note


def fmt(value):
    if value is None:
        return 'не выделено'
    return f'{value:.2f}'.replace('.', ',') if isinstance(value, (int, float)) else str(value)


def summary_text(result, patient_label=None, trajectory=None, twins=None):
    lines = ['TechnoMedNeuro / СВОДНЫЙ МУЛЬТИМОДАЛЬНЫЙ ОТЧЁТ', score_note,
             'Состояние: ' + result['status'], '']
    if patient_label:
        lines += [f'Пациент: {patient_label}', '']
    if trajectory and trajectory.get('overall_trend'):
        lines += [f'ДИНАМИКА ПАЦИЕНТА: {trajectory["overall_trend"]}']
        for d in trajectory.get('deltas', []):
            if d.get('delta_summary'):
                lines += [f'  • {d["delta_summary"]}']
        lines += ['']
    if twins:
        lines += ['КЛИНИЧЕСКИЕ ДВОЙНИКИ (ГРАФОВОЕ СХОДСТВО)']
        for t in twins:
            lines += [f'  • {t["label"]}: сходство {t["similarity"]:.1f}% (био {t["bio_similarity"]}%, синдромы {t["syn_similarity"]}%)']
        lines += ['']
    for module, assessment in result['assessments'].items():
        lines += [MODULE_NAMES[module] + ' / ' + assessment['recorded_at'],
                  'Источник: ' + assessment['source'] + (' / ' + assessment['source_file'] if assessment['source_file'] else '')]
        lines += [f'{key}: {fmt(value)} {assessment.get("units", {}).get(key, "")}'
                  for key, value in assessment['metrics'].items()]
        lines += assessment['warnings'] + ['']
    lines += ['ГИПОТЕЗЫ ДЛЯ ПРОВЕРКИ']
    for item in result['ranking']:
        lines += [f'{item["label"]}: индекс поддержки {item["score"]:.0f}%',
                  'Поддержка: ' + ', '.join(item['supporting']),
                  'Противоречия: ' + (', '.join(item['opposing']) or 'не зарегистрированы')]
    if not result['ranking']:
        lines.append('Нет ранжированных гипотез; отсутствие сработавших правил не исключает заболевание.')
    lines += ['', 'ПРАВИЛА И ВКЛАДЫ']
    for source, target, attrs in result['graph'].edges(data=True):
        lines.append(f'{source} → {target}: {attrs["type"]}, вес {attrs["weight"]:+.2f}')
    lines += ['', 'Формула: 100 × clip(сумма активных signed-весов / сумма всех положительных весов гипотезы, 0, 1).',
              'Правило EXCLUDES — отрицательный вклад модели; оно не исключает диагноз клинически.',
              'Дистанционные пробы не подтверждают позу покоя, время появления симптомов или отрицательный FAST.',
              'Не выполнены: ' + (', '.join(result['missing']) or 'все пять модулей выполнены')]
    lines += result['limited'] + ['', cdss_disclaimer, '', result['action']]
    return '\n'.join(lines)


class KnowledgeGraphView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setStyleSheet('background: #FFFFFF; border: none; border-radius: 0px;')
        self.setMinimumSize(0, 0)
        self.setAccessibleName('Интерактивный граф: масштаб колесом, перемещение мышью, подсказки узлов и связей')
        self.nodes = {}
        self.edges = []
        self.set_result(None)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        scale = self.transform().m11() * factor
        if .15 <= scale <= 3:
            self.scale(factor, factor)
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def mouseDoubleClickEvent(self, event):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        event.accept()

    def _node(self, key, label, rect, active=False, hypothesis=False, tooltip=''):
        alert = key.startswith('FACE_')
        border = '#DC2626' if active and alert else '#2563EB' if active else '#E2E8F0'
        if hypothesis and active:
            border = '#0F172A'
        item = QGraphicsRectItem(rect)
        item.setPen(QPen(QColor(border), 1.5 if active else 1))
        item.setBrush(QColor('#FFFFFF' if hypothesis else '#F8FAFC'))
        item.setToolTip(tooltip or key)
        item.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable)
        self.scene().addItem(item)
        text = QGraphicsSimpleTextItem(label, item)
        text.setFont(QFont('Consolas', 11))
        text.setBrush(QColor('#0F172A' if active else '#94A3B8'))
        text.setPos(rect.x() + 8, rect.y() + 4)
        text.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.nodes[key] = rect

    def _history_node(self, node_id, label, rect, ntype, tooltip=''):
        border = '#0F172A' if ntype in ('PATIENT', 'SYNDROME') else '#2563EB' if ntype == 'SESSION' else '#CBD5E1'
        bg = '#FFFFFF' if ntype in ('SESSION', 'SYNDROME') else '#F8FAFC'
        item = QGraphicsRectItem(rect)
        item.setPen(QPen(QColor(border), 1.5 if ntype in ('PATIENT', 'SESSION') else 1.0))
        item.setBrush(QColor(bg))
        item.setToolTip(tooltip or label)
        item.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable)
        self.scene().addItem(item)
        display = label if len(label) <= 28 else label[:26] + '…'
        text = QGraphicsSimpleTextItem(display, item)
        text.setFont(QFont('Consolas', 8 if ntype in ('BIOMARKER', 'SYNDROME') else 9))
        text.setBrush(QColor('#0F172A'))
        text.setPos(rect.x() + 6, rect.y() + 4)
        text.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.nodes[node_id] = rect

    def set_result(self, result):
        self.scene().clear()
        self.nodes = {}
        self.edges = []
        graph = result['graph'] if result else ontology()
        active = result['active'] if result else {}
        for i, (key, (label, module)) in enumerate(BIOMARKERS.items()):
            attrs = active.get(key, {})
            details = '\n'.join(f'{name}: {fmt(value)}' for name, value in attrs.get('metrics', {}).items())
            self._node(key, label, QRectF(10, 25 + i * 31, 320, 27), key in active,
                       tooltip=key + '\n' + MODULE_NAMES[module] + '\n' + details)
        for i, (key, label) in enumerate(HYPOTHESES.items()):
            reached = result is not None and key in graph and graph.in_degree(key) > 0
            score = graph.nodes[key].get('score', 0) if key in graph else 0
            words = label.split()
            midpoint = max(1, len(words)//2)
            title = ' '.join(words[:midpoint]) + '\n' + ' '.join(words[midpoint:])
            badge = f'  {score:.0f}%' if reached else '  —'
            self._node(key, title + badge, QRectF(640, 25 + i * 46, 440, 42), reached, True,
                       key + '\n' + score_note)
        for source, target, attrs in graph.edges(data=True):
            a, b = self.nodes[source], self.nodes[target]
            enabled = result is not None and attrs.get('active')
            negative = attrs['weight'] < 0
            color = '#DC2626' if enabled and negative else '#2563EB' if enabled else '#E2E8F0'
            pen = QPen(QColor(color), 1.4 if enabled else .8)
            if negative:
                pen.setStyle(Qt.PenStyle.DashLine)
            line = self.scene().addLine(a.right(), a.center().y(), b.left(), b.center().y(), pen)
            tooltip = f'{source} → {target}\n{attrs["type"]} {attrs["weight"]:+.2f}\n' + SCORE_NOTE
            line.setToolTip(tooltip)
            angle = math.atan2(b.center().y() - a.center().y(), b.left() - a.right())
            for sign in (-1, 1):
                x = b.left() - 8 * math.cos(angle + sign * .45)
                y = b.center().y() - 8 * math.sin(angle + sign * .45)
                self.scene().addLine(b.left(), b.center().y(), x, y, pen)
            if enabled:
                label = self.scene().addSimpleText(f'{attrs["type"]} {attrs["weight"]:+.2f}', QFont('Consolas', 7))
                label.setBrush(QColor(color))
                label.setPos(350, a.center().y() - 12)
                label.setToolTip(tooltip)
            self.edges.append({'from': source, 'to': target, 'type': attrs.get('type'), 'weight': attrs.get('weight')})
        caption = 'Ожидание завершения диагностических тестов' if not result or result['status'] == 'empty' else 'Колесо: масштаб / перетаскивание: обзор / двойной щелчок: весь граф'
        text = self.scene().addSimpleText(caption, QFont('Segoe UI', 10))
        text.setBrush(QColor('#64748B'))
        text.setPos(10, 0)
        self.scene().setSceneRect(0, 0, 1090, 310)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_history_data(self, history_data):
        self.scene().clear()
        self.nodes = {}
        self.edges = []
        if not history_data or not history_data.get('nodes'):
            caption = self.scene().addSimpleText('Нет данных графа истории', QFont('Segoe UI', 10))
            caption.setBrush(QColor('#64748B'))
            caption.setPos(10, 0)
            self.scene().setSceneRect(0, 0, 1090, 310)
            return

        nodes = history_data['nodes']
        patients = [n for n in nodes if n.get('type') == 'PATIENT']
        sessions = [n for n in nodes if n.get('type') == 'SESSION']
        biomarkers = [n for n in nodes if n.get('type') == 'BIOMARKER']
        syndromes = [n for n in nodes if n.get('type') == 'SYNDROME']

        y_min, y_max = 35, 275
        h_range = y_max - y_min

        for i, p in enumerate(patients):
            y = 155 if len(patients) == 1 else y_min + h_range * i / max(1, len(patients) - 1)
            rect = QRectF(40, y - 16, 160, 32)
            self._history_node(p['id'], p.get('label', p['id']), rect, 'PATIENT', f"Пациент: {p.get('label')}\nID: {p['id']}")

        for i, s in enumerate(sessions):
            y = 155 if len(sessions) == 1 else y_min + h_range * i / max(1, len(sessions) - 1)
            rect = QRectF(270, y - 16, 150, 32)
            ts = s.get('timestamp', '')[:19]
            bios = ', '.join(s.get('biomarkers', [])) or 'нет'
            syns = ', '.join(f'{k}: {v:.0f}%' for k, v in s.get('syndrome_scores', {}).items() if v > 0) or 'нет'
            tip = f"{s.get('label')}\nВремя: {ts}\nБиомаркеры: {bios}\nСиндромы: {syns}"
            self._history_node(s['id'], s.get('label', s['id']), rect, 'SESSION', tip)

        for j, b in enumerate(biomarkers):
            y = 155 if len(biomarkers) == 1 else y_min + h_range * j / max(1, len(biomarkers) - 1)
            rect = QRectF(500, y - 13, 220, 26)
            label = b.get('label', b['id'])
            self._history_node(b['id'], label, rect, 'BIOMARKER', f"Биомаркер: {label}\nID: {b['id']}")

        for k, syn in enumerate(syndromes):
            y = 155 if len(syndromes) == 1 else y_min + h_range * k / max(1, len(syndromes) - 1)
            rect = QRectF(800, y - 16, 260, 32)
            label = syn.get('label', syn['id'])
            self._history_node(syn['id'], label, rect, 'SYNDROME', f"Синдром: {label}\nID: {syn['id']}")

        for edge in history_data.get('edges', []):
            u = edge.get('from')
            v = edge.get('to')
            etype = edge.get('type')
            trend = edge.get('trend')
            if u in self.nodes and v in self.nodes:
                ra = self.nodes[u]
                rb = self.nodes[v]
                if etype == 'NEXT_SESSION':
                    if trend == 'УЛУЧШЕНИЕ':
                        color = '#10B981'
                    elif trend == 'УХУДШЕНИЕ':
                        color = '#DC2626'
                    else:
                        color = '#64748B'
                    pen = QPen(QColor(color), 2.0)
                    line = self.scene().addLine(ra.center().x(), ra.bottom(), rb.center().x(), rb.top(), pen)
                    line.setToolTip(f"Динамика: {trend}")
                    self.scene().addLine(rb.center().x() - 4, rb.top() - 6, rb.center().x(), rb.top(), pen)
                    self.scene().addLine(rb.center().x() + 4, rb.top() - 6, rb.center().x(), rb.top(), pen)
                    if trend:
                        trend_text = self.scene().addSimpleText(trend, QFont('Consolas', 7))
                        trend_text.setBrush(QColor(color))
                        trend_text.setPos(ra.center().x() + 6, (ra.bottom() + rb.top()) / 2 - 6)
                else:
                    color = '#2563EB' if etype in ('HAS_SESSION', 'EXHIBITS') else '#94A3B8'
                    pen = QPen(QColor(color), 1.2 if etype == 'HAS_SESSION' else 0.8)
                    line = self.scene().addLine(ra.right(), ra.center().y(), rb.left(), rb.center().y(), pen)
                    line.setToolTip(f"{etype}: {u} → {v}")
                    angle = math.atan2(rb.center().y() - ra.center().y(), rb.left() - ra.right())
                    for sign in (-1, 1):
                        x = rb.left() - 7 * math.cos(angle + sign * 0.45)
                        y = rb.center().y() - 7 * math.sin(angle + sign * 0.45)
                        self.scene().addLine(rb.left(), rb.center().y(), x, y, pen)
                self.edges.append(edge)

        caption_text = 'ГРАФ ИСТОРИИ ПАЦИЕНТА: динамика сессий, биомаркеры и синдромы'
        caption = self.scene().addSimpleText(caption_text, QFont('Segoe UI', 10))
        caption.setBrush(QColor('#64748B'))
        caption.setPos(10, 0)
        self.scene().setSceneRect(0, 0, 1090, 310)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class CDSSDashboard(QWidget):
    def __init__(self, session, parent=None, patient_graph=None):
        super().__init__(parent)
        self.session = session
        self.result = None
        self.graph_mode = 'CURRENT'
        self.patient_graph = patient_graph if patient_graph is not None else TemporalPatientGraph()
        self._active_patient_id = None
        self.unlocked_patients = set()
        self.setFixedHeight(260)
        self.setStyleSheet('''
            QFrame#cdssPanel { background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 0px; }
            QLabel { color: #475569; background: transparent; border: none; }
            QScrollArea { background: #FFFFFF; border: none; }
            QPushButton { background: #FFFFFF; color: #1D4ED8; border: 1px solid #CBD5E1; padding: 2px 6px; border-radius: 0px; }
            QPushButton:hover { border-color: #2563EB; }
            QPushButton:disabled { color: #94A3B8; }
            QComboBox { background: #FFFFFF; color: #0F172A; border: 1px solid #CBD5E1; padding: 1px 6px; border-radius: 0px; }
            QComboBox:hover { border-color: #2563EB; }
            QComboBox QAbstractItemView { background: #FFFFFF; color: #0F172A; border: 1px solid #CBD5E1; selection-background-color: #2563EB; selection-color: #FFFFFF; border-radius: 0px; }
            QProgressBar { background: #F8FAFC; color: #0F172A; border: 1px solid #E2E8F0; border-radius: 0px; }
            QProgressBar::chunk { background: #2563EB; border-radius: 0px; }
        ''')
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(3)

        ctrl_bar = QHBoxLayout()
        ctrl_bar.setContentsMargins(8, 0, 8, 0)
        ctrl_bar.setSpacing(6)

        patient_lbl = QLabel('ПАЦИЕНТ:')
        patient_lbl.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        patient_lbl.setStyleSheet('color: #0F172A;')
        ctrl_bar.addWidget(patient_lbl)

        self.patient_combo = QComboBox()
        self.patient_combo.setFont(QFont('Segoe UI', 8))
        self.patient_combo.setFixedHeight(22)
        self.patient_combo.currentIndexChanged.connect(self._on_patient_selected)
        ctrl_bar.addWidget(self.patient_combo)

        self.new_patient_button = QPushButton('+ НОВЫЙ ПАЦИЕНТ')
        self.new_patient_button.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        self.new_patient_button.setFixedHeight(22)
        self.new_patient_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.new_patient_button.clicked.connect(self._on_new_patient_clicked)
        ctrl_bar.addWidget(self.new_patient_button)

        self.save_session_button = QPushButton('СОХРАНИТЬ СЕССИЮ В ГРАФ')
        self.save_session_button.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        self.save_session_button.setFixedHeight(22)
        self.save_session_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_session_button.setEnabled(False)
        self.save_session_button.clicked.connect(self._on_save_session_clicked)
        ctrl_bar.addWidget(self.save_session_button)

        ctrl_bar.addStretch(1)

        mode_lbl = QLabel('РЕЖИМ:')
        mode_lbl.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        mode_lbl.setStyleSheet('color: #0F172A;')
        ctrl_bar.addWidget(mode_lbl)

        self.mode_button = QPushButton('ТЕКУЩИЙ СРЕЗ')
        self.mode_button.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        self.mode_button.setFixedHeight(22)
        self.mode_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mode_button.clicked.connect(self._toggle_mode)
        ctrl_bar.addWidget(self.mode_button)

        outer.addLayout(ctrl_bar)

        split = QHBoxLayout()
        split.setSpacing(10)
        left, ll = self._panel('ИНТЕРАКТИВНЫЙ ГРАФ БИОМАРКЕРОВ И СИНДРОМОВ')
        self.graph_view = KnowledgeGraphView()
        ll.addWidget(self.graph_view, 1)
        split.addWidget(left, 3)
        right, rl = self._panel('МУЛЬТИМОДАЛЬНЫЙ КЛИНИЧЕСКИЙ СИНТЕЗ')
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        body.setStyleSheet('background: #FFFFFF;')
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 3, 0)
        bl.setSpacing(3)
        self.primary = QLabel('Ожидание завершения диагностических тестов')
        self.primary.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        self.primary.setStyleSheet('color: #0F172A;')
        self.score = QProgressBar()
        self.score.setRange(0, 100)
        self.score.setValue(0)
        self.score.setFixedHeight(16)
        self.score.setFormat('Индекс поддержки: %p%')
        self.score.setToolTip(SCORE_NOTE)
        self.note = QLabel(SCORE_NOTE)
        self.differential = QLabel()
        self.evidence = QLabel()
        self.evidence.setFont(QFont('Consolas', 8))
        self.trajectory_label = QLabel('Динамика пациента: первичная сессия')
        self.trajectory_label.setFont(QFont('Segoe UI', 8))
        self.trajectory_label.setStyleSheet('color: #1E293B;')
        self.twins_label = QLabel('Графовое сходство: ожидание данных сессии')
        self.twins_label.setFont(QFont('Segoe UI', 8))
        self.twins_label.setStyleSheet('color: #1E293B;')
        self.action = QLabel()
        for widget in (self.primary, self.score, self.note, self.differential, self.evidence,
                       self.trajectory_label, self.twins_label, self.action):
            if isinstance(widget, QLabel):
                widget.setWordWrap(True)
                widget.setTextFormat(Qt.TextFormat.PlainText)
                widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            bl.addWidget(widget)
        scroll.setWidget(body)
        rl.addWidget(scroll, 1)
        self.export = QPushButton('СФОРМИРОВАТЬ СВОДНЫЙ PDF-ОТЧЕТ')
        self.export.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        self.export.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export.setEnabled(False)
        self.export.clicked.connect(self._export_pdf)
        rl.addWidget(self.export)
        split.addWidget(right, 2)
        outer.addLayout(split, 1)

        if not self.patient_graph.list_patients():
            self.patient_graph.create_patient('Пациент #01')
        self._populate_patients()

        session.updated.connect(self._update)
        session.pending.connect(self._pending)
        session.failed.connect(self._failed)

    def _panel(self, title):
        frame = QFrame()
        frame.setObjectName('cdssPanel')
        frame.setMinimumSize(0, 0)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(3)
        label = QLabel(title)
        label.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        label.setStyleSheet('color: #0F172A;')
        label.setWordWrap(True)
        layout.addWidget(label)
        return frame, layout

    def _populate_patients(self):
        self.patient_combo.blockSignals(True)
        self.patient_combo.clear()
        for p in self.patient_graph.list_patients():
            self.patient_combo.addItem(p['label'], p['id'])
        self.patient_combo.blockSignals(False)
        if self.patient_combo.count() > 0:
            self.patient_combo.setCurrentIndex(0)
            self._active_patient_id = self.patient_combo.currentData()
        self._update_trajectory_and_twins()

    def _on_new_patient_clicked(self):
        pid = self.patient_graph.create_patient()
        patients = self.patient_graph.list_patients()
        latest = next((p for p in patients if p['id'] == pid), None)
        label = latest['label'] if latest else pid
        self.patient_combo.addItem(label, pid)
        self.patient_combo.setCurrentIndex(self.patient_combo.count() - 1)
        self._active_patient_id = pid
        if self.session:
            win = self.window()
            if hasattr(win, 'reset_all_modules'):
                win.reset_all_modules()
            else:
                self.session.reset()
        self._update_trajectory_and_twins()
        if self.graph_mode == 'HISTORY':
            self._render_history_graph()
        win = self.window()
        if hasattr(win, 'patient_drawer') and win.patient_drawer:
            win.patient_drawer.refresh_cards()
            win.patient_drawer.sync_active_patient(pid)

    def _on_patient_selected(self, index):
        patient_id = self.patient_combo.currentData()
        if patient_id and patient_id != self._active_patient_id:
            self._active_patient_id = patient_id
            if self.session:
                win = self.window()
                if hasattr(win, 'reset_all_modules'):
                    win.reset_all_modules()
                else:
                    self.session.reset()
        self._update_trajectory_and_twins()
        if self.graph_mode == 'HISTORY':
            self._render_history_graph()
        win = self.window()
        if hasattr(win, 'patient_drawer') and win.patient_drawer:
            win.patient_drawer.sync_active_patient(patient_id)

    def _on_save_session_clicked(self):
        if self.session is None or self.result is None:
            return
        assessments = self.result.get('assessments') or self.session.assessments
        if not assessments:
            return
        patient_id = self.patient_combo.currentData()
        if not patient_id:
            patient_id = self.patient_graph.create_patient()
            self._populate_patients()
            self.patient_combo.setCurrentIndex(self.patient_combo.count() - 1)
            self._active_patient_id = patient_id
        self.patient_graph.commit_session(patient_id, assessments, synthesis_result=self.result)
        self._update_trajectory_and_twins()
        if self.graph_mode == 'HISTORY':
            self._render_history_graph()
        win = self.window()
        if hasattr(win, 'patient_drawer') and win.patient_drawer:
            win.patient_drawer.refresh_cards()

    def set_graph_mode(self, mode):
        self.graph_mode = mode
        if mode == 'HISTORY':
            self.mode_button.setText('ГРАФ ИСТОРИИ')
            self._render_history_graph()
        else:
            self.graph_mode = 'CURRENT'
            self.mode_button.setText('ТЕКУЩИЙ СРЕЗ')
            self.graph_view.set_result(self.result)

    def _toggle_mode(self):
        if self.graph_mode == 'CURRENT':
            self.set_graph_mode('HISTORY')
        else:
            self.set_graph_mode('CURRENT')

    def _render_history_graph(self):
        patient_id = self.patient_combo.currentData()
        if not patient_id:
            return
        if self.patient_graph.has_password(patient_id) and patient_id not in self.unlocked_patients:
            self.graph_view.scene().clear()
            self.graph_view.nodes = {}
            self.graph_view.edges = []
            caption = self.graph_view.scene().addSimpleText('Доступ к графу истории защищен паролем. Разблокируйте карту в картотеке.', QFont('Segoe UI', 10))
            caption.setBrush(QColor('#DC2626'))
            caption.setPos(10, 0)
            self.graph_view.scene().setSceneRect(0, 0, 1090, 310)
            return
        data = self.patient_graph.history_graph_data(patient_id)
        self.graph_view.set_history_data(data)

    def _update_trajectory_and_twins(self):
        patient_id = self.patient_combo.currentData() if hasattr(self, 'patient_combo') else None
        if not patient_id:
            self.trajectory_label.setText('Динамика пациента: не выбран')
            self.twins_label.setText('Графовое сходство: нет данных')
            return
        traj = self.patient_graph.get_patient_trajectory(patient_id)
        overall = traj.get('overall_trend', 'НЕТ ДАННЫХ')
        deltas = traj.get('deltas', [])
        if not traj.get('sessions'):
            self.trajectory_label.setText('Динамика пациента: нет сохраненных сессий')
        elif not deltas:
            self.trajectory_label.setText('Динамика пациента: первичная сессия')
        else:
            latest = deltas[-1]
            summary = latest.get('delta_summary', overall)
            self.trajectory_label.setText(f'Динамика пациента: {overall} — {summary}')

        bios = []
        syn_scores = {}
        if self.result and self.result.get('assessments'):
            bios = list(self.result.get('active', {}).keys())
            syn_scores = {item['id']: item['score'] for item in self.result.get('ranking', [])}
        elif traj.get('sessions'):
            last_s = traj['sessions'][-1]
            bios = last_s.get('active_biomarkers', [])
            syn_scores = last_s.get('syndrome_scores', {})

        if bios or syn_scores:
            twins = self.patient_graph.find_clinical_twins(bios, syn_scores, exclude_patient_id=patient_id, top_k=2)
            if twins:
                top = twins[0]
                self.twins_label.setText(f'Графовое сходство: {top["label"]} ({top["similarity"]:.0f}%)')
            else:
                self.twins_label.setText('Графовое сходство: нет близких профилей')
        else:
            self.twins_label.setText('Графовое сходство: ожидание данных сессии')

    def _pending(self, revision):
        self.result = None
        self.primary.setText('Обновление графа…')
        self.score.setValue(0)
        self.differential.clear()
        self.evidence.clear()
        self.action.clear()
        self.trajectory_label.setText('Динамика пациента: обновление…')
        self.twins_label.setText('Графовое сходство: обновление…')
        if self.graph_mode == 'CURRENT':
            self.graph_view.set_result(None)
        self.export.setEnabled(False)
        self.save_session_button.setEnabled(False)

    def _failed(self, message):
        self.primary.setText('Граф не рассчитан: ' + message)
        self.export.setEnabled(False)
        self.save_session_button.setEnabled(False)

    def _update(self, result):
        self.result = result
        if self.graph_mode == 'CURRENT':
            self.graph_view.set_result(result)
        else:
            self._render_history_graph()
        ranked = result['ranking']
        if ranked:
            primary = ranked[0]
            self.primary.setText('Гипотеза для проверки: ' + primary['label'])
            self.score.setValue(round(primary['score']))
            self.differential.setText('Другие направления:\n' + '\n'.join(
                f'{item["score"]:.0f}% — {item["label"]}' for item in ranked[1:]) if len(ranked) > 1 else 'Других ранжированных направлений нет.')
            evidence = ['• ' + BIOMARKERS[k][0] for k in primary['supporting']]
            evidence += ['• Противоречие: ' + BIOMARKERS[k][0] for k in primary['opposing']]
            unlinked = [k for k in result['active'] if result['graph'].out_degree(k) == 0]
            evidence += ['• Без заданной связи: ' + BIOMARKERS[k][0] for k in unlinked]
            self.evidence.setText('Вклады правил:\n' + '\n'.join(evidence))
        else:
            self.primary.setText({'empty': 'Ожидание завершения диагностических тестов',
                                  'no_flags': 'Сработавших порогов нет; патология не исключена',
                                  'limited': 'Данных для сопоставления недостаточно',
                                  'flagged': 'Есть биомаркеры без заданных связей'}[result['status']])
            self.score.setValue(0)
            self.differential.setText('Не выполнены: ' + (', '.join(result['missing']) or 'все пять проб выполнены'))
            self.evidence.setText('\n'.join(BIOMARKERS[k][0] for k in result['active']))
        self.action.setText('\n'.join(result['limited'] + [result['action']]))
        self._update_trajectory_and_twins()
        self.export.setEnabled(bool(result['assessments']))
        self.save_session_button.setEnabled(bool(result['assessments']))

    def _export_pdf(self):
        if self.result is None or not self.result['assessments']:
            return
        result = self.result
        path, _ = QFileDialog.getSaveFileName(self, 'Сохранить сводный PDF-отчёт', 'multimodal_session.pdf', 'PDF (*.pdf)')
        if not path:
            return
        output = Path(path)
        if not output.suffix:
            output = output.with_suffix('.pdf')
        try:
            patient_id = self.patient_combo.currentData() if hasattr(self, 'patient_combo') else None
            patient_label = self.patient_combo.currentText() if hasattr(self, 'patient_combo') else None
            traj = self.patient_graph.get_patient_trajectory(patient_id) if (patient_id and hasattr(self, 'patient_graph')) else None
            bios = list(result.get('active', {}).keys())
            syn_scores = {item['id']: item['score'] for item in result.get('ranking', [])}
            twins = self.patient_graph.find_clinical_twins(bios, syn_scores, exclude_patient_id=patient_id, top_k=2) if hasattr(self, 'patient_graph') else None
            report = summary_text(result, patient_label=patient_label, trajectory=traj, twins=twins)
            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(str(output))
            document = QTextDocument()
            document.setDefaultFont(QFont('Segoe UI', 9))
            document.setHtml('<pre style="white-space:pre-wrap">' + escape(report) + '</pre>')
            document.print(printer)
            if not output.is_file() or output.stat().st_size == 0:
                raise OSError('PDF не создан.')
        except (OSError, RuntimeError) as exc:
            QMessageBox.warning(self, 'Не удалось сохранить отчёт', str(exc))
