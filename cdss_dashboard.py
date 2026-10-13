"""Light engineering graph and research synthesis panel; no clinical probabilities."""

from html import escape
import math
from pathlib import Path

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QTextDocument
from PyQt6.QtPrintSupport import QPrinter
from PyQt6.QtWidgets import (QFileDialog, QFrame, QGraphicsRectItem, QGraphicsScene,
                            QGraphicsSimpleTextItem, QGraphicsView, QHBoxLayout, QLabel,
                            QMessageBox, QProgressBar, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from session_diagnostics import BIOMARKERS, HYPOTHESES, MODULE_NAMES, ontology


CDSS_DISCLAIMER = (
    'ДИСКЛЕЙМЕР: Мультимодальный графовый анализ является исследовательской системой поддержки '
    'принятия решений (CDSS) и не заменяет комплексную очную диагностику врача-невролога.'
)
SCORE_NOTE = 'Индекс поддержки правил, не вероятность диагноза. Веса не прошли клиническую калибровку.'


def fmt(value):
    if value is None:
        return 'не выделено'
    return f'{value:.2f}'.replace('.', ',') if isinstance(value, (int, float)) else str(value)


def summary_text(result):
    lines = ['TecnoMedNeuro / СВОДНЫЙ МУЛЬТИМОДАЛЬНЫЙ ОТЧЁТ', SCORE_NOTE,
             'Состояние: ' + result['status'], '']
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
    lines += result['limited'] + ['', CDSS_DISCLAIMER, '', result['action']]
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

    def set_result(self, result):
        self.scene().clear()
        self.nodes = {}
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
            # Split target titles so the complete clinical meaning remains readable.
            words = label.split()
            midpoint = max(1, len(words)//2)
            title = ' '.join(words[:midpoint]) + '\n' + ' '.join(words[midpoint:])
            badge = f'  [ {score:.0f}% ]' if reached else '  [ — ]'
            self._node(key, title + badge, QRectF(640, 25 + i * 46, 440, 42), reached, True,
                       key + '\n' + SCORE_NOTE)
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
        caption = 'Ожидание завершения диагностических тестов' if not result or result['status'] == 'empty' else 'Колесо: масштаб / перетаскивание: обзор / двойной щелчок: весь граф'
        text = self.scene().addSimpleText(caption, QFont('Segoe UI', 10))
        text.setBrush(QColor('#64748B'))
        text.setPos(10, 0)
        self.scene().setSceneRect(0, 0, 1090, 310)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class CDSSDashboard(QWidget):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.result = None
        self.setFixedHeight(260)
        self.setStyleSheet('''
            QFrame#cdssPanel { background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 0px; }
            QLabel { color: #475569; background: transparent; border: none; }
            QScrollArea { background: #FFFFFF; border: none; }
            QPushButton { background: #FFFFFF; color: #1D4ED8; border: 1px solid #CBD5E1; padding: 5px; border-radius: 0px; }
            QPushButton:hover { border-color: #2563EB; }
            QPushButton:disabled { color: #94A3B8; }
            QProgressBar { background: #F8FAFC; color: #0F172A; border: 1px solid #E2E8F0; border-radius: 0px; }
            QProgressBar::chunk { background: #2563EB; border-radius: 0px; }
        ''')
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(5)
        split = QHBoxLayout()
        split.setSpacing(12)
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
        bl.setSpacing(4)
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
        self.action = QLabel()
        for widget in (self.primary, self.score, self.note, self.differential, self.evidence, self.action):
            if isinstance(widget, QLabel):
                widget.setWordWrap(True)
                widget.setTextFormat(Qt.TextFormat.PlainText)
                widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            bl.addWidget(widget)
        scroll.setWidget(body)
        rl.addWidget(scroll, 1)
        self.export = QPushButton('[ СФОРМИРОВАТЬ СВОДНЫЙ PDF-ОТЧЕТ ]')
        self.export.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        self.export.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export.setEnabled(False)
        self.export.clicked.connect(self._export_pdf)
        rl.addWidget(self.export)
        split.addWidget(right, 2)
        outer.addLayout(split, 1)
        disclaimer = QLabel(CDSS_DISCLAIMER)
        disclaimer.setFont(QFont('Consolas', 7))
        disclaimer.setWordWrap(True)
        outer.addWidget(disclaimer)
        session.updated.connect(self._update)
        session.pending.connect(self._pending)
        session.failed.connect(self._failed)

    def _panel(self, title):
        frame = QFrame()
        frame.setObjectName('cdssPanel')
        frame.setMinimumSize(0, 0)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        label = QLabel(title)
        label.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        label.setStyleSheet('color: #0F172A;')
        label.setWordWrap(True)
        layout.addWidget(label)
        return frame, layout

    def _pending(self, revision):
        self.result = None
        self.primary.setText('Обновление графа…')
        self.score.setValue(0)
        self.differential.clear()
        self.evidence.clear()
        self.action.clear()
        self.graph_view.set_result(None)
        self.export.setEnabled(False)

    def _failed(self, message):
        self.primary.setText('Граф не рассчитан: ' + message)
        self.export.setEnabled(False)

    def _update(self, result):
        self.result = result
        self.graph_view.set_result(result)
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
        self.export.setEnabled(bool(result['assessments']))

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
            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(str(output))
            document = QTextDocument()
            document.setDefaultFont(QFont('Segoe UI', 9))
            document.setHtml('<pre style="white-space:pre-wrap">' + escape(summary_text(result)) + '</pre>')
            document.print(printer)
            if not output.is_file() or output.stat().st_size == 0:
                raise OSError('PDF не создан.')
        except (OSError, RuntimeError) as exc:
            QMessageBox.warning(self, 'Не удалось сохранить отчёт', str(exc))
