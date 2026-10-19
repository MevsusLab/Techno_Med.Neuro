# карточки модулей для дашборда

import math

from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QScrollArea,
                            QSizePolicy, QVBoxLayout, QWidget)


modules_catalog = (
    ('tremor', 'ТРЕМОР', 1, 'Анализ спектра и частоты покоя 1-12 Гц через FFT/PSD'),
    ('facial', 'НЕЙРОМИМИКА', 2, 'Экспресс-скрининг асимметрии лица и пареза по протоколу FAST'),
    ('voice', 'ГОЛОС', 5, 'Акустический анализ устойчивой фонации: F0, Jitter, Shimmer, HNR'),
    ('motor', 'МОТОРИКА', 6, 'Кинематика пальцевого таппинга, частота и декремент амплитуды'),
    ('gaze', 'АЙТРЕКИНГ', 4, 'Бесконтактная ассистивная клавиатура по фиксации взгляда (Dwell 900 мс)'),
    ('pulse', 'ПУЛЬС (rPPG)', 3, 'Бесконтактная детекция ЧСС по микропульсациям кожи лица (45-180 BPM)'),
)
MODULES = modules_catalog


class ModulePreview(QWidget):
    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setFixedHeight(76)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor('#F8FAFC'))
        painter.setPen(QPen(QColor('#E2E8F0'), 1))
        painter.drawRect(QRectF(.5, .5, self.width() - 1, self.height() - 1))
        painter.save()
        if self.kind in ('facial', 'gaze'):
            scale = min(max(1, self.width() - 18) / 300, (self.height() - 14) / 70)
            painter.translate((self.width() - 300 * scale) / 2, 7)
            painter.scale(scale, scale)
        else:
            painter.translate(9, 7)
            painter.scale(max(1, self.width() - 18) / 300, (self.height() - 14) / 70)
        painter.setPen(QPen(QColor('#334155'), 1.2))
        getattr(self, '_draw_' + self.kind)(painter)
        painter.restore()

    @staticmethod
    def _curve(painter, points, color='#2563EB', width=1.6):
        path = QPainterPath()
        for i, (x, y) in enumerate(points):
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        painter.setPen(QPen(QColor(color), width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def _draw_tremor(self, p):
        hand = QPainterPath()
        hand.moveTo(20, 66)
        hand.lineTo(17, 45)
        hand.cubicTo(3, 32, 4, 24, 12, 29)
        hand.lineTo(26, 40)
        hand.lineTo(22, 13)
        hand.cubicTo(21, 4, 30, 4, 31, 12)
        hand.lineTo(35, 32)
        hand.lineTo(35, 6)
        hand.cubicTo(35, -1, 44, -1, 44, 6)
        hand.lineTo(45, 31)
        hand.lineTo(49, 9)
        hand.cubicTo(50, 1, 59, 3, 57, 11)
        hand.lineTo(55, 34)
        hand.lineTo(63, 18)
        hand.cubicTo(68, 10, 75, 15, 70, 23)
        hand.lineTo(65, 44)
        hand.lineTo(59, 66)
        p.drawPath(hand)
        self._curve(p, [(82 + i * 1.75, 35 + 23 * math.exp(-i / 105) * math.sin(i * .34))
                        for i in range(120)])

    def _draw_facial(self, p):
        p.setPen(QPen(QColor('#CBD5E1'), 1))
        for y, width in ((14, 25), (24, 40), (36, 46), (48, 38), (60, 23)):
            self._curve(p, [(150 - width, y), (150, y + 3), (150 + width, y)], '#CBD5E1', .8)
        contour = [(150 + 38 * math.cos(a) * (1 - .22 * max(0, math.sin(a))), 34 + 32 * math.sin(a))
                   for a in [i * math.pi / 14 for i in range(29)]]
        self._curve(p, contour, '#334155', 1)
        for sign in (-1, 1):
            for start, y, dy in ((10, 18, -3), (12, 29, 2)):
                pts = [(150 + sign * (start + i * 6), y + dy * math.sin(i * math.pi / 4)) for i in range(5)]
                self._curve(p, pts, '#334155', .9)
            self._curve(p, [(150 + sign * 4, 31), (150 + sign * 8, 43), (150, 46)], '#CBD5E1', .9)
        lips = [(126, 53), (138, 50), (150, 52), (162, 50), (174, 53), (162, 57), (150, 58), (138, 57), (126, 53)]
        self._curve(p, lips, '#2563EB', 1)
        p.setPen(QPen(QColor('#2563EB'), 1))
        p.drawLine(QPointF(150, 0), QPointF(150, 70))
        p.setBrush(QColor('#2563EB'))
        for x, y in contour[::2] + lips[:-1] + [(150, 18), (150, 26), (150, 43), (126, 27), (174, 27)]:
            p.drawEllipse(QPointF(x, y), 1.8, 1.8)

    def _draw_voice(self, p):
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(36):
            height = 4 + 46 * math.exp(-((i % 7) / 1.65) ** 2) * math.exp(-i / 37)
            p.setBrush(QColor('#2563EB' if i % 7 <= 1 else '#CBD5E1'))
            p.drawRect(QRectF(7 + i * 8, 66 - height, 4, height))
        self._curve(p, [(6 + i * 2.4, 14 + 5 * math.sin(i * .085) + 2 * math.sin(i * .22))
                        for i in range(120)], '#334155', 1.4)

    def _draw_motor(self, p):
        self._curve(p, [(12 + i * 2.32, 58 - (38 - i * .22) * (.5 + .5 * math.cos(i * .6)))
                        for i in range(120)], '#CBD5E1', .9)
        self._curve(p, [(12, 17), (290, 45)], '#334155', 1.4)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor('#2563EB'))
        for i in range(14):
            p.drawEllipse(QPointF(12 + i * 21, 17 + i * 2.15 + 2 * math.sin(i * 1.9)), 2.7, 2.7)

    def _draw_gaze(self, p):
        eye = QPainterPath()
        eye.moveTo(65, 35)
        eye.cubicTo(115, -5, 185, -5, 235, 35)
        eye.cubicTo(185, 75, 115, 75, 65, 35)
        p.drawPath(eye)
        p.setPen(QPen(QColor('#2563EB'), 1.3))
        p.drawEllipse(QPointF(150, 35), 20, 20)
        p.setPen(QPen(QColor('#94A3B8'), 1))
        p.drawLine(QPointF(150, 4), QPointF(150, 66))
        p.drawLine(QPointF(112, 35), QPointF(188, 35))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor('#2563EB'))
        p.drawEllipse(QPointF(150, 35), 4, 4)

    def _draw_pulse(self, p):
        points = []
        for i in range(240):
            phase = (i / 52) % 1
            signal = (math.exp(-((phase - .23) / .12) ** 2)
                      + .3 * math.exp(-((phase - .53) / .1) ** 2))
            points.append((5 + i * 1.21, 56 - 42 * signal))
        self._curve(p, points)


class ModuleCard(QFrame):
    activated = pyqtSignal(int)

    def __init__(self, kind, title, module_index, description, parent=None):
        super().__init__(parent)
        self.module_index = module_index
        self._pressed = False
        self.setObjectName('modulePreviewCard')
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName('Открыть модуль ' + title)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet('''
            QFrame#modulePreviewCard { background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 0px; }
            QFrame#modulePreviewCard:hover, QFrame#modulePreviewCard:focus { border: 1px solid #2563EB; }
            QFrame#modulePreviewCard QLabel { background: transparent; border: none; border-radius: 0px; }
        ''')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        header = QHBoxLayout()
        header.setSpacing(6)
        self.title = QLabel(title)
        self.title.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        self.title.setStyleSheet('color: #0F172A;')
        header.addWidget(self.title, 1)
        self.status = QLabel('')
        self.status.setVisible(False)
        layout.addLayout(header)
        self.preview = ModulePreview(kind)
        layout.addWidget(self.preview)
        self.description = QLabel(description)
        self.description.setFont(QFont('Segoe UI', 8))
        self.description.setStyleSheet('color: #475569;')
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        for child in (self.title, self.status, self.description):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setToolTip('Открыть модуль «' + title + '». Иллюстрация интерфейса, не результат измерения.')

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        activate = (self._pressed and event.button() == Qt.MouseButton.LeftButton
                    and self.rect().contains(event.position().toPoint()))
        self._pressed = False
        if activate:
            self.activated.emit(self.module_index)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            if not event.isAutoRepeat():
                self.activated.emit(self.module_index)
            event.accept()
        else:
            super().keyPressEvent(event)


class DashboardCatalog(QScrollArea):
    module_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('dashboardCatalog')
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumSize(0, 0)
        self.setStyleSheet('''
            QScrollArea#dashboardCatalog { background: #FFFFFF; border: none; border-radius: 0px; }
            QScrollBar:vertical { background: #F8FAFC; width: 8px; margin: 0px; }
            QScrollBar::handle:vertical { background: #CBD5E1; min-height: 24px; border-radius: 0px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: #F8FAFC; }
        ''')
        body = QWidget()
        body.setStyleSheet('background: #FFFFFF;')
        columns = QHBoxLayout(body)
        columns.setContentsMargins(0, 0, 8, 0)
        columns.setSpacing(12)
        self.cards = []
        for title, modules, stretch in (
            ('ИССЛЕДОВАТЕЛЬСКИЕ МОДУЛИ', MODULES[:4], 2),
            ('ВСПОМОГАТЕЛЬНЫЕ И МОНИТОРИНГОВЫЕ МОДУЛИ', MODULES[4:], 1),
        ):
            section = QWidget()
            section.setMinimumWidth(0)
            section.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            layout = QVBoxLayout(section)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
            heading = QLabel(title)
            heading.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
            heading.setWordWrap(True)
            heading.setFixedHeight(32)
            heading.setStyleSheet('color: #1E293B; border: none;')
            layout.addWidget(heading)
            grid = QGridLayout()
            grid.setSpacing(10)
            for i, args in enumerate(modules):
                card = ModuleCard(*args)
                card.activated.connect(self.module_requested)
                grid.addWidget(card, i // 2 if stretch == 2 else i, i % 2 if stretch == 2 else 0)
                self.cards.append(card)
            if stretch == 2:
                grid.setColumnStretch(0, 1)
                grid.setColumnStretch(1, 1)
            else:
                grid.setColumnStretch(0, 1)
            layout.addLayout(grid)
            layout.addStretch()
            columns.addWidget(section, stretch)
        self.setWidget(body)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # подгоняем высоту под разрешение экрана
        height = 174 if self.viewport().width() < 1250 else 158
        for card in self.cards:
            card.setFixedHeight(height)
