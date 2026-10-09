import sys
import time
import traceback
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFrame, QGridLayout,
    QSizePolicy, QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QImage, QPixmap
from tremor_engine import TremorEngine
from capture_worker import ProcessCaptureWorker as CaptureWorker
from protocol_panel import ProtocolPanel, FacialProtocolPanel, RPPGProtocolPanel, VoiceProtocolPanel
from face_capture import FaceCaptureWorker
from facial_engine import analyze_stage, compare_stages, PAIRS
from rppg_capture import RPPGCaptureWorker
from rppg_engine import RPPGEngine
from voice_capture import VoiceCaptureWorker, VoiceFileCaptureWorker
from voice_dependencies import check_voice_dependencies
from i18n_ru import RussianLabel as QLabel, ru


BG = "#FFFFFF"
BG_ALT = "#F8FAFC"
NAVY = "#0F172A"
NAVY_MID = "#1E293B"
NAVY_DEEP = "#172554"
BORDER = "#CBD5E1"
BORDER_LIGHT = "#E2E8F0"
TEXT = "#0F172A"
TEXT_DIM = "#475569"
TEXT_MUTED = "#94A3B8"
GREEN = "#16A34A"
AMBER = "#D97706"
RED = "#DC2626"

def f_sans(size=11, bold=False):
    f = QFont("Segoe UI", size)
    f.setBold(bold)
    return f


def f_mono(size=11, bold=False):
    f = QFont("Consolas", size)
    f.setBold(bold)
    return f


def make_status_badge(text, color=TEXT_MUTED):
    from PyQt6.QtWidgets import QLabel as StatusLabel
    lbl = StatusLabel(text)
    lbl.setFont(f_mono(8, True))
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setFixedHeight(18)
    lbl.setStyleSheet(f"background: {color}; color: #FFFFFF; padding: 0 6px; border: none;")
    return lbl


def set_badge(lbl, text, color):
    lbl.setText(text)
    lbl.setStyleSheet(f"background: {color}; color: #FFFFFF; padding: 0 6px; border: none;")


def panel(title=""):
    frame = QFrame()
    frame.setStyleSheet(f"QFrame {{ background: {BG}; border: 1px solid {BORDER}; }}")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(10, 8, 10, 10)
    layout.setSpacing(6)
    if title:
        header = QHBoxLayout()
        lbl = QLabel(title.upper())
        lbl.setFont(f_sans(9, True))
        lbl.setStyleSheet(f"color: {TEXT_DIM}; border: none; letter-spacing: 1px;")
        header.addWidget(lbl)
        header.addStretch()
        layout.addLayout(header)
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {BORDER_LIGHT}; border: none;")
        layout.addWidget(sep)
    return frame, layout


def panel_with_badge(title, badge_text, badge_color=TEXT_MUTED):
    frame = QFrame()
    frame.setStyleSheet(f"QFrame {{ background: {BG}; border: 1px solid {BORDER}; }}")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(10, 8, 10, 10)
    layout.setSpacing(6)
    header = QHBoxLayout()
    lbl = QLabel(title.upper())
    lbl.setFont(f_sans(9, True))
    lbl.setStyleSheet(f"color: {TEXT_DIM}; border: none; letter-spacing: 1px;")
    header.addWidget(lbl)
    header.addStretch()
    badge = make_status_badge(badge_text, badge_color)
    header.addWidget(badge)
    layout.addLayout(header)
    sep = QFrame()
    sep.setFixedHeight(1)
    sep.setStyleSheet(f"background: {BORDER_LIGHT}; border: none;")
    layout.addWidget(sep)
    return frame, layout, badge


def feed_panel(label="VIDEO FEED"):
    frame = QFrame()
    frame.setMinimumSize(400, 300)
    frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    frame.setStyleSheet(f"QFrame {{ background: #0F172A; border: 1px solid {BORDER}; }}")
    lay = QVBoxLayout(frame)
    lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl = QLabel(label)
    lbl.setFont(f_mono(10))
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet("color: #475569; border: none;")
    lay.addWidget(lbl)
    res = QLabel("640x480 @ 30fps")
    res.setFont(f_mono(8))
    res.setAlignment(Qt.AlignmentFlag.AlignCenter)
    res.setStyleSheet("color: #334155; border: none;")
    lay.addWidget(res)
    return frame


def metric_readout(label, value="---", unit=""):
    w = QWidget()
    w.setStyleSheet("border: none;")
    lay = QVBoxLayout(w)
    lay.setContentsMargins(8, 4, 8, 4)
    lay.setSpacing(1)
    lbl = QLabel(label)
    lbl.setFont(f_sans(8))
    lbl.setStyleSheet(f"color: {TEXT_MUTED}; border: none;")
    lay.addWidget(lbl)
    row = QHBoxLayout()
    row.setSpacing(3)
    val = QLabel(value)
    val.setFont(f_mono(18, True))
    val.setStyleSheet(f"color: {TEXT}; border: none;")
    row.addWidget(val)
    if unit:
        u = QLabel(unit)
        u.setFont(f_mono(10))
        u.setStyleSheet(f"color: {TEXT_DIM}; border: none;")
        u.setAlignment(Qt.AlignmentFlag.AlignBottom)
        row.addWidget(u)
    row.addStretch()
    lay.addLayout(row)
    return w


def metric_readout_large(label, value="---", unit="", color=TEXT):
    w = QWidget()
    w.setStyleSheet("border: none;")
    lay = QVBoxLayout(w)
    lay.setContentsMargins(8, 4, 8, 4)
    lay.setSpacing(1)
    lbl = QLabel(label)
    lbl.setFont(f_sans(9))
    lbl.setStyleSheet(f"color: {TEXT_MUTED}; border: none;")
    lay.addWidget(lbl)
    row = QHBoxLayout()
    row.setSpacing(4)
    val = QLabel(value)
    val.setFont(f_mono(32, True))
    val.setStyleSheet(f"color: {color}; border: none;")
    row.addWidget(val)
    if unit:
        u = QLabel(unit)
        u.setFont(f_mono(14))
        u.setStyleSheet(f"color: {TEXT_DIM}; border: none;")
        u.setAlignment(Qt.AlignmentFlag.AlignBottom)
        row.addWidget(u)
    row.addStretch()
    lay.addLayout(row)
    return w


def graph_area(title="SIGNAL"):
    frame = QFrame()
    frame.setMinimumHeight(140)
    frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    frame.setStyleSheet(f"QFrame {{ background: {BG_ALT}; border: 1px solid {BORDER}; }}")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(8, 6, 8, 6)
    lbl = QLabel(title)
    lbl.setFont(f_sans(8, True))
    lbl.setStyleSheet(f"color: {TEXT_MUTED}; border: none; letter-spacing: 1px;")
    lay.addWidget(lbl)
    lay.addStretch()
    placeholder = QLabel("No data")
    placeholder.setFont(f_mono(9))
    placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
    placeholder.setStyleSheet(f"color: {BORDER}; border: none;")
    lay.addWidget(placeholder)
    lay.addStretch()
    return frame


def cmd_button(text, primary=True):
    btn = QPushButton(ru(text))
    btn.setFont(f_sans(10, True))
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedHeight(30)
    if primary:
        btn.setStyleSheet(f"""
            QPushButton {{ background: {NAVY}; color: #FFFFFF; border: 1px solid {NAVY}; padding: 0 16px; }}
            QPushButton:hover {{ background: {NAVY_MID}; }}
            QPushButton:disabled {{ background: #94A3B8; border-color: #94A3B8; }}
        """)
    else:
        btn.setStyleSheet(f"""
            QPushButton {{ background: {BG}; color: {NAVY}; border: 1px solid {BORDER}; padding: 0 16px; }}
            QPushButton:hover {{ background: {BG_ALT}; }}
            QPushButton:disabled {{ color: #CBD5E1; }}
        """)
    return btn


def planned_button(text, primary=True):
    btn = cmd_button(text, primary)
    btn.setEnabled(False)
    return btn


def thin_progress():
    p = QProgressBar()
    p.setFixedHeight(3)
    p.setValue(0)
    p.setTextVisible(False)
    p.setStyleSheet(f"""
        QProgressBar {{ background: {BORDER_LIGHT}; border: none; }}
        QProgressBar::chunk {{ background: {NAVY}; }}
    """)
    return p


class MetricDisplay(QWidget):
    def __init__(self, label, initial="---", unit=""):
        super().__init__()
        self.setStyleSheet("border: none;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(1)
        lbl = QLabel(label)
        lbl.setFont(f_sans(8))
        lbl.setStyleSheet(f"color: {TEXT_MUTED}; border: none;")
        lay.addWidget(lbl)
        row = QHBoxLayout()
        row.setSpacing(3)
        self.val = QLabel(initial)
        self.val.setFont(f_mono(18, True))
        self.val.setStyleSheet(f"color: {TEXT}; border: none;")
        row.addWidget(self.val)
        if unit:
            u = QLabel(unit)
            u.setFont(f_mono(10))
            u.setStyleSheet(f"color: {TEXT_DIM}; border: none;")
            u.setAlignment(Qt.AlignmentFlag.AlignBottom)
            row.addWidget(u)
        row.addStretch()
        lay.addLayout(row)

    def set_value(self, text, color=None):
        self.val.setText(str(text))
        if color:
            self.val.setStyleSheet(f"color: {color}; border: none;")


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 2), dpi=100, facecolor='#F8FAFC')
        self.fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.18)
        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        super().__init__(self.fig)
        self.setStyleSheet(f"border: 1px solid {BORDER};")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(140)

    def _style_axes(self):
        self.ax.set_facecolor('#FFFFFF')
        self.ax.tick_params(colors='#475569', labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_color('#CBD5E1')
        self.ax.grid(True, alpha=0.3, color='#CBD5E1', linewidth=0.5)

    def clear_plot(self):
        self.ax.clear()
        self._style_axes()
        self.draw_idle()


class NavBar(QWidget):
    def __init__(self, tabs, callback):
        super().__init__()
        self.setFixedHeight(36)
        self.setStyleSheet(f"QWidget {{ background: {NAVY}; border: none; }}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 12, 0)
        lay.setSpacing(0)
        brand = QLabel("TECNOMEDNEURO")
        brand.setFont(f_sans(10, True))
        brand.setStyleSheet("color: #FFFFFF; border: none; letter-spacing: 2px;")
        lay.addWidget(brand)
        sep = QLabel("|")
        sep.setFont(f_sans(10))
        sep.setStyleSheet("color: #334155; border: none; padding: 0 12px;")
        lay.addWidget(sep)
        self.buttons = []
        for i, label in enumerate(tabs):
            btn = QPushButton(ru(label))
            btn.setFont(f_sans(9))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setCheckable(True)
            btn.setFixedHeight(36)
            btn.clicked.connect(lambda checked, idx=i: callback(idx))
            btn.setStyleSheet(self._btn_style(False))
            lay.addWidget(btn)
            self.buttons.append(btn)
        lay.addStretch()
        ver = QLabel("v0.1")
        ver.setFont(f_mono(8))
        ver.setStyleSheet("color: #475569; border: none;")
        lay.addWidget(ver)

    def set_active(self, index):
        for i, btn in enumerate(self.buttons):
            btn.setChecked(i == index)
            btn.setStyleSheet(self._btn_style(i == index))

    def _btn_style(self, active):
        if active:
            return f"""
                QPushButton {{ background: #1E293B; color: #FFFFFF; border: none;
                    border-bottom: 2px solid #3B82F6; padding: 0 14px; font-weight: bold; }}
            """
        return f"""
            QPushButton {{ background: transparent; color: #94A3B8; border: none;
                border-bottom: 2px solid transparent; padding: 0 14px; }}
            QPushButton:hover {{ color: #E2E8F0; background: #1E293B; }}
        """


class StatusBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(22)
        self.setStyleSheet(f"background: {BG_ALT}; border-top: 1px solid {BORDER_LIGHT};")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.setSpacing(16)
        self.badges = {}
        for name, state, color in [
            ("CAM", "DISCONNECTED", TEXT_MUTED),
            ("MIC", "DISCONNECTED", TEXT_MUTED),
            ("MEDIAPIPE", "NOT LOADED", TEXT_MUTED),
            ("DSP", "STANDBY", TEXT_MUTED),
        ]:
            row = QHBoxLayout()
            row.setSpacing(4)
            n = QLabel(name)
            n.setFont(f_mono(7, True))
            n.setStyleSheet(f"color: {TEXT_DIM}; border: none;")
            row.addWidget(n)
            badge = make_status_badge(state, color)
            row.addWidget(badge)
            self.badges[name] = badge
            lay.addLayout(row)
        lay.addStretch()
        ts = QLabel("MevsusLab / NDTP")
        ts.setFont(f_mono(7))
        ts.setStyleSheet(f"color: {TEXT_MUTED}; border: none;")
        lay.addWidget(ts)

    def update_badge(self, name, text, color):
        if name in self.badges:
            set_badge(self.badges[name], text, color)


class DashboardPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)
        title = QLabel("SYSTEM OVERVIEW")
        title.setFont(f_sans(14, True))
        title.setStyleSheet(f"color: {TEXT};")
        layout.addWidget(title)
        top = QHBoxLayout()
        top.setSpacing(12)
        info_panel, info_lay = panel("Platform Information")
        info_grid = QGridLayout()
        info_grid.setSpacing(4)
        for r, (k, v) in enumerate([
            ("System", "TecnoMedNeuro Screening Suite"),
            ("Version", "0.1.0-alpha"),
            ("Backend", "MediaPipe + SciPy DSP"),
            ("Input", "USB Webcam + Microphone"),
            ("Modules", "4 prototypes / 1 planned"),
            ("Framework", "PyQt6 / OpenCV"),
        ]):
            kl = QLabel(k)
            kl.setFont(f_sans(9, True))
            kl.setStyleSheet(f"color: {TEXT_DIM}; border: none;")
            info_grid.addWidget(kl, r, 0)
            vl = QLabel(v)
            vl.setFont(f_mono(9))
            vl.setStyleSheet(f"color: {TEXT}; border: none;")
            info_grid.addWidget(vl, r, 1)
        info_lay.addLayout(info_grid)
        top.addWidget(info_panel)
        mod_panel, mod_lay = panel("Module Registry")
        table = QTableWidget(5, 3)
        table.setHorizontalHeaderLabels(["MODULE", "TYPE", "STATUS"])
        table.horizontalHeader().setFont(f_sans(8, True))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setStyleSheet(f"""
            QTableWidget {{ background: {BG}; border: 1px solid {BORDER_LIGHT};
                gridline-color: {BORDER_LIGHT}; font-family: Consolas; font-size: 10px; }}
            QHeaderView::section {{ background: {BG_ALT}; border: 1px solid {BORDER_LIGHT};
                padding: 4px; color: {TEXT_DIM}; }}
        """)
        for r, (name, typ, st) in enumerate([
            ("Hand Motion", "Research", "PROTOTYPE"),
            ("Facial Asymmetry", "Research", "PROTOTYPE"),
            ("rPPG Vitals", "Research", "PROTOTYPE"),
            ("Eye-Tracking AAC", "Assistive", "PLANNED"),
            ("Voice Biomarkers", "Research", "PROTOTYPE"),
        ]):
            table.setItem(r, 0, QTableWidgetItem(ru(name)))
            table.setItem(r, 1, QTableWidgetItem(ru(typ)))
            item = QTableWidgetItem(st)
            item.setForeground(QColor(TEXT_DIM))
            table.setItem(r, 2, item)
        table.setFixedHeight(180)
        mod_lay.addWidget(table)
        top.addWidget(mod_panel)
        layout.addLayout(top)
        arch_panel, arch_lay = panel("Architecture")
        arch_text = QLabel(
            "INPUT (Webcam / Mic)  -->  MediaPipe Landmark Extraction  "
            "-->  Signal Preprocessing (Butterworth / Hamming)  "
            "-->  Spectral Analysis (FFT / PSD)  -->  Classification / Metrics  "
            "-->  Motion Measurements (research use only)"
        )
        arch_text.setFont(f_mono(9))
        arch_text.setStyleSheet(f"color: {TEXT_DIM}; border: none;")
        arch_text.setWordWrap(True)
        arch_lay.addWidget(arch_text)
        layout.addWidget(arch_panel)
        bottom = QHBoxLayout()
        bottom.setSpacing(12)
        diag_panel, diag_lay = panel("Research Modules")
        for name, desc in [
            ("TREMOR", "Webcam hand-motion measurements: frequency, RMS displacement and spectrum"),
            ("FACIAL", "Rest / smile / brow-raise bilateral geometry and movement measurements"),
            ("VOICE", "Sustained vowel: F0, local jitter/shimmer, RAP/APQ5 and HNR"),
        ]:
            row = QHBoxLayout()
            tag = QLabel(name)
            tag.setFont(f_mono(8, True))
            tag.setFixedWidth(60)
            tag.setStyleSheet(f"color: {NAVY_DEEP}; border: none;")
            row.addWidget(tag)
            d = QLabel(desc)
            d.setFont(f_sans(9))
            d.setStyleSheet(f"color: {TEXT_DIM}; border: none;")
            d.setWordWrap(True)
            row.addWidget(d)
            diag_lay.addLayout(row)
        bottom.addWidget(diag_panel)
        assist_panel, assist_lay = panel("Assistive + Monitoring")
        for name, desc in [
            ("AAC", "Planned: eye-gaze communication board with dwell-time activation"),
            ("rPPG", "Skin-region green-channel pulse-rate research prototype"),
        ]:
            row = QHBoxLayout()
            tag = QLabel(name)
            tag.setFont(f_mono(8, True))
            tag.setFixedWidth(60)
            tag.setStyleSheet(f"color: {NAVY_DEEP}; border: none;")
            row.addWidget(tag)
            d = QLabel(desc)
            d.setFont(f_sans(9))
            d.setStyleSheet(f"color: {TEXT_DIM}; border: none;")
            d.setWordWrap(True)
            row.addWidget(d)
            assist_lay.addLayout(row)
        assist_lay.addStretch()
        bottom.addWidget(assist_panel)
        layout.addLayout(bottom)
        layout.addStretch()


class VideoLabel(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.source_frame = None
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def setPixmap(self, pixmap):
        self.source_frame = pixmap
        self._scale_frame()

    def _scale_frame(self):
        if self.source_frame is not None:
            super().setPixmap(self.source_frame.scaled(
                self.contentsRect().size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation))

    def clear(self):
        self.source_frame = None
        super().clear()

    def setText(self, text):
        self.source_frame = None
        super().setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_frame()


class TremorPage(QWidget):
    capture_finished = pyqtSignal()

    def __init__(self, status_bar=None):
        super().__init__()
        self.status_bar = status_bar
        self.engine = TremorEngine()
        self.worker = None
        self.recording = False
        self.record_start = None
        self.record_end = None
        self.record_duration = 15.0
        self.frame_count = 0
        self._stop_time = None
        self._analyze_on_finish = False
        self._failure = ""
        self._closing = False
        self._deadline_requested = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_frame)

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("HAND MOTION ANALYSIS")
        title.setFont(f_sans(13, True))
        title.setStyleSheet(f"color: {TEXT};")
        header.addWidget(title)
        header.addStretch()
        self.page_badge = make_status_badge("IDLE", TEXT_MUTED)
        header.addWidget(self.page_badge)
        layout.addLayout(header)

        sub = QLabel("Research prototype  |  1-12 Hz (camera-dependent)  |  Hand held still  |  Not a medical diagnosis")
        sub.setFont(f_mono(8))
        sub.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(sub)

        body = QHBoxLayout()
        body.setSpacing(8)

        left = QVBoxLayout()
        left.setSpacing(8)

        self.capture_area = QWidget()
        self.capture_area.setMinimumSize(0, 0)
        self.capture_layout = QVBoxLayout(self.capture_area)
        self.capture_layout.setContentsMargins(0, 0, 0, 0)
        self.capture_layout.setSpacing(0)
        self.video_label = VideoLabel("HAND TRACKING FEED\n\n640x480 @ 30fps")
        self.video_label.setMinimumSize(480, 0)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFont(f_mono(10))
        self.video_label.setStyleSheet(f"background: {NAVY}; border: 1px solid {BORDER}; color: #475569;")
        self.capture_layout.addWidget(self.video_label, 1)
        self.protocol_panel = ProtocolPanel()
        self.protocol_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.capture_layout.addWidget(self.protocol_panel, 1)
        left.addWidget(self.capture_area, 1)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.btn_start = cmd_button("START CAPTURE")
        self.btn_start.clicked.connect(lambda checked=False: self._start())
        self.btn_stop = cmd_button("STOP", primary=False)
        self.btn_stop.clicked.connect(lambda checked=False: self._stop())
        self.btn_stop.setEnabled(False)
        self.btn_reset = cmd_button("RESET", primary=False)
        self.btn_reset.clicked.connect(lambda checked=False: self._reset())
        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_stop)
        controls.addWidget(self.btn_reset)
        left.addLayout(controls)

        self.progress = thin_progress()
        left.addWidget(self.progress)

        self.info_label = QLabel("Samples: 0  |  FPS: --  |  0.0s / 15.0s")
        self.info_label.setFont(f_mono(8))
        self.info_label.setStyleSheet(f"color: {TEXT_MUTED};")
        self.info_label.setWordWrap(True)
        left.addWidget(self.info_label)

        body.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(8)

        m_frame, m_lay, self.analysis_badge = panel_with_badge("Analysis", "WAITING", TEXT_MUTED)
        m_grid = QGridLayout()
        m_grid.setSpacing(2)
        self.m_freq = MetricDisplay("Dominant Frequency", "---", "Hz")
        self.m_amp = MetricDisplay("Motion RMS (image)", "---", "px")
        self.m_class = MetricDisplay("Signal Assessment", "---")
        self.m_class.val.setFont(f_mono(11, True))
        self.m_conf = MetricDisplay("Peak Concentration (+/-1 Hz)", "---", "%")
        m_grid.addWidget(self.m_freq, 0, 0)
        m_grid.addWidget(self.m_amp, 0, 1)
        m_grid.addWidget(self.m_class, 1, 0)
        m_grid.addWidget(self.m_conf, 1, 1)
        m_lay.addLayout(m_grid)
        self.analysis_detail = QLabel("Hold one hand in view. Preparation comes before the 15-second recording.")
        self.analysis_detail.setFont(f_sans(9))
        self.analysis_detail.setWordWrap(True)
        self.analysis_detail.setTextFormat(Qt.TextFormat.PlainText)
        self.analysis_detail.setStyleSheet(f"color: {TEXT_DIM}; border: none;")
        m_lay.addWidget(self.analysis_detail)
        right.addWidget(m_frame)

        self.psd_canvas = PlotCanvas()
        right.addWidget(self.psd_canvas)

        self.wave_canvas = PlotCanvas()
        right.addWidget(self.wave_canvas)

        body.addLayout(right, 2)
        layout.addLayout(body)

    def _start(self):
        if self.worker is not None or self._closing:
            return
        self._reset()
        self._analyze_on_finish = True
        self.worker = CaptureWorker(self.record_duration, self)
        self.worker.status.connect(self._on_status)
        self.worker.failed.connect(self._on_failure)
        self.worker.finished.connect(self._on_finished)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        set_badge(self.page_badge, "STARTING", AMBER)
        set_badge(self.analysis_badge, "WAITING", TEXT_MUTED)
        self.info_label.setText("Starting model and camera...")
        if self.status_bar:
            self.status_bar.update_badge("MEDIAPIPE", "STARTING", AMBER)
        self.timer.start(30)
        self.worker.start()

    def _stop(self, analyze=True):
        if self.worker is None:
            return
        if self._stop_time is None:
            self._stop_time = time.perf_counter()
        self._analyze_on_finish = analyze and not self._failure and not self._closing
        self.recording = False
        self.btn_stop.setEnabled(False)
        self.worker.requestInterruption()
        if not self._failure:
            set_badge(self.page_badge, "STOPPING", AMBER)
            set_badge(self.analysis_badge, "WAITING" if analyze else "CANCELLED", TEXT_MUTED)

    def _reset(self):
        self.protocol_panel.clear()
        if self.worker is not None:
            self._stop(analyze=False)
        else:
            self._stop_time = None
        self.recording = False
        self.record_start = None
        self.record_end = None
        self.frame_count = 0
        self._analyze_on_finish = False
        self._failure = ""
        self._deadline_requested = False
        self.engine.reset()
        self.progress.setValue(0)
        self.info_label.setText("Samples: 0  |  FPS: --  |  0.0s / 15.0s")
        self.m_freq.set_value("---")
        self.m_amp.set_value("---")
        self.m_class.set_value("---", TEXT)
        self.m_conf.set_value("---")
        self.analysis_detail.setText("Hold one hand in view. Preparation comes before the 15-second recording.")
        self.psd_canvas.clear_plot()
        self.wave_canvas.clear_plot()
        set_badge(self.page_badge, "IDLE", TEXT_MUTED)
        set_badge(self.analysis_badge, "WAITING", TEXT_MUTED)
        self.btn_start.setEnabled(self.worker is None and not self._closing)
        self.btn_stop.setEnabled(False)
        if self.status_bar:
            self.status_bar.update_badge("CAM", "DISCONNECTED", TEXT_MUTED)
            self.status_bar.update_badge("MEDIAPIPE", "NOT LOADED", TEXT_MUTED)
            self.status_bar.update_badge("DSP", "STANDBY", TEXT_MUTED)
        self.video_label.clear()
        self.video_label.setText("HAND TRACKING FEED\n\n640x480 @ 30fps")
        self.video_label.setStyleSheet(f"background: {NAVY}; border: 1px solid {BORDER}; color: #475569;")

    def _on_status(self, status):
        if self._stop_time is not None or self._failure or self._closing:
            return
        self.info_label.setText(status)
        self.analysis_detail.setText(status)
        if status.startswith("PREPARING"):
            set_badge(self.page_badge, "PREPARING", AMBER)
        if self.status_bar:
            if status == "CAM OPENING":
                self.status_bar.update_badge("MEDIAPIPE", "LOADED", GREEN)
                self.status_bar.update_badge("CAM", "STARTING", AMBER)
            elif status == "CAM READY":
                self.status_bar.update_badge("CAM", "ACTIVE", GREEN)

    def _on_failure(self, message):
        if not self._analyze_on_finish or self._closing:
            return
        self.protocol_panel.clear()
        self._failure = message
        self._stop(analyze=False)
        self.engine.reset()
        self.psd_canvas.clear_plot()
        self.wave_canvas.clear_plot()
        set_badge(self.page_badge, "CAPTURE ERROR", RED)
        set_badge(self.analysis_badge, "CAPTURE FAILED", RED)
        self.info_label.setText(message)
        self.analysis_detail.setText(message)
        self.m_freq.set_value("---")
        self.m_amp.set_value("---")
        self.m_conf.set_value("---")
        self.m_class.set_value("CAPTURE FAILED", RED)
        self.video_label.clear()
        self.video_label.setText("CAPTURE FAILED")
        if self.status_bar:
            self.status_bar.update_badge("CAM", "STOPPING", AMBER)
            self.status_bar.update_badge("DSP", "STANDBY", TEXT_MUTED)

    def _on_frame(self):
        if self.worker is None:
            return
        start, end, read_wait, read_timeout = self.worker.snapshot()
        latest, samples = self.worker.take_pending()
        if not self._analyze_on_finish:
            return
        if start is not None:
            self.record_start = start
            self.record_end = end
            self.recording = self._stop_time is None and not self.worker.isInterruptionRequested()
            if self.recording:
                set_badge(self.page_badge, "RECORDING", GREEN)
                set_badge(self.analysis_badge, "RECORDING", GREEN)
                self.analysis_detail.setText("Recording. Keep the hand fully visible until capture finishes.")
        old_count = self.frame_count
        for acquired, x, y in samples:
            if self._stop_time is not None and acquired > self._stop_time:
                continue
            self.frame_count += 1
            if x is not None and y is not None:
                self.engine.add_sample(x, y, acquired)
        now = time.perf_counter()
        if self.record_start is not None:
            current = self._stop_time if self._stop_time is not None else now
            elapsed = min(max(current - self.record_start, 0), self.record_duration)
            self.progress.setValue(int(elapsed / self.record_duration * 100))
            fps_est = self.frame_count / max(elapsed, 0.001)
            self.info_label.setText(
                f"Samples: {self.engine.count}  |  FPS: {fps_est:.0f}  |  "
                f"{elapsed:.1f}s / {self.record_duration:.0f}s"
                + ("  |  NO HAND" if latest is not None and latest[2] is None else "")
            )
            if self.frame_count // 15 > old_count // 15 and self.engine.count > 10:
                self._update_live_wave()
        if self._stop_time is None and not self.worker.isInterruptionRequested():
            if read_wait is not None and now - read_wait > read_timeout:
                self._on_failure("Camera stopped delivering frames")
                return
            if self.record_start is not None and now - self.record_start >= self.record_duration:
                self.recording = False
                self._deadline_requested = True
                self.btn_stop.setEnabled(False)
                self.worker.requestInterruption()
                set_badge(self.page_badge, "STOPPING", AMBER)
        if latest is not None:
            display = latest[0]
            h, w = display.shape[:2]
            qimg = QImage(display.data, w, h, display.strides[0], QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pixmap)

    def _on_finished(self):
        worker = self.worker
        if worker is None:
            return
        try:
            self._finish_capture(worker)
        except Exception as exc:
            self._analysis_failed(exc)
        finally:
            self.timer.stop()
            self.recording = False
            self.worker = None
            self.btn_start.setEnabled(not self._closing)
            self.btn_stop.setEnabled(False)
            worker.deleteLater()
            if self.status_bar:
                self.status_bar.update_badge("CAM", "DISCONNECTED", TEXT_MUTED)
                self.status_bar.update_badge("MEDIAPIPE", "NOT LOADED", TEXT_MUTED)
            self.capture_finished.emit()

    def _finish_capture(self, worker):
        if worker.error:
            self._on_failure(worker.error)
        elif worker.reason == "stopped" and self._stop_time is None and not self._closing and not self._deadline_requested:
            self._analyze_on_finish = False
            self._failure = "Capture ended before the recording was complete"
            set_badge(self.page_badge, "CAPTURE STOPPED", AMBER)
            set_badge(self.analysis_badge, "INCOMPLETE", AMBER)
            self.analysis_detail.setText(self._failure)
        self._on_frame()
        self.timer.stop()
        self.recording = False
        if self._analyze_on_finish and not self._closing:
            self.record_end = self._stop_time if self._stop_time is not None else self.record_end
            self.video_label.clear()
            if self.record_start is not None:
                self.video_label.setText("RECORDING COMPLETE")
                self._run_analysis()
            else:
                self.video_label.setText("CAPTURE CANCELLED")
                set_badge(self.page_badge, "IDLE", TEXT_MUTED)
                set_badge(self.analysis_badge, "WAITING", TEXT_MUTED)
                self.info_label.setText("Capture cancelled before the first frame")

    def _update_live_wave(self):
        t = np.array(self.engine.timestamps)
        y = np.array(self.engine.y)
        t_rel = t - t[0]
        y_centered = y - np.mean(y)

        ax = self.wave_canvas.ax
        ax.clear()
        self.wave_canvas._style_axes()
        ax.plot(t_rel, y_centered, color=NAVY, linewidth=0.6)
        ax.set_xlabel('Time (s)', fontsize=7, color=TEXT_DIM)
        ax.set_ylabel('Y displacement (px)', fontsize=7, color=TEXT_DIM)
        ax.set_title('LIVE WAVEFORM', fontsize=8, fontweight='bold', color=TEXT_DIM, loc='left')
        self.wave_canvas.draw_idle()

    def _run_analysis(self):
        try:
            self._analyze_and_render()
        except Exception as exc:
            self._analysis_failed(exc)

    def _analysis_failed(self, exc):
        self.protocol_panel.clear()
        traceback.print_exc()
        set_badge(self.page_badge, "ANALYSIS ERROR", RED)
        set_badge(self.analysis_badge, "FAILED", RED)
        self.analysis_detail.setText(f"Analysis failed: {type(exc).__name__}: {exc}. Capture can be retried.")
        self.video_label.setText("ANALYSIS FAILED")
        self.m_freq.set_value("---")
        self.m_amp.set_value("---")
        self.m_conf.set_value("---")
        self.m_class.set_value("ANALYSIS FAILED", RED)
        self.psd_canvas.clear_plot()
        if self.status_bar:
            self.status_bar.update_badge("DSP", "ERROR", RED)

    def _analyze_and_render(self):
        self.protocol_panel.clear()
        set_badge(self.analysis_badge, "PROCESSING", AMBER)
        results = self.engine.analyze(record_start=self.record_start, record_end=self.record_end)

        if results is None:
            set_badge(self.page_badge, "RETRY", AMBER)
            set_badge(self.analysis_badge, "INVALID CAPTURE", AMBER)
            self.m_class.set_value("INVALID CAPTURE", AMBER)
            self.info_label.setText(self.engine.last_error)
            self.analysis_detail.setText(self.engine.last_error)
            quality = self.engine.last_quality
            if 'fs' in quality:
                self.info_label.setText(
                    f"Samples: {quality['samples']}  |  Rate: {quality['fs']:.1f}/s  |  "
                    f"Coverage: {quality['coverage'] * 100:.0f}%  |  Max gap: {quality['max_gap_sec']:.3f}s"
                )
            self.psd_canvas.clear_plot()
            self.psd_canvas.ax.text(0.5, 0.5, "Spectrum withheld: invalid capture", ha='center',
                                    va='center', transform=self.psd_canvas.ax.transAxes, color=TEXT_DIM, fontsize=9)
            self.psd_canvas.draw_idle()
            self.wave_canvas.ax.set_title('REJECTED CAPTURE / RAW PREVIEW', fontsize=8,
                                          fontweight='bold', color=AMBER, loc='left')
            self.wave_canvas.draw_idle()
            if self.status_bar:
                self.status_bar.update_badge("DSP", "REJECTED", AMBER)
            return

        if self.status_bar:
            self.status_bar.update_badge("DSP", "COMPLETE", GREEN)

        peak = results['dominant_freq']
        self.m_freq.set_value(f"{peak:.2f}" if peak is not None else "---")
        self.m_amp.set_value(f"{results['rms_px']:.2f}")
        self.m_conf.set_value(f"{results['concentration']:.1f}")
        self.m_class.set_value("SPECTRAL PEAK" if peak is not None else "NO CLEAR PEAK", TEXT)
        limited = results['max_freq'] < 11.95
        if limited:
            self.analysis_detail.setText(
                f"Limited to 1-{results['max_freq']:.1f} Hz by a {results['max_gap_sec']:.3f}s frame gap. "
                + ("Energy is strongest outside the usable band. " if results.get('above_band') else "")
                + "Faster motion cannot be assessed; retry for a wider range. Not a diagnosis."
            )
        else:
            self.analysis_detail.setText("Image-plane motion only. A spectral peak is not a diagnosis.")
        self.info_label.setText(
            f"Samples: {results['samples']}  |  Rate: {results['fs']:.1f}/s  |  "
            f"Coverage: {results['coverage'] * 100:.0f}%  |  "
            f"Resolution: {results['resolution_hz']:.2f} Hz  |  "
            f"Max gap: {results['max_gap_sec']:.3f}s  |  Band: 1-{results['max_freq']:.1f} Hz"
        )

        set_badge(self.page_badge, "LIMITED BAND" if limited else "COMPLETE", AMBER if limited else GREEN)
        set_badge(self.analysis_badge, "LIMITED BAND" if limited else "COMPLETE", AMBER if limited else GREEN)
        if limited and self.status_bar:
            self.status_bar.update_badge("DSP", "LIMITED", AMBER)

        self._plot_psd(results)
        self._plot_waveform(results)
        self.protocol_panel.set_results(results)
        self.protocol_panel.show()

    def _plot_psd(self, r):
        ax = self.psd_canvas.ax
        ax.clear()
        self.psd_canvas._style_axes()

        freqs, psd = r['psd_freqs'], r['psd']
        mask = freqs <= r['max_freq']
        if np.any(psd > 0):
            ax.semilogy(freqs[mask], np.maximum(psd[mask], np.finfo(float).tiny), color=NAVY, linewidth=0.8)
        else:
            ax.plot(freqs[mask], psd[mask], color=NAVY, linewidth=0.8)

        dom = r['dominant_freq']
        if dom is not None:
            ax.axvline(dom, color=RED, linewidth=1, linestyle='--', alpha=0.8)

        ax.set_xlabel('Frequency (Hz)', fontsize=7, color=TEXT_DIM)
        ax.set_ylabel('PSD (px²/Hz)', fontsize=7, color=TEXT_DIM)
        ax.set_title('COMBINED X + Y PSD (WELCH)', fontsize=8,
                      fontweight='bold', color=TEXT_DIM, loc='left')
        ax.set_xlim(0, r['max_freq'])
        self.psd_canvas.draw()

    def _plot_waveform(self, r):
        ax = self.wave_canvas.ax
        ax.clear()
        self.wave_canvas._style_axes()

        t = r['time']
        raw = r['raw_y']
        filt = r['filtered_y']

        ax.plot(t, raw, color='#CBD5E1', linewidth=0.4, label='Y (detrended)')
        ax.plot(t, filt, color=NAVY, linewidth=0.7, label='Y filtered')
        ax.plot(t, r['filtered_x'], color='#2563EB', linewidth=0.7, label='X filtered')
        ax.axvspan(t[0], min(1.0, t[-1]), color='#CBD5E1', alpha=0.2)
        ax.axvspan(max(t[0], t[-1] - 1), t[-1], color='#CBD5E1', alpha=0.2)

        ax.set_xlabel('Time (s)', fontsize=7, color=TEXT_DIM)
        ax.set_ylabel('Displacement (px)', fontsize=7, color=TEXT_DIM)
        ax.set_title('MOTION WAVEFORM', fontsize=8, fontweight='bold', color=TEXT_DIM, loc='left')
        ax.legend(fontsize=6, loc='upper right', framealpha=0.7)
        self.wave_canvas.draw()

    def cleanup(self):
        self._closing = True
        self._stop(analyze=False)
        self.timer.stop()
        self.btn_start.setEnabled(False)
        self.btn_reset.setEnabled(False)
        return self.worker is None


class FacialPage(QWidget):
    capture_finished = pyqtSignal()

    def __init__(self, status_bar=None):
        super().__init__()
        self.status_bar = status_bar
        self.worker = None
        self.samples = []
        self.stage_bounds = {}
        self.stage_counts = {}
        self.stage_rejected = {}
        self.cancelled = False
        self.closing = False
        self.results = None
        self.preview_seen = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel("FACIAL SYMMETRY / FACE COMPONENT OF FAST")
        title.setFont(f_sans(13, True))
        title.setStyleSheet(f"color: {TEXT};")
        header.addWidget(title)
        header.addStretch()
        self.badge = make_status_badge("IDLE", TEXT_MUTED)
        header.addWidget(self.badge)
        layout.addLayout(header)
        sub = QLabel("Sudden facial droop, arm weakness or speech difficulty: seek emergency help immediately. Do not wait for this test.")
        sub.setFont(f_mono(9, True))
        sub.setStyleSheet(f"color: {RED};")
        sub.setWordWrap(True)
        layout.addWidget(sub)
        body = QHBoxLayout()
        body.setSpacing(8)
        left = QVBoxLayout()
        left.setSpacing(8)
        self.guidance = QLabel('One face, frontal pose, consistent distance. Follow the expression instructions here.')
        self.guidance.setWordWrap(True)
        self.guidance.setFont(f_mono(10, True))
        self.guidance.setTextFormat(Qt.TextFormat.PlainText)
        self.guidance.setStyleSheet(f'color: {NAVY}; background: {BG_ALT}; border: 1px solid {BORDER}; padding: 6px;')
        left.addWidget(self.guidance)
        self.video = VideoLabel('FACE MESH / REST -> SMILE -> RAISE EYEBROWS')
        self.video.setMinimumSize(480, 0)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setStyleSheet(f'background: {NAVY}; color: #94A3B8; border: 1px solid {BORDER};')
        self.capture_area = QWidget()
        self.capture_area.setMinimumSize(0, 0)
        self.capture_layout = QVBoxLayout(self.capture_area)
        self.capture_layout.setContentsMargins(0, 0, 0, 0)
        self.capture_layout.setSpacing(0)
        self.capture_layout.addWidget(self.video, 1)
        self.protocol_panel = FacialProtocolPanel()
        self.protocol_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.capture_layout.addWidget(self.protocol_panel, 1)
        left.addWidget(self.capture_area, 1)
        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.start = cmd_button('START FACIAL TEST')
        self.stop = cmd_button('STOP', primary=False)
        self.reset = cmd_button('RESET', primary=False)
        self.start.clicked.connect(self._start)
        self.stop.clicked.connect(self._stop)
        self.reset.clicked.connect(self._reset)
        self.stop.setEnabled(False)
        controls.addWidget(self.start)
        controls.addWidget(self.stop)
        controls.addWidget(self.reset)
        left.addLayout(controls)
        self.progress = thin_progress()
        left.addWidget(self.progress)
        body.addLayout(left, 3)
        right = QVBoxLayout()
        right.setSpacing(8)
        idx_panel, idx_lay = panel("Asymmetry Index")
        self.index = MetricDisplay('Normalized geometric error', '---', '%')
        idx_lay.addWidget(self.index)
        self.verdict = QLabel('Percentage of eye-corner span; not a probability of paresis or stroke.')
        self.verdict.setWordWrap(True)
        self.verdict.setTextFormat(Qt.TextFormat.PlainText)
        self.verdict.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.verdict.setFont(f_mono(8))
        self.verdict.setStyleSheet(f"color: {TEXT_DIM}; border: none;")
        idx_lay.addWidget(self.verdict)
        right.addWidget(idx_panel)
        det_panel, det_lay = panel("Bilateral Measurements")
        det_grid = QGridLayout()
        det_grid.setSpacing(2)
        self.mouth = MetricDisplay('Smile excursion imbalance', '---', '%')
        self.brow = MetricDisplay('Brow excursion imbalance', '---', '%')
        self.eye = MetricDisplay('Rest eye aperture R / L', '---')
        self.cheek = MetricDisplay('Cheek geometry error', '---', '%')
        det_grid.addWidget(self.mouth, 0, 0)
        det_grid.addWidget(self.eye, 0, 1)
        det_grid.addWidget(self.brow, 1, 0)
        det_grid.addWidget(self.cheek, 1, 1)
        det_lay.addLayout(det_grid)
        self.distances = QLabel('Axis distances R / L: awaiting capture')
        self.distances.setWordWrap(True)
        self.distances.setFont(f_mono(8))
        self.distances.setStyleSheet(f'color: {TEXT_DIM}; border: none;')
        det_lay.addWidget(self.distances)
        right.addWidget(det_panel)
        self.map = PlotCanvas()
        map_header = QHBoxLayout()
        map_header.addWidget(QLabel('DEVIATION MAP / EXPRESSION'))
        self.expression = QComboBox()
        self.expression.addItems(['Rest', 'Smile', 'Brows'])
        self.expression.currentIndexChanged.connect(self._expression_changed)
        self.expression.setEnabled(False)
        map_header.addWidget(self.expression)
        right.addLayout(map_header)
        right.addWidget(self.map, 1)
        disclaimer = QLabel('Research prototype only. Face observations are not the full FAST test and cannot diagnose or rule out stroke. Cheek landmarks do not measure nasolabial fold depth.')
        disclaimer.setWordWrap(True)
        disclaimer.setFont(f_mono(8))
        right.addWidget(disclaimer)
        body.addLayout(right, 2)
        layout.addLayout(body)

    def _start(self):
        if self.worker is not None or self.closing:
            return
        self._reset()
        self.worker = FaceCaptureWorker(self)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._capture_failed)
        self.worker.status.connect(self._capture_status)
        self.start.setEnabled(False)
        self.stop.setEnabled(True)
        set_badge(self.badge, 'STARTING', AMBER)
        self.guidance.setText('Starting camera and face model. First run may download the model.')
        if self.status_bar:
            self.status_bar.update_badge('MEDIAPIPE', 'STARTING', AMBER)
        self.timer.start(40)
        self.worker.start()

    def _stop(self):
        self.cancelled = True
        if self.worker is not None:
            self.worker.requestInterruption()
            self.stop.setEnabled(False)
            set_badge(self.badge, 'STOPPING', AMBER)

    def _reset(self):
        self._stop()
        self.protocol_panel.clear()
        self.samples = []
        self.stage_bounds = {}
        self.stage_counts = {}
        self.stage_rejected = {}
        self.results = None
        self.preview_seen = False
        self.expression.setEnabled(False)
        self.progress.setValue(0)
        for metric in (self.index, self.mouth, self.brow, self.eye, self.cheek):
            metric.set_value('---')
        self.map.clear_plot()
        self.video.setText('FACE MESH / REST -> SMILE -> RAISE EYEBROWS')
        self.distances.setText('Axis distances R / L: awaiting capture')
        self.guidance.setText('Face the camera directly and use a clear smile and brow raise.')
        self.verdict.setText('Percentage of eye-corner span; not a probability of paresis or stroke.')
        self.verdict.setStyleSheet(f'color: {TEXT_DIM}; border: none;')
        self.start.setEnabled(self.worker is None and not self.closing)
        self.stop.setEnabled(False)
        set_badge(self.badge, 'IDLE', TEXT_MUTED)
        if self.worker is None:
            self.cancelled = False

    def _poll(self):
        try:
            self._drain_frames()
        except Exception as exc:
            self._capture_failed(f'Face preview failed: {exc}')
            self.cancelled = True
            if self.worker is not None:
                self.worker.requestInterruption()

    def _capture_status(self, status):
        if not self.cancelled and self.worker is not None and not self.preview_seen:
            self.guidance.setText('Face capture setup: ' + status)

    def _capture_failed(self, message):
        if not self.cancelled:
            self.protocol_panel.clear()
            self.guidance.setText(message)
            self.verdict.setText('CAPTURE ERROR: ' + message)
            self.verdict.setStyleSheet(f'color: {RED}; border: none;')
            set_badge(self.badge, 'CAPTURE ERROR', RED)

    def _drain_frames(self):
        if self.worker is None:
            return
        latest, samples = self.worker.take_pending()
        if self.cancelled:
            return
        self.samples.extend(samples)
        if latest is not None:
            self.preview_seen = True
            self.stage_bounds = latest['stage_bounds']
            self.stage_counts = latest['counts']
            self.stage_rejected = latest['rejected']
            self.guidance.setText(latest['message'])
            self.progress.setValue(int(latest['progress'] * 100))
            set_badge(self.badge, latest['stage'].upper(), GREEN)
            image = latest['image']
            h, w = image.shape[:2]
            self.video.setPixmap(QPixmap.fromImage(QImage(image.data, w, h, image.strides[0], QImage.Format.Format_RGB888)))
            if self.status_bar:
                self.status_bar.update_badge('CAM', 'ACTIVE', GREEN)
                self.status_bar.update_badge('MEDIAPIPE', 'FACE LOADED', GREEN)

    def _finished(self):
        worker = self.worker
        if worker is None:
            return
        try:
            self._drain_frames()
            if not self.cancelled:
                if worker.error:
                    raise ValueError(worker.error)
                if worker.reason != 'deadline':
                    raise ValueError('Face capture did not complete.')
                stages = {}
                for stage in ('rest', 'smile', 'brows'):
                    selected = [sample for sample in self.samples if sample['stage'] == stage]
                    if stage not in self.stage_bounds:
                        raise ValueError('Missing expression: ' + stage)
                    count = self.stage_counts.get(stage, 0)
                    if count == 0 or self.stage_rejected.get(stage, 0) / count > 0.2:
                        rejected = self.stage_rejected.get(stage, 0)
                        raise ValueError(f'{stage.upper()}: {len(selected)} accepted / {count} frames, {rejected} rejected. Repeat with stable lighting and frontal pose.')
                    try:
                        stages[stage] = analyze_stage(selected, *self.stage_bounds[stage])
                    except ValueError as exc:
                        raise ValueError(stage.upper() + ': ' + str(exc)) from exc
                result = compare_stages(stages['rest'], stages['smile'], stages['brows'])
                self.results = result
                self.index.set_value(f"{result['asymmetry_index']:.2f}")
                self.mouth.set_value(f"{result['smile_imbalance']:.1f}")
                self.brow.set_value(f"{result['brow_imbalance']:.1f}")
                self.eye.set_value(f"{stages['rest']['eye'][0]:.2f} / {stages['rest']['eye'][1]:.2f}")
                self.cheek.set_value(f"{stages['rest']['regions']['cheeks']:.2f}")
                self.expression.setEnabled(True)
                self.expression.setCurrentIndex(1)
                self._expression_changed()
                self.progress.setValue(100)
                set_badge(self.badge, 'COMPLETE', GREEN)
                self.guidance.setText(
                    'Complete. Valid frames: ' + ' / '.join(f"{name}: {stages[name]['samples']}" for name in stages)
                    + '. Geometric percentages are not clinical risk scores.'
                )
                self.verdict.setText('Measurements complete. Percentage of eye-corner span; not a probability of paresis or stroke.')
                self.video.setText('RECORDING COMPLETE / SEE RESULTS')
                self.protocol_panel.set_results(result)
                self.protocol_panel.show()
            else:
                set_badge(self.badge, 'IDLE', TEXT_MUTED)
                self.video.setText('CAPTURE CANCELLED')
        except Exception as exc:
            self.protocol_panel.clear()
            self.results = None
            self.expression.setEnabled(False)
            self.map.clear_plot()
            self.distances.setText('No valid facial measurements available. Repeat capture.')
            for metric in (self.index, self.mouth, self.brow, self.eye, self.cheek):
                metric.set_value('---')
            set_badge(self.badge, 'RETRY', AMBER)
            self.guidance.setText(str(exc))
            self.verdict.setText('MEASUREMENT NOT AVAILABLE: ' + str(exc))
            self.verdict.setStyleSheet(f'color: {AMBER}; border: none;')
            self.video.setText('RECORDING ENDED / RETRY REQUIRED')
            self.map.ax.text(0.5, 0.5, 'No valid result: see reason above', ha='center', va='center',
                             transform=self.map.ax.transAxes, fontsize=9, color=TEXT_DIM)
            self.map.draw_idle()
            print(f'FACIAL measurement rejected: {exc}', file=sys.stderr)
        finally:
            self.timer.stop()
            self.worker = None
            worker.deleteLater()
            self.start.setEnabled(not self.closing)
            self.stop.setEnabled(False)
            if self.status_bar:
                self.status_bar.update_badge('CAM', 'DISCONNECTED', TEXT_MUTED)
                self.status_bar.update_badge('MEDIAPIPE', 'NOT LOADED', TEXT_MUTED)
            self.capture_finished.emit()

    def _plot_face(self, result):
        ax = self.map.ax
        ax.clear()
        points = result['points']
        ax.scatter(points[:468, 0], -points[:468, 1], s=2, color='#CBD5E1')
        selected = [i for pairs in PAIRS.values() for pair in pairs for i in pair]
        errors = [result['regions'][name] for name, pairs in PAIRS.items() for pair in pairs for _ in pair]
        ax.scatter(points[selected, 0], -points[selected, 1], c=errors, vmin=0, vmax=10, cmap='YlOrRd', s=22)
        slope, offset = result['axis']
        yy = np.array([points[:, 1].min(), points[:, 1].max()])
        ax.plot(slope * yy + offset, -yy, color=NAVY, linewidth=0.5)
        ax.set_aspect('equal')
        ax.set_title(self.expression.currentText().upper() + ' / ERROR: YELLOW 0% -> RED 10%+', fontsize=8)
        ax.set_xlabel('Normalized horizontal position; anatomical R at left', fontsize=7)
        self.map.draw_idle()

    def _expression_changed(self, index=None):
        if self.results is None:
            return
        key = ('rest', 'smile', 'brows')[self.expression.currentIndex()]
        self._plot_face(self.results[key])
        result = self.results[key]
        self.distances.setText(
            f"{key.upper()} axis distances (% eye span), anatomical R / L:\n"
            f"Mouth: {result['mouth'][0]*100:.2f} / {result['mouth'][1]*100:.2f}; "
            f"cheek: {result['cheek'][0]*100:.2f} / {result['cheek'][1]*100:.2f}; "
            f"brow elevation: {result['brow'][0]*100:.2f} / {result['brow'][1]*100:.2f}"
        )

    def cleanup(self):
        self.closing = True
        self._stop()
        self.start.setEnabled(False)
        self.reset.setEnabled(False)
        return self.worker is None


class RPPGPage(QWidget):
    capture_finished = pyqtSignal()

    def __init__(self, status_bar=None):
        super().__init__()
        self.status_bar = status_bar
        self.worker = None
        self.engine = RPPGEngine()
        self.cancelled = False
        self.closing = False
        self.frames = 0
        self.last_wave = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel("CONTACTLESS rPPG / HEART RATE ESTIMATE")
        title.setFont(f_sans(13, True))
        title.setStyleSheet(f"color: {TEXT};")
        header.addWidget(title)
        header.addStretch()
        self.badge = make_status_badge('IDLE', TEXT_MUTED)
        header.addWidget(self.badge)
        layout.addLayout(header)
        sub = QLabel("Green-channel skin modulation | 0.75-3.0 Hz / 45-180 BPM | 30-second trial | Research estimate, not a clinical monitor")
        sub.setFont(f_mono(8))
        sub.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(sub)
        body = QHBoxLayout()
        body.setSpacing(8)
        left = QVBoxLayout()
        left.setSpacing(8)
        self.guidance = QLabel('Sit still, keep the forehead uncovered, use steady diffuse light. Avoid speaking during capture.')
        self.guidance.setWordWrap(True)
        self.guidance.setFont(f_mono(9))
        left.addWidget(self.guidance)
        self.video = VideoLabel('SKIN ROI PREVIEW / FOREHEAD AND UPPER CHEEKS')
        self.video.setMinimumSize(480, 0)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setStyleSheet(f'background: {NAVY}; color: #94A3B8; border: 1px solid {BORDER};')
        self.capture_area = QWidget()
        self.capture_area.setMinimumSize(0, 0)
        self.capture_layout = QVBoxLayout(self.capture_area)
        self.capture_layout.setContentsMargins(0, 0, 0, 0)
        self.capture_layout.setSpacing(0)
        self.capture_layout.addWidget(self.video, 1)
        self.protocol_panel = RPPGProtocolPanel()
        self.protocol_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.capture_layout.addWidget(self.protocol_panel, 1)
        left.addWidget(self.capture_area, 1)
        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.start = cmd_button('START MEASUREMENT')
        self.stop = cmd_button('STOP', primary=False)
        self.reset = cmd_button('RESET', primary=False)
        self.start.clicked.connect(self._start)
        self.stop.clicked.connect(self._stop)
        self.reset.clicked.connect(self._reset)
        self.stop.setEnabled(False)
        controls.addWidget(self.start)
        controls.addWidget(self.stop)
        controls.addWidget(self.reset)
        left.addLayout(controls)
        self.progress = thin_progress()
        left.addWidget(self.progress)
        body.addLayout(left, 3)
        right = QVBoxLayout()
        right.setSpacing(8)
        bpm_panel, bpm_lay, self.analysis_badge = panel_with_badge("Heart Rate Estimate", "WAITING", TEXT_MUTED)
        self.bpm = MetricDisplay('Estimated pulse rate', '---', 'BPM')
        self.bpm.val.setFont(f_mono(30, True))
        bpm_lay.addWidget(self.bpm)
        qual_row = QHBoxLayout()
        self.concentration = MetricDisplay('Spectral concentration', '---', '%')
        self.coverage = MetricDisplay('Valid coverage', '---', '%')
        qual_row.addWidget(self.concentration)
        qual_row.addWidget(self.coverage)
        bpm_lay.addLayout(qual_row)
        self.detail = QLabel('No estimate yet. Timing and skin-region agreement are checked before BPM is displayed.')
        self.detail.setFont(f_mono(8))
        self.detail.setTextFormat(Qt.TextFormat.PlainText)
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet(f'color: {TEXT_DIM}; border: none;')
        bpm_lay.addWidget(self.detail)
        right.addWidget(bpm_panel)
        self.wave = PlotCanvas()
        self.spectrum = PlotCanvas()
        right.addWidget(self.wave, 1)
        right.addWidget(self.spectrum, 1)
        body.addLayout(right, 2)
        layout.addLayout(body)

    def _reset(self):
        self.protocol_panel.clear()
        if self.worker is not None:
            self._stop()
        else:
            self.cancelled = False
        self.engine.reset()
        self.frames = 0
        self.last_wave = 0
        self.progress.setValue(0)
        self.bpm.set_value('---')
        self.concentration.set_value('---')
        self.coverage.set_value('---')
        self.wave.clear_plot()
        self.spectrum.clear_plot()
        self.video.setText('SKIN ROI PREVIEW / FOREHEAD AND UPPER CHEEKS')
        self.detail.setText('Sit still under constant light. BPM appears only after a valid completed recording.')
        self.guidance.setText('Sit still, keep the forehead uncovered, use steady diffuse light. Avoid speaking during capture.')
        set_badge(self.badge, 'IDLE', TEXT_MUTED)
        set_badge(self.analysis_badge, 'WAITING', TEXT_MUTED)
        self.start.setEnabled(self.worker is None and not self.closing)
        self.stop.setEnabled(False)

    def _start(self):
        if self.worker is not None or self.closing:
            return
        self._reset()
        self.worker = RPPGCaptureWorker(self)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.status.connect(self._status)
        self.start.setEnabled(False)
        self.stop.setEnabled(True)
        set_badge(self.badge, 'STARTING', AMBER)
        self.guidance.setText('Starting the face model and camera; prepare for a 30-second resting capture.')
        if self.status_bar:
            self.status_bar.update_badge('DSP', 'STANDBY', TEXT_MUTED)
        self.timer.start(40)
        self.worker.start()

    def _status(self, text):
        if not self.cancelled and self.worker is not None and self.frames == 0:
            self.detail.setText('Setup: ' + text)

    def _failed(self, message):
        if not self.cancelled:
            self.protocol_panel.clear()
            self.detail.setText(message)
            set_badge(self.badge, 'CAPTURE ERROR', RED)

    def _stop(self):
        self.cancelled = True
        if self.worker is not None:
            self.worker.requestInterruption()
            self.stop.setEnabled(False)
            set_badge(self.badge, 'STOPPING', AMBER)

    def _poll(self):
        try:
            self._drain()
        except Exception as exc:
            self._failed(f'Preview failed: {exc}')
            self._stop()

    def _drain(self):
        if self.worker is None:
            return
        latest, samples = self.worker.take_pending()
        if self.cancelled:
            return
        for sample in samples:
            self.engine.add_sample(sample)
        if latest is not None:
            self.frames = latest['frames']
            self.guidance.setText(latest['message'])
            self.progress.setValue(int(latest['progress'] * 100))
            set_badge(self.badge, 'RECORDING' if latest['start'] is not None else 'PREPARING', GREEN)
            image = latest['image']
            h, w = image.shape[:2]
            self.video.setPixmap(QPixmap.fromImage(QImage(image.data, w, h, image.strides[0], QImage.Format.Format_RGB888)))
            if self.status_bar:
                self.status_bar.update_badge('CAM', 'ACTIVE', GREEN)
                self.status_bar.update_badge('MEDIAPIPE', 'FACE LOADED', GREEN)
        if len(self.engine.samples) > 10 and time.perf_counter() - self.last_wave > 0.7:
            self.last_wave = time.perf_counter()
            t = np.array([sample['timestamp'] for sample in self.engine.samples])
            green = np.array([sample['rgb'][:, 1].mean() for sample in self.engine.samples])
            self.wave.clear_plot()
            self.wave.ax.plot(t - t[0], green, color=NAVY, linewidth=0.7)
            self.wave.ax.set_title('RAW MEAN GREEN / NOT YET A PULSE ESTIMATE', fontsize=8)
            self.wave.ax.set_xlabel('Time (s)', fontsize=7)
            self.wave.ax.set_ylabel('Intensity (0-255)', fontsize=7)
            self.wave.draw_idle()

    def _finished(self):
        worker = self.worker
        if worker is None:
            return
        try:
            self._drain()
            if not self.cancelled:
                if worker.error:
                    raise ValueError(worker.error)
                if worker.reason != 'deadline':
                    raise ValueError('Capture ended before the full rPPG trial completed.')
                start, end, _, _ = worker.snapshot()
                result = self.engine.analyze(start, end, self.frames)
                if result is None:
                    raise ValueError(self.engine.last_error)
                self._render(result)
            else:
                set_badge(self.badge, 'IDLE', TEXT_MUTED)
                self.video.setText('MEASUREMENT CANCELLED')
        except Exception as exc:
            self.bpm.set_value('---')
            self.protocol_panel.clear()
            self.concentration.set_value('---')
            self.coverage.set_value('---')
            self.detail.setText(str(exc))
            self.spectrum.clear_plot()
            self.video.setText('MEASUREMENT ENDED / RETRY REQUIRED')
            set_badge(self.badge, 'RETRY', AMBER)
            set_badge(self.analysis_badge, 'REJECTED', AMBER)
            self.guidance.setText('Measurement ended. See the rejection reason in the Heart Rate Estimate panel.')
            if self.status_bar:
                self.status_bar.update_badge('DSP', 'REJECTED', AMBER)
        finally:
            self.timer.stop()
            self.worker = None
            worker.deleteLater()
            self.start.setEnabled(not self.closing)
            self.stop.setEnabled(False)
            if self.status_bar:
                self.status_bar.update_badge('CAM', 'DISCONNECTED', TEXT_MUTED)
                self.status_bar.update_badge('MEDIAPIPE', 'NOT LOADED', TEXT_MUTED)
            self.capture_finished.emit()

    def _render(self, result):
        warnings = result['warnings']
        self.bpm.set_value(f"{result['bpm']:.0f}")
        self.concentration.set_value(f"{result['concentration']:.1f}")
        self.coverage.set_value(f"{result['coverage'] * 100:.0f}")
        self.detail.setText(
            f"{result['samples']} samples, {result['rate']:.1f}/s, FFT bin spacing {result['resolution_bpm']:.1f} BPM. "
            f"ROI estimates: {', '.join(f'{value:.0f}' for value in result['roi_bpm'])} BPM. "
            + (' '.join(warnings) + ' ' if warnings else '') +
            'Research estimate only; shared lighting artifacts cannot always be excluded. No SpO2 measurement.'
        )
        self.wave.clear_plot()
        self.wave.ax.plot(result['time'], result['pulse'], color=NAVY, linewidth=0.7)
        self.wave.ax.set_title('FILTERED GREEN MODULATION / 0.75-3.0 Hz', fontsize=8)
        self.wave.ax.set_xlabel('Time (s)', fontsize=7)
        self.wave.ax.set_ylabel('Fractional intensity (%)', fontsize=7)
        self.wave.draw_idle()
        self.spectrum.clear_plot()
        self.spectrum.ax.plot(result['frequencies'] * 60, result['spectrum'], color=NAVY, linewidth=0.8)
        self.spectrum.ax.axvline(result['bpm'], color=RED, linestyle='--', linewidth=1)
        self.spectrum.ax.set_xlim(45, 180)
        self.spectrum.ax.set_title('NORMALIZED FFT POWER / PULSE CANDIDATE', fontsize=8)
        self.spectrum.ax.set_xlabel('Pulse rate (BPM)', fontsize=7)
        self.spectrum.ax.set_ylabel('Normalized power', fontsize=7)
        self.spectrum.draw_idle()
        self.progress.setValue(100)
        self.video.setText('RECORDING COMPLETE / SEE ESTIMATE')
        set_badge(self.badge, 'CHECK QUALITY' if warnings else 'COMPLETE', AMBER if warnings else GREEN)
        set_badge(self.analysis_badge, 'ESTIMATE / CAUTION' if warnings else 'ACCEPTED', AMBER if warnings else GREEN)
        self.guidance.setText('Measurement complete. A camera-derived pulse estimate needs validation against a reference sensor.')
        if self.status_bar:
            self.status_bar.update_badge('DSP', 'CAUTION' if warnings else 'COMPLETE', AMBER if warnings else GREEN)
        self.protocol_panel.set_results(result)
        self.protocol_panel.show()

    def cleanup(self):
        self.closing = True
        self._stop()
        self.start.setEnabled(False)
        self.reset.setEnabled(False)
        return self.worker is None


class AACPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel("EYE-TRACKING AAC INTERFACE")
        title.setFont(f_sans(13, True))
        title.setStyleSheet(f"color: {TEXT};")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(make_status_badge("PLANNED", TEXT_MUTED))
        layout.addLayout(header)
        sub = QLabel("Interface preview only  |  Gaze tracking and communication are not implemented")
        sub.setFont(f_mono(8))
        sub.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(sub)
        top = QHBoxLayout()
        top.setSpacing(8)
        tracking_panel, tracking_lay = panel("Eye Tracking Feed")
        tracking_feed = QFrame()
        tracking_feed.setMinimumSize(280, 180)
        tracking_feed.setStyleSheet(f"background: {NAVY}; border: 1px solid {BORDER};")
        tf_lay = QVBoxLayout(tracking_feed)
        tf_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tf_lbl = QLabel("IRIS TRACKING FEED")
        tf_lbl.setFont(f_mono(9))
        tf_lbl.setStyleSheet("color: #475569; border: none;")
        tf_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tf_lay.addWidget(tf_lbl)
        tracking_lay.addWidget(tracking_feed)
        gaze_row = QHBoxLayout()
        gaze_row.addWidget(metric_readout("Gaze X", "---"))
        gaze_row.addWidget(metric_readout("Gaze Y", "---"))
        gaze_row.addWidget(metric_readout("Dwell", "---", "ms"))
        tracking_lay.addLayout(gaze_row)
        top.addWidget(tracking_panel, 2)
        output_panel, output_lay = panel("Output Buffer")
        selected = QLabel("---")
        selected.setFont(f_mono(20, True))
        selected.setStyleSheet(f"color: {NAVY}; border: none;")
        selected.setAlignment(Qt.AlignmentFlag.AlignCenter)
        output_lay.addWidget(selected)
        output_lay.addStretch()
        btn_row = QHBoxLayout()
        btn_row.addWidget(planned_button("SPEAK"))
        btn_row.addWidget(planned_button("CLEAR", primary=False))
        output_lay.addLayout(btn_row)
        top.addWidget(output_panel, 1)
        layout.addLayout(top)
        grid_panel, grid_lay = panel("Communication Grid")
        grid = QGridLayout()
        grid.setSpacing(4)
        for text, r, c in [
            ("WATER", 0, 0), ("DOCTOR", 0, 1), ("PAIN", 0, 2),
            ("YES", 1, 0), ("NO", 1, 1), ("HELP", 1, 2),
        ]:
            btn = QPushButton(text)
            btn.setFont(f_sans(12, True))
            btn.setFixedHeight(80)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setEnabled(False)
            btn.setStyleSheet(f"""
                QPushButton {{ background: {BG_ALT}; color: {NAVY}; border: 1px solid {BORDER}; padding: 0 8px; }}
                QPushButton:hover {{ background: {NAVY}; color: #FFFFFF; }}
            """)
            grid.addWidget(btn, r, c)
        grid_lay.addLayout(grid)
        nav_row = QHBoxLayout()
        nav_row.addWidget(planned_button("< PREV CATEGORY", primary=False))
        nav_row.addStretch()
        cat_label = QLabel("CATEGORY 1 / 3: BASIC NEEDS")
        cat_label.setFont(f_mono(9, True))
        cat_label.setStyleSheet(f"color: {TEXT_DIM}; border: none;")
        nav_row.addWidget(cat_label)
        nav_row.addStretch()
        nav_row.addWidget(planned_button("NEXT CATEGORY >", primary=False))
        grid_lay.addLayout(nav_row)
        layout.addWidget(grid_panel)
        layout.addStretch()


class VoicePage(QWidget):
    capture_finished = pyqtSignal()

    def __init__(self, status_bar=None):
        super().__init__()
        self.status_bar = status_bar
        self.worker = None
        self.cancelled = False
        self.closing = False
        self.result = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel("VOICE BIOMARKERS / ACOUSTIC ANALYSIS")
        title.setFont(f_sans(13, True))
        title.setStyleSheet(f"color: {TEXT};")
        header.addWidget(title)
        header.addStretch()
        self.badge = make_status_badge('IDLE', TEXT_MUTED)
        header.addWidget(self.badge)
        layout.addLayout(header)
        sub = QLabel("Sustained vowel /a/ | 5 seconds | Praat perturbation metrics + librosa | Research use only")
        sub.setFont(f_mono(8))
        sub.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(sub)
        top = QHBoxLayout()
        top.setSpacing(8)
        rec_panel, rec_lay, _ = panel_with_badge("Recording", "МИКРОФОН", TEXT_MUTED)
        self.time_label = QLabel("0.0 / 5.0")
        self.time_label.setFont(f_mono(28, True))
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet(f"color: {TEXT}; border: none;")
        rec_lay.addWidget(self.time_label)
        timer_unit = QLabel("seconds")
        timer_unit.setFont(f_mono(9))
        timer_unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        timer_unit.setStyleSheet(f"color: {TEXT_MUTED}; border: none;")
        rec_lay.addWidget(timer_unit)
        self.progress = thin_progress()
        rec_lay.addWidget(self.progress)
        self.level = MetricDisplay('Peak input level', '---', 'dBFS')
        rec_lay.addWidget(self.level)
        self.guidance = QLabel('Use a quiet room. Sustain Ah at a comfortable, steady volume and pitch for five seconds. Keep microphone distance fixed.')
        self.guidance.setWordWrap(True)
        self.guidance.setStyleSheet(f'color: {TEXT_DIM}; border: none;')
        rec_lay.addWidget(self.guidance)
        controls = QHBoxLayout()
        self.start = cmd_button('START RECORDING')
        self.start.setToolTip('Новая запись с микрофона. Скажите протяжный звук «А» после начала отсчёта.')
        self.upload = cmd_button('UPLOAD AUDIO FILE', primary=False)
        self.stop = cmd_button('STOP', primary=False)
        self.reset = cmd_button('RESET', primary=False)
        self.start.clicked.connect(self._start)
        self.upload.clicked.connect(self._upload)
        self.stop.clicked.connect(self._stop)
        self.reset.clicked.connect(self._reset)
        self.stop.setEnabled(False)
        for button in (self.start, self.upload, self.stop, self.reset):
            controls.addWidget(button)
        rec_lay.addLayout(controls)
        top.addWidget(rec_panel, 1)
        met_panel, met_lay = panel("Acoustic Metrics")
        met_grid = QGridLayout()
        met_grid.setSpacing(2)
        self.f0 = MetricDisplay('Median F0', '---', 'Hz')
        self.jitter = MetricDisplay('Jitter (local)', '---', '%')
        self.shimmer = MetricDisplay('Shimmer (local)', '---', '%')
        self.hnr = MetricDisplay('HNR (autocorrelation)', '---', 'dB')
        met_grid.addWidget(self.f0, 0, 0)
        met_grid.addWidget(self.jitter, 0, 1)
        met_grid.addWidget(self.shimmer, 1, 0)
        met_grid.addWidget(self.hnr, 1, 1)
        met_lay.addLayout(met_grid)
        self.detail = QLabel('No result yet. A sustained vowel does not assess speech articulation or diagnose dysarthria.')
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.detail.setStyleSheet(f'color: {TEXT_DIM}; border: none;')
        met_lay.addWidget(self.detail)
        top.addWidget(met_panel, 1)
        layout.addLayout(top)
        graphs = QHBoxLayout()
        graphs.setSpacing(8)
        self.wave = PlotCanvas()
        self.pitch_plot = PlotCanvas()
        self.spectrogram = PlotCanvas()
        for plot in (self.wave, self.pitch_plot, self.spectrogram):
            graphs.addWidget(plot, 1)
        layout.addLayout(graphs, 1)
        self.protocol_panel = VoiceProtocolPanel()
        layout.addWidget(self.protocol_panel, 1)

    def _reset(self):
        if self.worker is not None:
            self._stop()
        else:
            self.cancelled = False
        self.result = None
        self.protocol_panel.clear()
        self.time_label.setText('0.0 / 5.0')
        self.progress.setValue(0)
        for metric in (self.f0, self.jitter, self.shimmer, self.hnr, self.level):
            metric.set_value('---')
        for plot in (self.wave, self.pitch_plot, self.spectrogram):
            plot.clear_plot()
        self.detail.setText('No result yet. A sustained vowel does not assess speech articulation or diagnose dysarthria.')
        self.guidance.setText('Use a quiet room. Sustain Ah at a comfortable, steady volume and pitch for five seconds. Keep microphone distance fixed.')
        set_badge(self.badge, 'IDLE', TEXT_MUTED)
        self.start.setEnabled(self.worker is None and not self.closing)
        self.upload.setEnabled(self.worker is None and not self.closing)
        self.stop.setEnabled(False)

    def _start(self):
        if self.worker is not None or self.closing or not self._check_dependencies():
            return
        self._begin(VoiceCaptureWorker)

    def _check_dependencies(self):
        try:
            check_voice_dependencies()
            return True
        except RuntimeError as exc:
            self.detail.setText(str(exc))
            self.guidance.setText('Не хватает библиотек для анализа голоса. Установите зависимости указанной командой; запись пока не начата.')
            set_badge(self.badge, 'DEPENDENCY ERROR', RED)
            return False

    def _upload(self):
        if self.worker is not None or self.closing or not self._check_dependencies():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, 'Выберите аудиофайл с протяжным звуком «А»', '',
            'Аудиофайлы (*.wav *.flac *.ogg *.mp3 *.m4a *.aac *.aiff *.aif);;Все файлы (*.*)'
        )
        if not path:
            return
        self._begin(VoiceFileCaptureWorker, path)

    def _stop(self):
        self.cancelled = True
        if self.worker is not None:
            self.worker.requestInterruption()
            self.stop.setEnabled(False)
            set_badge(self.badge, 'STOPPING', AMBER)

    def _begin(self, worker_type, *args):
        self._reset()
        self.cancelled = False
        self.result = None
        self.protocol_panel.clear()
        self.start.setEnabled(False)
        self.upload.setEnabled(False)
        self.stop.setEnabled(True)
        set_badge(self.badge, 'STARTING', AMBER)
        if self.status_bar:
            self.status_bar.update_badge('MIC', 'STARTING', AMBER)
        self.worker = worker_type(*args, parent=self)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.timer.start(40)
        self.worker.start()

    def _failed(self, message):
        if not self.cancelled:
            self.detail.setText(message)
            set_badge(self.badge, 'RETRY', AMBER)

    def _poll(self):
        try:
            self._drain()
        except Exception as exc:
            self._failed(str(exc))
            self._stop()

    def _drain(self):
        if self.worker is None:
            return
        latest, _ = self.worker.take_pending()
        if latest is None or self.cancelled:
            return
        self.time_label.setText(f"{latest['elapsed']:.1f} / 5.0")
        self.progress.setValue(int(latest['elapsed'] * 20))
        self.guidance.setText(latest['message'])
        set_badge(self.badge, latest['state'], GREEN if latest['state'] == 'RECORDING' else AMBER)
        if latest['level'] > 0:
            self.level.set_value(f"{20 * np.log10(latest['level']):.1f}")
        if 'result' in latest:
            self.result = latest['result']
        if self.status_bar:
            self.status_bar.update_badge('MIC', 'ACTIVE' if latest['state'] == 'RECORDING' else 'STANDBY', GREEN)

    def _finished(self):
        worker = self.worker
        if worker is None:
            return
        try:
            self._drain()
            if not self.cancelled:
                if worker.error:
                    raise ValueError(worker.error)
                if worker.reason != 'deadline' or self.result is None:
                    raise ValueError('Voice recording or analysis did not complete.')
                self._render(self.result)
            else:
                set_badge(self.badge, 'IDLE', TEXT_MUTED)
        except Exception as exc:
            self.result = None
            self.protocol_panel.clear()
            for metric in (self.f0, self.jitter, self.shimmer, self.hnr):
                metric.set_value('---')
            self.detail.setText(str(exc))
            self.guidance.setText('Запись или анализ не завершены. Причина и рекомендации указаны в панели акустических показателей справа.')
            print(f'VOICE recording/analysis failed: {exc}', file=sys.stderr)
            set_badge(self.badge, 'RETRY', AMBER)
        finally:
            self.timer.stop()
            self.worker = None
            worker.deleteLater()
            self.start.setEnabled(not self.closing)
            self.upload.setEnabled(not self.closing)
            self.stop.setEnabled(False)
            if self.status_bar:
                self.status_bar.update_badge('MIC', 'DISCONNECTED', TEXT_MUTED)
            self.capture_finished.emit()

    def _render(self, result):
        for metric, key, digits in ((self.f0, 'f0', 1), (self.jitter, 'jitter_local', 3),
                                    (self.shimmer, 'shimmer_local', 3), (self.hnr, 'hnr', 1)):
            metric.set_value(f"{result[key]:.{digits}f}")
        self.detail.setText(
            f"Voiced frames: {result['voiced_fraction']*100:.1f}%; level: {result['level_dbfs']:.1f} dBFS. "
            f"Jitter RAP: {result['jitter_rap']:.3f}%; shimmer APQ5: {result['shimmer_apq5']:.3f}%. "
            + ' '.join(result['warnings'])
        )
        if result.get('source') == 'file':
            self.detail.setText(
                f"Файл: {result.get('file_name', '')}; выбранный фрагмент: {result.get('segment_start_sec', 0):.2f} с; "
                + self.detail.text()
            )
        for plot in (self.wave, self.pitch_plot, self.spectrogram):
            plot.clear_plot()
        self.wave.ax.plot(result['wave_time'], result['wave'], color=NAVY, linewidth=0.4)
        self.wave.ax.set_title('VOICE WAVEFORM', fontsize=8)
        self.wave.ax.set_xlabel('Time (s)', fontsize=7)
        self.pitch_plot.ax.plot(result['pitch_time'], result['pitch'], color=NAVY, linewidth=0.8)
        self.pitch_plot.ax.set_title('F0 / PERIODIC VOICING', fontsize=8)
        self.pitch_plot.ax.set_ylabel('Frequency (Hz)', fontsize=7)
        self.spectrogram.ax.pcolormesh(result['spectrogram_time'], result['spectrogram_freq'], result['spectrogram'],
                                      shading='auto', cmap='magma', vmin=-80, vmax=0)
        self.spectrogram.ax.set_ylim(0, 5000)
        self.spectrogram.ax.set_title('STFT / RELATIVE dB', fontsize=8)
        self.spectrogram.ax.set_xlabel('Time (s)', fontsize=7)
        for plot in (self.wave, self.pitch_plot, self.spectrogram):
            plot.draw_idle()
        self.protocol_panel.set_results(result)
        self.protocol_panel.show()
        self.guidance.setText('Recording and acoustic analysis complete. See the reference guidance below.')
        set_badge(self.badge, 'COMPLETE', GREEN)

    def cleanup(self):
        self.closing = True
        self._stop()
        self.start.setEnabled(False)
        self.reset.setEnabled(False)
        return self.worker is None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TecnoMedNeuro — Исследовательский прототип веб-измерений")
        self.setMinimumSize(1200, 750)
        self.resize(1400, 860)
        self.setStyleSheet(f"QMainWindow {{ background: {BG}; }} * {{ border-radius: 0px; }}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        tabs = ["ОБЗОР", "ТРЕМОР", "ЛИЦО", "ПУЛЬС", "КОММУНИКАЦИЯ", "ГОЛОС"]
        self.nav = NavBar(tabs, self.switch_page)
        root.addWidget(self.nav)

        self.status_bar = StatusBar()

        self.stack = QStackedWidget()
        self.tremor_page = TremorPage(status_bar=self.status_bar)
        self._close_pending = False
        self.tremor_page.capture_finished.connect(self._finish_close)
        self.stack.addWidget(DashboardPage())
        self.stack.addWidget(self.tremor_page)
        self.facial_page = FacialPage(status_bar=self.status_bar)
        self.facial_page.capture_finished.connect(self._finish_close)
        self.stack.addWidget(self.facial_page)
        self.rppg_page = RPPGPage(status_bar=self.status_bar)
        self.rppg_page.capture_finished.connect(self._finish_close)
        self.stack.addWidget(self.rppg_page)
        self.stack.addWidget(AACPage())
        self.voice_page = VoicePage(status_bar=self.status_bar)
        self.voice_page.capture_finished.connect(self._finish_close)
        self.stack.addWidget(self.voice_page)
        root.addWidget(self.stack)

        root.addWidget(self.status_bar)
        self.switch_page(0)

    def switch_page(self, index):
        if hasattr(self, 'voice_page') and index in (1, 2, 3, 5):
            pages = {1: self.tremor_page, 2: self.facial_page, 3: self.rppg_page, 5: self.voice_page}
            if any(page.worker is not None and key != index for key, page in pages.items()):
                return
        self.stack.setCurrentIndex(index)
        self.nav.set_active(index)

    def closeEvent(self, event):
        tremor_done = self.tremor_page.cleanup()
        face_done = self.facial_page.cleanup()
        rppg_done = self.rppg_page.cleanup()
        voice_done = self.voice_page.cleanup()
        if tremor_done and face_done and rppg_done and voice_done:
            event.accept()
        else:
            self._close_pending = True
            event.ignore()

    def _finish_close(self):
        if self._close_pending:
            QTimer.singleShot(0, self.close)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
