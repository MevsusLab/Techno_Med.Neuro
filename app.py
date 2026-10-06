import sys
import time
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFrame, QGridLayout,
    QSizePolicy, QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QImage, QPixmap
from tremor_engine import TremorEngine
from capture_worker import CaptureWorker


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
    lbl = QLabel(text)
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
    btn = QPushButton(text)
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
            btn = QPushButton(label)
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
            ("Modules", "1 prototype / 4 planned"),
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
            ("Facial Asymmetry", "Research", "PLANNED"),
            ("rPPG Vitals", "Monitoring", "PLANNED"),
            ("Eye-Tracking AAC", "Assistive", "PLANNED"),
            ("Voice Biomarkers", "Research", "PLANNED"),
        ]):
            table.setItem(r, 0, QTableWidgetItem(name))
            table.setItem(r, 1, QTableWidgetItem(typ))
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
            ("FACIAL", "Planned: bilateral facial landmark measurements"),
            ("VOICE", "Planned: acoustic features from sustained phonation"),
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
            ("rPPG", "Planned: camera-based pulse-rate research prototype"),
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

        self.video_label = QLabel("HAND TRACKING FEED\n\n640x480 @ 30fps")
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFont(f_mono(10))
        self.video_label.setStyleSheet(f"background: {NAVY}; border: 1px solid {BORDER}; color: #475569;")
        left.addWidget(self.video_label)

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
                self.btn_stop.setEnabled(False)
                self.worker.requestInterruption()
                set_badge(self.page_badge, "STOPPING", AMBER)
        if latest is not None:
            display = latest[0]
            h, w = display.shape[:2]
            qimg = QImage(display.data, w, h, display.strides[0], QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pixmap.scaled(
                self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation))

    def _on_finished(self):
        worker = self.worker
        if worker is None:
            return
        if worker.error:
            self._on_failure(worker.error)
        elif worker.reason == "stopped" and self._stop_time is None and not self._closing:
            self._analyze_on_finish = False
            self._failure = "Capture ended before the recording was complete"
            set_badge(self.page_badge, "CAPTURE STOPPED", AMBER)
            set_badge(self.analysis_badge, "INCOMPLETE", AMBER)
            self.analysis_detail.setText(self._failure)
        self._on_frame()
        self.timer.stop()
        self.recording = False
        self.worker = None
        worker.deleteLater()
        self.btn_start.setEnabled(not self._closing)
        self.btn_stop.setEnabled(False)
        if self.status_bar:
            self.status_bar.update_badge("CAM", "DISCONNECTED", TEXT_MUTED)
            self.status_bar.update_badge("MEDIAPIPE", "NOT LOADED", TEXT_MUTED)
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
        self.capture_finished.emit()

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
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel("FACIAL ASYMMETRY / PLANNED")
        title.setFont(f_sans(13, True))
        title.setStyleSheet(f"color: {TEXT};")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(make_status_badge("PLANNED", TEXT_MUTED))
        layout.addLayout(header)
        sub = QLabel("Interface preview only  |  No face tracking or assessment is implemented")
        sub.setFont(f_mono(8))
        sub.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(sub)
        body = QHBoxLayout()
        body.setSpacing(8)
        left = QVBoxLayout()
        left.setSpacing(8)
        left.addWidget(feed_panel("FACE MESH OVERLAY"))
        controls = QHBoxLayout()
        controls.setSpacing(6)
        controls.addWidget(planned_button("CAPTURE RESTING"))
        controls.addWidget(planned_button("CAPTURE SMILE", primary=False))
        controls.addWidget(planned_button("RESET", primary=False))
        left.addLayout(controls)
        body.addLayout(left, 3)
        right = QVBoxLayout()
        right.setSpacing(8)
        idx_panel, idx_lay = panel("Asymmetry Index")
        idx_lay.addWidget(metric_readout_large("Overall Asymmetry", "---", "%", NAVY))
        verdict = QLabel("Awaiting capture")
        verdict.setFont(f_mono(9))
        verdict.setStyleSheet(f"color: {TEXT_MUTED}; border: none;")
        idx_lay.addWidget(verdict)
        right.addWidget(idx_panel)
        det_panel, det_lay = panel("Bilateral Distances")
        det_grid = QGridLayout()
        det_grid.setSpacing(2)
        det_grid.addWidget(metric_readout("Mouth dL / dR", "--- / ---", "px"), 0, 0)
        det_grid.addWidget(metric_readout("Eye Aperture L/R", "--- / ---", "px"), 0, 1)
        det_grid.addWidget(metric_readout("Brow Elevation L/R", "--- / ---", "px"), 1, 0)
        det_grid.addWidget(metric_readout("Nasolabial L/R", "--- / ---", "px"), 1, 1)
        det_lay.addLayout(det_grid)
        right.addWidget(det_panel)
        right.addWidget(graph_area("DEVIATION HEATMAP"))
        body.addLayout(right, 2)
        layout.addLayout(body)


class RPPGPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel("CONTACTLESS rPPG / PLANNED")
        title.setFont(f_sans(13, True))
        title.setStyleSheet(f"color: {TEXT};")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(make_status_badge("PLANNED", TEXT_MUTED))
        layout.addLayout(header)
        sub = QLabel("Interface preview only  |  No pulse measurement is implemented")
        sub.setFont(f_mono(8))
        sub.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(sub)
        body = QHBoxLayout()
        body.setSpacing(8)
        left = QVBoxLayout()
        left.setSpacing(8)
        left.addWidget(feed_panel("ROI TRACKING (FOREHEAD + CHEEKS)"))
        controls = QHBoxLayout()
        controls.setSpacing(6)
        controls.addWidget(planned_button("START MEASUREMENT"))
        controls.addWidget(planned_button("STOP", primary=False))
        left.addLayout(controls)
        left.addWidget(thin_progress())
        body.addLayout(left, 3)
        right = QVBoxLayout()
        right.setSpacing(8)
        bpm_panel, bpm_lay, _ = panel_with_badge("Heart Rate", "PLANNED", TEXT_MUTED)
        bpm_lay.addWidget(metric_readout_large("BPM", "---", "bpm", NAVY))
        qual_row = QHBoxLayout()
        qual_row.addWidget(metric_readout("Signal Quality", "---"))
        bpm_lay.addLayout(qual_row)
        right.addWidget(bpm_panel)
        right.addWidget(graph_area("PULSE WAVEFORM (GREEN CHANNEL)"))
        right.addWidget(graph_area("FFT SPECTRUM / BPM PEAK"))
        body.addLayout(right, 2)
        layout.addLayout(body)


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
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel("VOICE BIOMARKERS / PLANNED")
        title.setFont(f_sans(13, True))
        title.setStyleSheet(f"color: {TEXT};")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(make_status_badge("PLANNED", TEXT_MUTED))
        layout.addLayout(header)
        sub = QLabel("Interface preview only  |  No audio recording or voice assessment is implemented")
        sub.setFont(f_mono(8))
        sub.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(sub)
        top = QHBoxLayout()
        top.setSpacing(8)
        rec_panel, rec_lay, _ = panel_with_badge("Recording", "PLANNED", TEXT_MUTED)
        timer_val = QLabel("0.0 / 5.0")
        timer_val.setFont(f_mono(28, True))
        timer_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        timer_val.setStyleSheet(f"color: {TEXT}; border: none;")
        rec_lay.addWidget(timer_val)
        timer_unit = QLabel("seconds")
        timer_unit.setFont(f_mono(9))
        timer_unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        timer_unit.setStyleSheet(f"color: {TEXT_MUTED}; border: none;")
        rec_lay.addWidget(timer_unit)
        rec_lay.addWidget(thin_progress())
        rec_lay.addStretch()
        rec_lay.addWidget(planned_button("START RECORDING"))
        top.addWidget(rec_panel, 1)
        met_panel, met_lay = panel("Acoustic Metrics")
        met_grid = QGridLayout()
        met_grid.setSpacing(2)
        met_grid.addWidget(metric_readout("F0 (Fundamental)", "---", "Hz"), 0, 0)
        met_grid.addWidget(metric_readout("Jitter (RAP)", "---", "%"), 0, 1)
        met_grid.addWidget(metric_readout("Shimmer (APQ)", "---", "%"), 1, 0)
        met_grid.addWidget(metric_readout("HNR", "---", "dB"), 1, 1)
        met_lay.addLayout(met_grid)
        top.addWidget(met_panel, 1)
        layout.addLayout(top)
        graphs = QHBoxLayout()
        graphs.setSpacing(8)
        graphs.addWidget(graph_area("AUDIO WAVEFORM"))
        graphs.addWidget(graph_area("FREQUENCY SPECTRUM (FFT)"))
        graphs.addWidget(graph_area("MFCC COEFFICIENTS"))
        layout.addLayout(graphs)
        result_panel, result_lay = panel("Assessment")
        r_row = QHBoxLayout()
        r_lbl = QLabel("Vocal stability assessment will be computed after recording")
        r_lbl.setFont(f_mono(9))
        r_lbl.setStyleSheet(f"color: {TEXT_MUTED}; border: none;")
        r_row.addWidget(r_lbl)
        r_row.addStretch()
        r_row.addWidget(planned_button("EXPORT REPORT", primary=False))
        result_lay.addLayout(r_row)
        layout.addWidget(result_panel)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TecnoMedNeuro - Hand Motion Research Prototype")
        self.setMinimumSize(1200, 750)
        self.resize(1400, 860)
        self.setStyleSheet(f"QMainWindow {{ background: {BG}; }} * {{ border-radius: 0px; }}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        tabs = ["DASHBOARD", "TREMOR", "FACIAL", "rPPG", "AAC", "VOICE"]
        self.nav = NavBar(tabs, self.switch_page)
        root.addWidget(self.nav)

        self.status_bar = StatusBar()

        self.stack = QStackedWidget()
        self.tremor_page = TremorPage(status_bar=self.status_bar)
        self._close_pending = False
        self.tremor_page.capture_finished.connect(self._finish_close)
        self.stack.addWidget(DashboardPage())
        self.stack.addWidget(self.tremor_page)
        self.stack.addWidget(FacialPage())
        self.stack.addWidget(RPPGPage())
        self.stack.addWidget(AACPage())
        self.stack.addWidget(VoicePage())
        root.addWidget(self.stack)

        root.addWidget(self.status_bar)
        self.switch_page(0)

    def switch_page(self, index):
        self.stack.setCurrentIndex(index)
        self.nav.set_active(index)

    def closeEvent(self, event):
        if self.tremor_page.cleanup():
            event.accept()
        else:
            self._close_pending = True
            event.ignore()

    def _finish_close(self):
        if self._close_pending:
            QTimer.singleShot(0, self.close)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
