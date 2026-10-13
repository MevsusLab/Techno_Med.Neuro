import sys
import os
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
    QFileDialog, QPlainTextEdit
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt6.QtGui import QFont, QColor, QImage, QPixmap
from PyQt6.QtWidgets import QLabel as PlainLabel
from tremor_engine import TremorEngine
from capture_worker import ProcessCaptureWorker as CaptureWorker
from protocol_panel import ProtocolPanel, FacialProtocolPanel, RPPGProtocolPanel, VoiceProtocolPanel
from face_capture import FaceCaptureWorker
from facial_engine import analyze_stage, compare_stages, PAIRS
from rppg_capture import RPPGCaptureWorker
from rppg_engine import RPPGEngine
from voice_capture import VoiceCaptureWorker, VoiceFileCaptureWorker
from voice_dependencies import check_voice_dependencies, check_voice_file_dependencies
from gaze_capture import GazeCaptureWorker
from video_capture import VideoCaptureWorker, VIDEO_FILTER
from video_capture import check_video_audio_dependency
from gaze_engine import (DwellEngine, DwellState, GazeTargeting, fit_gaze_classifier,
                         inconsistent_targets, calibration_sample, best_effort_sample,
                         DWELL_THRESHOLD_MS, COOLDOWN_MS, FEATURE_DIM,
                          CALIBRATION_SETTLE_SECONDS, CALIBRATION_SAMPLE_SECONDS,
                          MAX_GAZE_GAP_SECONDS, DWELL_GRACE_SECONDS,
                         CALIBRATION_MAX_WINDOW_SECONDS, CALIBRATION_GAP_SECONDS,
                         CALIBRATION_ACCEPT_SECONDS, CALIBRATION_MIN_SAMPLES, HEAD_SWEEP_SECONDS)
from aac_speech import SpeechWorker
from phrase_graph import PhraseGraph, ROOT_WORDS
from motor_capture import MotorCaptureWorker, MOTOR_VIDEO_FILTER
from motor_engine import analyze_tapping, motor_band
from motor_protocol import MotorProtocolPanel, MOTOR_DISCLAIMER
from dashboard_catalog import DashboardCatalog
from session_diagnostics import SessionDiagnostics
from cdss_dashboard import CDSSDashboard
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

    def update_badge(self, name, text, color):
        if name in self.badges:
            set_badge(self.badges[name], text, color)


class DashboardPage(QWidget):
    module_requested = pyqtSignal(int)

    def __init__(self, session=None):
        super().__init__()
        self.session = session if session is not None else SessionDiagnostics(self)
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
            ("Modules", "6 исследовательских модулей"),
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
        table = QTableWidget(6, 3)
        table.setHorizontalHeaderLabels(["МОДУЛЬ", "ТИП", "СТАТУС"])
        table.horizontalHeader().setFont(f_sans(8, True))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setStyleSheet(f"""
            QTableWidget {{ background: {BG}; color: {TEXT_DIM}; border: 1px solid {BORDER_LIGHT};
                gridline-color: {BORDER_LIGHT}; font-family: Consolas; font-size: 10px; }}
            QHeaderView::section {{ background: {BG_ALT}; border: 1px solid {BORDER_LIGHT};
                padding: 4px; color: {TEXT_DIM}; }}
        """)
        for r, (name, typ, st) in enumerate([
            ("ТРЕМОР", "Исследовательский", "ГОТОВ / ПРОТОТИП"),
            ("НЕЙРОМИМИКА", "Исследовательский", "ГОТОВ / ПРОТОТИП"),
            ("ПУЛЬС (rPPG)", "Мониторинг", "ГОТОВ / ПРОТОТИП"),
            ("АЙТРЕКИНГ", "Вспомогательный", "ГОТОВ / ПРОТОТИП"),
            ("ГОЛОС", "Исследовательский", "ГОТОВ / ПРОТОТИП"),
            ("МОТОРИКА", "Исследовательский", "ГОТОВ / ПРОТОТИП"),
        ]):
            for column, value in enumerate((name, typ, st)):
                item = QTableWidgetItem(value)
                item.setForeground(QColor(TEXT_DIM))
                table.setItem(r, column, item)
        table.setFixedHeight(210)
        mod_lay.addWidget(table)
        top.addWidget(mod_panel)
        layout.addLayout(top)
        self.cdss = CDSSDashboard(self.session)
        layout.addWidget(self.cdss)
        self.catalog = DashboardCatalog()
        self.catalog.module_requested.connect(self.module_requested)
        layout.addWidget(self.catalog, 1)


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
        self._source_kind = 'camera'
        self._source_path = ''
        self._video_elapsed = 0.0
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
        self.protocol_panel.animate_in(self.capture_area)
        left.addWidget(self.capture_area, 1)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.btn_start = cmd_button("START CAPTURE")
        self.btn_start.clicked.connect(lambda checked=False: self._start())
        self.btn_video = cmd_button('UPLOAD VIDEO FILE', primary=False)
        self.btn_video.setToolTip('Видео: неподвижная кисть в кадре не менее 10,5 секунд; предпочтительно 15 секунд, 25 кадр/с и выше.')
        self.btn_video.clicked.connect(self._upload_video)
        self.btn_stop = cmd_button("STOP", primary=False)
        self.btn_stop.clicked.connect(lambda checked=False: self._stop())
        self.btn_stop.setEnabled(False)
        self.btn_reset = cmd_button("RESET", primary=False)
        self.btn_reset.clicked.connect(lambda checked=False: self._reset())
        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_video)
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
        self._start_source()

    def _upload_video(self):
        if self.worker is not None or self._closing:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, 'Выберите видеофайл с неподвижной рукой', '',
            VIDEO_FILTER)
        if path:
            self._start_source(path)

    def _start_source(self, video_path=None):
        if self.worker is not None or self._closing:
            return
        self._reset()
        self._source_kind = 'video' if video_path else 'camera'
        self._source_path = video_path or ''
        self._video_elapsed = 0.0
        self._analyze_on_finish = True
        self.worker = (VideoCaptureWorker(video_path, 'tremor', self) if video_path
                       else CaptureWorker(self.record_duration, self))
        self.worker.status.connect(self._on_status)
        self.worker.failed.connect(self._on_failure)
        self.worker.finished.connect(self._on_finished)
        self.btn_start.setEnabled(False)
        self.btn_video.setEnabled(False)
        self.btn_stop.setEnabled(True)
        set_badge(self.page_badge, "STARTING", AMBER)
        set_badge(self.analysis_badge, "WAITING", TEXT_MUTED)
        self.info_label.setText('Обработка видеофайла…' if video_path else 'Starting model and camera...')
        if video_path:
            self.analysis_detail.setText(f'Видео: {os.path.basename(video_path)}')
        if self.status_bar:
            self.status_bar.update_badge("MEDIAPIPE", "STARTING", AMBER)
        self.timer.start(30)
        self.worker.start()

    def _stop(self, analyze=True):
        if self.worker is None:
            return
        if self._stop_time is None:
            self._stop_time = self._video_elapsed if self._source_kind == 'video' else time.perf_counter()
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
        self._source_kind = 'camera'
        self._source_path = ''
        self._video_elapsed = 0.0
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
        self.btn_video.setEnabled(self.worker is None and not self._closing)
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
            if self._source_kind == 'video' and end is not None and end > start:
                self.record_duration = end - start
            self.recording = self._stop_time is None and not self.worker.isInterruptionRequested()
            if self.recording:
                set_badge(self.page_badge, "RECORDING", GREEN)
                set_badge(self.analysis_badge, "RECORDING", GREEN)
                self.analysis_detail.setText("Recording. Keep the hand fully visible until capture finishes.")
        old_count = self.frame_count
        for acquired, x, y in samples:
            if (self._stop_time is not None and self._source_kind == 'camera'
                    and acquired > self._stop_time):
                continue
            self.frame_count += 1
            if x is not None and y is not None:
                self.engine.add_sample(x, y, acquired)
        now = time.perf_counter()
        if self._source_kind == 'video' and latest is not None:
            self._video_elapsed = max(self._video_elapsed, float(latest[1]))
            self.record_start = 0.0
            self.record_end = max(self.record_end or 0.0, float(latest[1]))
        if self.record_start is not None:
            current = (self._stop_time if self._stop_time is not None else
                       self._video_elapsed if self._source_kind == 'video' else now)
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
        if self._source_kind == 'camera' and self._stop_time is None and not self.worker.isInterruptionRequested():
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
            self.btn_video.setEnabled(not self._closing)
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
        if self._source_kind == 'video':
            self.info_label.setText(f'Видео: {os.path.basename(self._source_path)}  |  ' + self.info_label.text())
            self.analysis_detail.setText('Источник: видеофайл. ' + self.analysis_detail.text())

        set_badge(self.page_badge, "LIMITED BAND" if limited else "COMPLETE", AMBER if limited else GREEN)
        set_badge(self.analysis_badge, "LIMITED BAND" if limited else "COMPLETE", AMBER if limited else GREEN)
        if limited and self.status_bar:
            self.status_bar.update_badge("DSP", "LIMITED", AMBER)

        self._plot_psd(results)
        self._plot_waveform(results)
        if self._source_kind == 'video':
            results['source'] = 'video'
            results['source_file'] = os.path.basename(self._source_path)
        self.protocol_panel.set_results(results)
        self.protocol_panel.reveal()

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
        self.btn_video.setEnabled(False)
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
        self.source_kind = 'camera'
        self.source_path = ''
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
        self.protocol_panel.animate_in(self.capture_area)
        left.addWidget(self.capture_area, 1)
        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.start = cmd_button('START FACIAL TEST')
        self.upload_video = cmd_button('UPLOAD VIDEO FILE', primary=False)
        self.upload_video.setToolTip('Видео 15 секунд: покой 0–5 с, улыбка 5–10 с, поднятие бровей 10–15 с.')
        self.stop = cmd_button('STOP', primary=False)
        self.reset = cmd_button('RESET', primary=False)
        self.start.clicked.connect(self._start)
        self.upload_video.clicked.connect(self._upload_video)
        self.stop.clicked.connect(self._stop)
        self.reset.clicked.connect(self._reset)
        self.stop.setEnabled(False)
        controls.addWidget(self.start)
        controls.addWidget(self.upload_video)
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
        self._start_source()

    def _upload_video(self):
        if self.worker is not None or self.closing:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, 'Выберите подготовленный видеофайл', '',
            VIDEO_FILTER)
        if path:
            self._start_source(path)

    def _start_source(self, video_path=None):
        if self.worker is not None or self.closing:
            return
        self._reset()
        self.source_kind = 'video' if video_path else 'camera'
        self.source_path = video_path or ''
        self.worker = (VideoCaptureWorker(video_path, 'facial', self) if video_path
                       else FaceCaptureWorker(self))
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._capture_failed)
        self.worker.status.connect(self._capture_status)
        self.start.setEnabled(False)
        self.upload_video.setEnabled(False)
        self.stop.setEnabled(True)
        set_badge(self.badge, 'STARTING', AMBER)
        self.guidance.setText(('Видео: нейтральное лицо 0–5 с, улыбка 5–10 с, поднятые брови 10–15 с.'
                               if video_path else 'Starting camera and face model. First run may download the model.'))
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
        self.source_kind = 'camera'
        self.source_path = ''
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
        self.upload_video.setEnabled(self.worker is None and not self.closing)
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
            if self.status_bar and self.source_kind == 'camera':
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
                if self.source_kind == 'video':
                    result['source'] = 'video'
                    result['source_file'] = os.path.basename(self.source_path)
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
                if self.source_kind == 'video':
                    self.guidance.setText(f'Видео: {os.path.basename(self.source_path)}. ' + self.guidance.text())
                self.verdict.setText('Measurements complete. Percentage of eye-corner span; not a probability of paresis or stroke.')
                self.video.setText('RECORDING COMPLETE / SEE RESULTS')
                self.protocol_panel.set_results(result)
                self.protocol_panel.reveal()
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
            self.upload_video.setEnabled(not self.closing)
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
        self.upload_video.setEnabled(False)
        self.reset.setEnabled(False)
        return self.worker is None


class MotorPage(QWidget):
    capture_finished = pyqtSignal()

    def __init__(self, status_bar=None):
        super().__init__()
        self.status_bar = status_bar
        self.worker = None
        self.samples = []
        self.results = None
        self.cancelled = self.closing = self._reset_pending = False
        self.capture_error = ''
        self.source_kind, self.source_path = 'camera', ''
        self.last_plot = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel('МОТОРИКА / ОЦЕНКА БРАДИКИНЕЗИИ (ТЕСТ UPDRS 3.4)')
        title.setFont(f_sans(13, True))
        title.setStyleSheet(f'color: {TEXT};')
        title.setWordWrap(True)
        header.addWidget(title, 1)
        self.badge = make_status_badge('IDLE', TEXT_MUTED)
        header.addWidget(self.badge)
        layout.addLayout(header)
        subtitle = QLabel('Анализ частоты и затухания амплитуды движений | MediaPipe Hands | Исследовательский прототип')
        subtitle.setFont(f_mono(9))
        subtitle.setStyleSheet(f'color: {TEXT_DIM};')
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        body = QHBoxLayout()
        body.setSpacing(8)
        left = QVBoxLayout()
        left.setSpacing(8)
        self.guidance = QLabel()
        self.guidance.setWordWrap(True)
        self.guidance.setFont(f_mono(10, True))
        self.guidance.setStyleSheet(f'color: {NAVY}; background: {BG_ALT}; border: 1px solid {BORDER}; padding: 6px;')
        left.addWidget(self.guidance)
        self.video = VideoLabel('МОТОРИКА / БОЛЬШОЙ И УКАЗАТЕЛЬНЫЙ ПАЛЬЦЫ (4–8)')
        self.video.setMinimumSize(480, 0)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setStyleSheet(f'background: {NAVY}; color: #94A3B8; border: 1px solid {BORDER};')
        self.capture_area = QWidget()
        self.capture_area.setMinimumSize(0, 0)
        self.capture_layout = QVBoxLayout(self.capture_area)
        self.capture_layout.setContentsMargins(0, 0, 0, 0)
        self.capture_layout.setSpacing(0)
        self.capture_layout.addWidget(self.video, 1)
        self.protocol_panel = MotorProtocolPanel()
        self.protocol_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.capture_layout.addWidget(self.protocol_panel, 1)
        self.protocol_panel.animate_in(self.capture_area)
        self.split_animation = self.protocol_panel.reveal_controller.animation
        left.addWidget(self.capture_area, 1)
        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.start = cmd_button('НАЧАТЬ ТЕСТ МОТОРИКИ')
        self.upload_video = cmd_button('UPLOAD VIDEO FILE', primary=False)
        self.upload_video.setToolTip('MP4, AVI, MOV: одна кисть, полный цикл раскрытия–смыкания, 10–15 секунд, минимум 25 кадр/с.')
        self.stop = cmd_button('STOP', primary=False)
        self.reset = cmd_button('RESET', primary=False)
        for button in (self.start, self.upload_video, self.stop, self.reset):
            controls.addWidget(button)
        self.start.clicked.connect(self._start)
        self.upload_video.clicked.connect(self._upload_video)
        self.stop.clicked.connect(self._stop)
        self.reset.clicked.connect(self._reset)
        left.addLayout(controls)
        self.progress = thin_progress()
        left.addWidget(self.progress)
        body.addLayout(left, 3)
        right = QVBoxLayout()
        right.setSpacing(8)
        index_panel, index_layout = panel('ИНДЕКС МОТОРИКИ И ЗАМЕДЛЕНИЯ')
        index_grid = QGridLayout()
        index_grid.setSpacing(2)
        self.frequency = MetricDisplay('Частота тапов', '---', 'Гц')
        self.decrement = MetricDisplay('Изменение амплитуды', '---', '%')
        self.frequency.setToolTip('Отображаемый ориентир >3 Гц; диапазон наблюдения протокола начинается с 2,8 Гц.')
        self.decrement.setToolTip('Отрицательное значение — снижение. Процент рассчитывается по линейному тренду пиков.')
        index_grid.addWidget(self.frequency, 0, 0)
        index_grid.addWidget(self.decrement, 0, 1)
        index_layout.addLayout(index_grid)
        self.verdict = QLabel('Данные появятся после завершённой пробы.')
        self.verdict.setWordWrap(True)
        self.verdict.setFont(f_mono(8))
        self.verdict.setStyleSheet(f'color: {TEXT_DIM}; border: none;')
        index_layout.addWidget(self.verdict)
        right.addWidget(index_panel)
        parameters, parameter_layout = panel('КИНЕМАТИЧЕСКИЕ ПАРАМЕТРЫ')
        grid = QGridLayout()
        grid.setSpacing(2)
        self.taps = MetricDisplay('Полных циклов', '---')
        self.rhythm = MetricDisplay('Ритм CV', '---', '%')
        self.rhythm.setToolTip('Отображаемый ориентир <15%; диапазон наблюдения протокола допускает CV до 20%.')
        self.amplitude = MetricDisplay('Средняя пиковая амплитуда', '---', 'норм. ед.')
        grid.addWidget(self.taps, 0, 0)
        grid.addWidget(self.rhythm, 0, 1)
        grid.addWidget(self.amplitude, 1, 0, 1, 2)
        parameter_layout.addLayout(grid)
        self.quality = QLabel('Нормирование: расстояние 4–8 / длина ладони 0–9.')
        self.quality.setWordWrap(True)
        self.quality.setFont(f_mono(8))
        self.quality.setStyleSheet(f'color: {TEXT_DIM}; border: none;')
        parameter_layout.addWidget(self.quality)
        right.addWidget(parameters)
        chart_header = QHBoxLayout()
        chart_header.addWidget(QLabel('РАСКРЫТИЕ ПАЛЬЦЕВ / ПИКИ И ТРЕНД'), 1)
        self.chart_mode = QComboBox()
        self.chart_mode.addItems(['Сигнал', 'Пики и тренд'])
        self.chart_mode.currentIndexChanged.connect(self._plot_result)
        chart_header.addWidget(self.chart_mode)
        right.addLayout(chart_header)
        self.wave = PlotCanvas()
        self.wave.mpl_connect('scroll_event', self._zoom_chart)
        right.addWidget(self.wave, 1)
        disclaimer = QLabel(MOTOR_DISCLAIMER)
        disclaimer.setWordWrap(True)
        disclaimer.setFont(f_mono(8))
        right.addWidget(disclaimer)
        body.addLayout(right, 2)
        layout.addLayout(body)
        self._reset()

    def _restore_video(self):
        self.protocol_panel.hide()

    def _show_protocol(self):
        self.protocol_panel.reveal()

    def _start(self):
        self._start_source()

    def _upload_video(self):
        if self.worker is not None or self.closing:
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Выберите видео теста моторики', '', MOTOR_VIDEO_FILTER)
        if path:
            self._start_source(path)

    def _start_source(self, path=''):
        if self.worker is not None or self.closing:
            return
        self._reset()
        self.cancelled = self._reset_pending = False
        self.source_kind, self.source_path = ('video', str(path)) if path else ('camera', '')
        self.worker = MotorCaptureWorker(path=path, parent=self)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._capture_failed)
        self.worker.status.connect(self._capture_status)
        self.start.setEnabled(False)
        self.upload_video.setEnabled(False)
        self.stop.setEnabled(True)
        self.guidance.setText('Проверка видео и запуск модели кисти…' if path else 'Запуск камеры и модели кисти…')
        set_badge(self.badge, 'STARTING', AMBER)
        self.last_plot = 0
        self.timer.start(40)
        self.worker.start()

    def _stop(self):
        if self.worker is not None:
            self.cancelled = True
            self.worker.requestInterruption()
            self.stop.setEnabled(False)
            set_badge(self.badge, 'STOPPING', AMBER)

    def _reset(self):
        self._stop()
        self._reset_pending = self.worker is not None
        self._restore_video()
        self.protocol_panel.clear()
        self.samples = []
        self.results = None
        self.capture_error = ''
        self.progress.setValue(0)
        self.chart_mode.setEnabled(False)
        self.wave.clear_plot()
        for metric in (self.frequency, self.decrement, self.taps, self.rhythm, self.amplitude):
            metric.set_value('---', TEXT)
        self.verdict.setText('Данные появятся после завершённой пробы.')
        self.quality.setText('Нормирование: расстояние 4–8 / длина ладони 0–9.')
        self.guidance.setText('Одна кисть ладонью к камере. Максимально быстро и широко раскрывайте и смыкайте большой и указательный пальцы 15 секунд.')
        self.video.setText('МОТОРИКА / БОЛЬШОЙ И УКАЗАТЕЛЬНЫЙ ПАЛЬЦЫ (4–8)')
        self.start.setEnabled(self.worker is None and not self.closing)
        self.upload_video.setEnabled(self.worker is None and not self.closing)
        self.stop.setEnabled(False)
        set_badge(self.badge, 'IDLE', TEXT_MUTED)
        if self.worker is None:
            self.cancelled = False

    def _capture_status(self, message):
        if self.worker is not None and not self.cancelled and not self.samples:
            self.guidance.setText(message)

    def _capture_failed(self, message):
        if not self.cancelled:
            self.capture_error = message
            self.guidance.setText(message)
            self.verdict.setText('Запись не завершена: ' + message)
            self.protocol_panel.clear()
            set_badge(self.badge, 'RETRY', AMBER)

    def _poll(self):
        try:
            self._drain_frames()
        except Exception as exc:
            self._capture_failed('Ошибка обработки кадров: ' + str(exc))
            if self.worker is not None:
                self.worker.requestInterruption()

    def _drain_frames(self):
        if self.worker is None:
            return
        latest, samples = self.worker.take_pending()
        if self.cancelled or self.capture_error:
            return
        self.samples.extend(samples)
        if latest is not None:
            if latest.get('message'):
                self.guidance.setText(latest['message'])
            self.progress.setValue(int(latest['progress'] * 100))
            if latest['state'] != 'COMPLETE':
                set_badge(self.badge, latest['state'], GREEN)
            image = latest.get('image')
            if image is not None:
                h, w = image.shape[:2]
                self.video.setPixmap(QPixmap.fromImage(QImage(image.data, w, h, image.strides[0], QImage.Format.Format_RGB888)))
            if self.status_bar:
                if self.source_kind == 'camera':
                    self.status_bar.update_badge('CAM', 'ACTIVE', GREEN)
                self.status_bar.update_badge('MEDIAPIPE', 'HAND LOADED', GREEN)
        if samples and time.perf_counter() - self.last_plot > .3:
            self.last_plot = time.perf_counter()
            self.wave.clear_plot()
            origin = self.samples[0]['timestamp']
            self.wave.ax.plot([s['timestamp'] - origin for s in self.samples],
                              [s['distance'] if s['distance'] is not None else np.nan for s in self.samples],
                              color=NAVY, linewidth=.8)
            self._chart_labels()
            self.wave.draw_idle()

    def _finished(self):
        worker = self.worker
        if worker is None:
            return
        try:
            self._drain_frames()
            if not self.cancelled:
                if worker.error or self.capture_error:
                    raise ValueError(worker.error or self.capture_error)
                if worker.reason != 'deadline':
                    raise ValueError('Проба прервана до окончания записи.')
                start, end, _, _ = worker.snapshot()
                result = analyze_tapping(self.samples, start, end)
                result['source'] = self.source_kind
                if self.source_kind == 'video':
                    result['source_file'] = os.path.basename(self.source_path)
                self._render(result)
            elif not self._reset_pending:
                self.guidance.setText('Проба отменена. Начните новую запись для полного анализа.')
                self.video.setText('RECORDING CANCELLED')
                set_badge(self.badge, 'IDLE', TEXT_MUTED)
        except Exception as exc:
            self.results = None
            self.protocol_panel.clear()
            self._restore_video()
            self.verdict.setText('Результат не рассчитан: ' + str(exc))
            self.guidance.setText(str(exc))
            self.video.setText('RECORDING ENDED / RETRY REQUIRED')
            set_badge(self.badge, 'RETRY', AMBER)
        finally:
            self.timer.stop()
            self.worker = None
            worker.deleteLater()
            self.start.setEnabled(not self.closing)
            self.upload_video.setEnabled(not self.closing)
            self.stop.setEnabled(False)
            if self.status_bar:
                self.status_bar.update_badge('CAM', 'DISCONNECTED', TEXT_MUTED)
                self.status_bar.update_badge('MEDIAPIPE', 'NOT LOADED', TEXT_MUTED)
            self.capture_finished.emit()

    def _render(self, result):
        self.results = result
        self.frequency.set_value(f'{result["tap_frequency"]:.2f}')
        self.decrement.set_value(f'{result["amplitude_decrement"]:.1f}')
        self.taps.set_value(str(result['total_taps']))
        self.rhythm.set_value(f'{result["rhythm_cv"]:.1f}')
        self.amplitude.set_value(f'{result["mean_amplitude"]:.2f}')
        band = motor_band(result)
        self.verdict.setText({'stable': 'Темп, амплитуда и ритм стабильны по выбранным правилам наблюдения.',
                              'borderline': 'Пограничные показатели: сравните повторную пробу после отдыха.',
                              'marked': 'Выраженное замедление, снижение амплитуды или нерегулярный ритм: нужна очная оценка.'}[band])
        self.quality.setText(f'Кадров: {result["samples"]}; частота: {result["rate"]:.1f}/с; покрытие: {result["coverage"] * 100:.1f}%.')
        self.chart_mode.setEnabled(True)
        self._plot_result()
        self.protocol_panel.set_results(result)
        self.progress.setValue(100)
        self.guidance.setText('RECORDING COMPLETE / SEE RESULTS')
        set_badge(self.badge, 'COMPLETE', GREEN)
        self._show_protocol()

    def _chart_labels(self):
        self.wave.ax.set_xlabel('Время (с)', fontsize=7)
        self.wave.ax.set_ylabel('Расстояние 4–8 / 0–9', fontsize=7)
        self.wave.ax.set_title('РАСКРЫТИЕ ПАЛЬЦЕВ / ПИКИ И ТРЕНД', fontsize=8, loc='left')

    def _plot_result(self, index=None):
        if self.results is None:
            return
        r = self.results
        self.wave.clear_plot()
        ax = self.wave.ax
        if self.chart_mode.currentIndex() == 0:
            ax.plot(r['time'], r['distance'], color='#CBD5E1', linewidth=.5)
            ax.plot(r['time'], r['smoothed'], color=NAVY, linewidth=.8, label='Раскрытие')
        ax.scatter(r['peak_times'], r['peak_amplitudes'], s=12, color='#2563EB', label='Полные циклы')
        ax.plot(r['peak_times'], r['trend'], color=AMBER, linewidth=1.2, label='Линейный тренд')
        ax.legend(fontsize=6, loc='upper right')
        self._chart_labels()
        self.wave.draw_idle()

    def _zoom_chart(self, event):
        if event.inaxes is not self.wave.ax or event.xdata is None or self.results is None:
            return
        left, right = self.wave.ax.get_xlim()
        factor = .8 if event.button == 'up' else 1.25
        self.wave.ax.set_xlim(event.xdata + (left - event.xdata) * factor,
                              event.xdata + (right - event.xdata) * factor)
        self.wave.draw_idle()

    def cleanup(self):
        self.closing = True
        self.split_animation.stop()
        self._stop()
        self.start.setEnabled(False)
        self.upload_video.setEnabled(False)
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
        self.source_kind = 'camera'
        self.source_path = ''
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
        self.protocol_panel.animate_in(self.capture_area)
        left.addWidget(self.capture_area, 1)
        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.start = cmd_button('START MEASUREMENT')
        self.upload_video = cmd_button('UPLOAD VIDEO FILE', primary=False)
        self.upload_video.setToolTip('Видео спокойного лица, лоб и обе щеки видны, не менее 20 секунд; предпочтительно 30 секунд.')
        self.stop = cmd_button('STOP', primary=False)
        self.reset = cmd_button('RESET', primary=False)
        self.start.clicked.connect(self._start)
        self.upload_video.clicked.connect(self._upload_video)
        self.stop.clicked.connect(self._stop)
        self.reset.clicked.connect(self._reset)
        self.stop.setEnabled(False)
        controls.addWidget(self.start)
        controls.addWidget(self.upload_video)
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
        self.source_kind = 'camera'
        self.source_path = ''
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
        self.upload_video.setEnabled(self.worker is None and not self.closing)
        self.stop.setEnabled(False)

    def _start(self):
        self._start_source()

    def _upload_video(self):
        if self.worker is not None or self.closing:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, 'Выберите видеофайл спокойного лица', '',
            VIDEO_FILTER)
        if path:
            self._start_source(path)

    def _start_source(self, video_path=None):
        if self.worker is not None or self.closing:
            return
        self._reset()
        self.source_kind = 'video' if video_path else 'camera'
        self.source_path = video_path or ''
        self.worker = (VideoCaptureWorker(video_path, 'rppg', self) if video_path
                       else RPPGCaptureWorker(self))
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.status.connect(self._status)
        self.start.setEnabled(False)
        self.upload_video.setEnabled(False)
        self.stop.setEnabled(True)
        set_badge(self.badge, 'STARTING', AMBER)
        self.guidance.setText(('Обработка первых 30 секунд видео: смотрите прямо в камеру и оставайтесь неподвижны.'
                               if video_path else 'Starting the face model and camera; prepare for a 30-second resting capture.'))
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
            if self.status_bar and self.source_kind == 'camera':
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
                if self.source_kind == 'video':
                    self.detail.setText(f'Видео: {os.path.basename(self.source_path)}. ' + self.detail.text())
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
            self.upload_video.setEnabled(not self.closing)
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
        if self.source_kind == 'video':
            self.guidance.setText(f'Источник: видеофайл {os.path.basename(self.source_path)}. ' + self.guidance.text())
        if self.status_bar:
            self.status_bar.update_badge('DSP', 'CAUTION' if warnings else 'COMPLETE', AMBER if warnings else GREEN)
        self.protocol_panel.set_results(result)
        self.protocol_panel.reveal()

    def cleanup(self):
        self.closing = True
        self._stop()
        self.start.setEnabled(False)
        self.upload_video.setEnabled(False)
        self.reset.setEnabled(False)
        return self.worker is None


class DwellButton(QPushButton):
    def __init__(self, text, parent=None, navigation=False):
        super().__init__(text, parent)
        self.progress = 0.0
        self.target = False
        self.flash_until = 0.0
        self.flash_timer = QTimer(self)
        self.flash_timer.setSingleShot(True)
        self.flash_timer.timeout.connect(self.update)
        self.navigation = navigation
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(QFont('Consolas', 11 if navigation else 15, QFont.Weight.Bold))
        self.setStyleSheet(f"""
            QPushButton {{ background: {BG}; color: {NAVY}; border: 1px solid {NAVY_MID};
                border-radius: 0px; padding: 3px 5px; }}
            QPushButton:hover {{ background: {BG_ALT}; }}
        """)
        self.navigation_size_hint = super().sizeHint() if navigation else None

    def sizeHint(self):
        hint = getattr(self, 'navigation_size_hint', None)
        return hint if hint is not None else super().sizeHint()

    def minimumSizeHint(self):
        return self.sizeHint() if self.navigation else super().minimumSizeHint()

    def set_dwell(self, progress, target=False):
        self.progress = max(0.0, min(1.0, float(progress)))
        self.target = bool(target)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        from PyQt6.QtGui import QColor, QPainter, QPen
        painter = QPainter(self)
        flashing = time.perf_counter() < self.flash_until
        if self.target or flashing:
            pen = QPen(QColor('#16A34A' if flashing or self.progress >= 1 else '#2563EB'), 2)
            painter.setPen(pen)
            painter.drawRect(self.rect().adjusted(2, 2, -3, -3))
        if self.progress > 0:
            width = int((self.width() - 2) * self.progress)
            painter.fillRect(1, self.height() - 6, width, 5, QColor('#2563EB'))
        painter.end()

    def flash(self):
        self.flash_until = time.perf_counter() + 0.15
        self.flash_timer.start(160)
        self.update()

    def clear_flash(self):
        self.flash_until = 0.0
        self.flash_timer.stop()
        self.update()


AAC_WORDS_PER_SHEET = 4
AAC_GRID_COLUMNS = 2
AAC_CATEGORIES = [
    ('ЭКСТРЕННАЯ ПОМОЩЬ / ПОТРЕБНОСТИ', ['ПОМОЩЬ', 'ВОДА', 'БОЛЬ', 'ТУАЛЕТ']),
    ('ОТВЕТЫ / ПОНИМАНИЕ', ['ДА', 'НЕТ', 'НЕ ПОНИМАЮ', 'ХОРОШО']),
    ('ПОЛОЖЕНИЕ / ТЕМПЕРАТУРА', ['ПЕРЕВЕРНУТЬ', 'ПОДНЯТЬ ГОЛОВУ', 'ХОЛОДНО', 'ЖАРКО']),
    ('КОМФОРТ / ОКРУЖЕНИЕ', ['ДУШНО', 'ЧЕШЕТСЯ', 'СВЕТ', 'ОЧКИ']),
    ('МЕДИЦИНА / БЫТ', ['ВРАЧ', 'ТАБЛЕТКА', 'ЕДА', 'УСТАЛ']),
    ('ОБЩЕНИЕ / БЛИЗКИЕ', ['СПАСИБО', 'ПОЖАЛУЙСТА', 'СЕМЬЯ', 'ТЕЛЕФОН']),
]


class AACContextLabel(PlainLabel):
    """Russian context text must not change the calibrated navigation geometry."""

    def freeze_size_hint(self):
        self.fixed_hint = super().sizeHint()
        self.fixed_minimum_hint = super().minimumSizeHint()

    def sizeHint(self):
        return self.fixed_hint if hasattr(self, 'fixed_hint') else super().sizeHint()

    def minimumSizeHint(self):
        return self.fixed_minimum_hint if hasattr(self, 'fixed_minimum_hint') else super().minimumSizeHint()


class AACPage(QWidget):
    capture_finished = pyqtSignal()
    dwell_threshold = DWELL_THRESHOLD_MS / 1000
    cooldown_duration = COOLDOWN_MS / 1000

    def __init__(self, status_bar=None):
        super().__init__()
        self.status_bar = status_bar
        self.worker = None
        self.closing = False
        self.tracking_stopping = False
        self.tracking_error = ''
        # Calibration: CAPTURE pass, head-movement step, reverse CHECK pass and,
        # if a button's windows disagree, one REFINE pass for those buttons.
        self.calibration_model = None
        self.calibration_active = False
        self.calibration_stage = 'capture'
        self.calibration_queue = []
        self.calibration_stage_total = 0
        self.calibration_windows = []
        self.calibration_head_motion = None
        self.calibration_refined = False
        self.calibration_started = None
        self.calibration_last_timestamp = None
        self.calibration_pending = []
        self.calibration_pending_times = []
        self.calibration_next_check = 0.0
        self.calibration_hint = ''
        self.calibration_warning = ''
        self.frames_seen = False
        self.last_sample_arrival = None
        self.last_gaze_timestamp = None
        self.latest_gaze = None
        self.latest_features = None
        self.current_target = None
        self.dwell_started = None
        self.cooldown_until = 0.0
        self.category_index = 0
        self.phrase_graph = PhraseGraph()
        self.is_predictive = False
        self.predictive_context = None
        self.root_words = tuple(ROOT_WORDS)
        self.dwell_engine = DwellEngine()
        self.speech = SpeechWorker()
        self.speech_error = ''
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._poll)
        self.speech_timer = QTimer(self)
        self.speech_timer.timeout.connect(self._poll_speech)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(5)
        header = QHBoxLayout()
        title = QLabel('AAC / КОММУНИКАЦИЯ С ПОМОЩЬЮ ВЗГЛЯДА')
        title.setFont(f_sans(13, True))
        title.setStyleSheet(f'color: {TEXT};')
        header.addWidget(title)
        header.addStretch()
        self.badge = make_status_badge('IDLE', TEXT_MUTED)
        header.addWidget(self.badge)
        layout.addLayout(header)
        self.guidance = QLabel('Калибровка нужна для привязки взгляда к экрану. Сидите неподвижно и следуйте подсказкам.')
        self.guidance.setFont(f_mono(8))
        self.guidance.setWordWrap(True)
        self.guidance.setStyleSheet(f'color: {TEXT_DIM};')
        layout.addWidget(self.guidance)

        top = QHBoxLayout()
        top.setSpacing(6)
        tracking_panel, tracking_layout = panel('Калибровка взгляда и камера')
        self.tracking_video = VideoLabel('ПОСМОТРИТЕ В КАМЕРУ')
        self.tracking_video.setMinimumSize(240, 110)
        self.tracking_video.setStyleSheet(f'background: {NAVY}; color: #94A3B8; border: 1px solid {BORDER};')
        self.tracking_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tracking_layout.addWidget(self.tracking_video, 1)
        metrics = QHBoxLayout()
        self.gaze_x = MetricDisplay('Gaze X', '---')
        self.gaze_y = MetricDisplay('Gaze Y', '---')
        self.dwell_display = MetricDisplay('DWELL', '0', 'мс')
        for metric in (self.gaze_x, self.gaze_y, self.dwell_display):
            metrics.addWidget(metric)
        tracking_layout.addLayout(metrics)
        track_buttons = QHBoxLayout()
        self.btn_tracking = cmd_button('START GAZE TRACKING')
        self.btn_calibrate = cmd_button('CALIBRATE GAZE', primary=False)
        self.btn_stop_tracking = cmd_button('STOP TRACKING', primary=False)
        self.btn_calibrate.setEnabled(False)
        self.btn_stop_tracking.setEnabled(False)
        self.btn_tracking.clicked.connect(self._start_tracking)
        self.btn_calibrate.clicked.connect(self._begin_calibration)
        self.btn_stop_tracking.clicked.connect(self._stop_tracking)
        for button in (self.btn_tracking, self.btn_calibrate, self.btn_stop_tracking):
            button.setFont(f_mono(8, True))
            button.setFixedHeight(27)
            track_buttons.addWidget(button)
        tracking_layout.addLayout(track_buttons)
        top.addWidget(tracking_panel, 2)

        output_panel, output_layout = panel('Вывод сообщения')
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumHeight(88)
        self.output.setPlaceholderText('Выбранные слова появятся здесь')
        self.output.setFont(f_mono(13, True))
        self.output.setStyleSheet(f'background: {BG_ALT}; color: {NAVY}; border: 1px solid {BORDER};')
        output_layout.addWidget(self.output, 1)
        output_buttons = QHBoxLayout()
        self.btn_speak = cmd_button('SPEAK')
        self.btn_clear = cmd_button('CLEAR', primary=False)
        self.btn_speak.clicked.connect(self._speak_output)
        self.btn_clear.clicked.connect(self._on_clear_clicked)
        output_buttons.addWidget(self.btn_speak)
        output_buttons.addWidget(self.btn_clear)
        output_layout.addLayout(output_buttons)
        top.addWidget(output_panel, 1)
        layout.addLayout(top, 1)

        self.grid_panel, grid_layout = panel('Сетка фраз')
        self.category_label = AACContextLabel('')
        self.category_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.category_label.setFont(f_mono(10, True))
        self.category_label.setStyleSheet(f'color: {NAVY}; border: none;')
        navigation = QHBoxLayout()
        navigation.setSpacing(6)
        self.previous_button = DwellButton('< ПРЕДЫДУЩАЯ КАТЕГОРИЯ', navigation=True)
        self.previous_button.setMinimumHeight(58)
        self.previous_button.clicked.connect(lambda: self._change_category(-1))
        navigation.addWidget(self.previous_button, 1)
        navigation.addWidget(self.category_label, 1)
        self.next_button = DwellButton('СЛЕДУЮЩАЯ КАТЕГОРИЯ >', navigation=True)
        self.next_button.setMinimumHeight(58)
        self.next_button.clicked.connect(lambda: self._change_category(1))
        navigation.addWidget(self.next_button, 1)
        grid_layout.addLayout(navigation)
        self.matrix = QWidget()
        matrix_layout = QGridLayout(self.matrix)
        matrix_layout.setContentsMargins(0, 0, 0, 0)
        matrix_layout.setSpacing(6)
        self.card_buttons = []
        for index in range(AAC_WORDS_PER_SHEET):
            button = DwellButton('')
            button.setMinimumHeight(100)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            button.clicked.connect(lambda checked=False, i=index: self._select_word(i))
            self.card_buttons.append(button)
            matrix_layout.addWidget(button, index // AAC_GRID_COLUMNS, index % AAC_GRID_COLUMNS)
        for column in range(AAC_GRID_COLUMNS):
            matrix_layout.setColumnStretch(column, 1)
        matrix_layout.setRowStretch(0, 1)
        matrix_layout.setRowStretch(1, 1)
        grid_layout.addWidget(self.matrix, 1)
        self._set_category(0)
        self.category_label.freeze_size_hint()
        layout.addWidget(self.grid_panel, 2)

    # ------------------------------------------------------------------ session

    def _start_tracking(self):
        if self.worker is not None or self.closing:
            return
        self.tracking_stopping = False
        self.tracking_error = ''
        self.calibration_model = None
        self.frames_seen = False
        self.last_sample_arrival = None
        self.last_gaze_timestamp = None
        self._clear_gaze()
        self.dwell_engine.reset()
        self._reset_dwell()
        self.worker = GazeCaptureWorker(self)
        self.worker.finished.connect(self._tracking_finished)
        self.worker.failed.connect(self._capture_failed)
        self.btn_tracking.setEnabled(False)
        self.btn_stop_tracking.setEnabled(True)
        self._begin_calibration_state()
        self.guidance.setText('Запуск камеры и модели взгляда… Затем смотрите в центр подсвеченной кнопки.')
        if self.status_bar:
            self.status_bar.update_badge('CAM', 'STARTING', AMBER)
            self.status_bar.update_badge('MEDIAPIPE', 'STARTING', AMBER)
        self.timer.start(16)
        self.worker.start()

    def _begin_calibration(self):
        if self.closing or self.tracking_stopping or self.calibration_active:
            return
        if self.worker is None:
            self._start_tracking()
            return
        self._begin_calibration_state()

    # -------------------------------------------------------------- calibration

    def _calibration_buttons(self):
        return [self.previous_button, self.next_button, *self.card_buttons]

    def _target_centers(self):
        centers = [button.mapTo(self, button.rect().center()) for button in self._calibration_buttons()]
        return np.array([[point.x() / max(self.width(), 1), point.y() / max(self.height(), 1)]
                         for point in centers])

    def _target_rectangles(self):
        rectangles = {}
        for button in self._calibration_buttons():
            if button.isVisible() and button.isEnabled():
                origin = button.mapTo(self, QPoint(0, 0))
                rect = button.rect().translated(origin)
                rectangles[button] = (rect.left(), rect.top(), rect.right(), rect.bottom())
        return rectangles

    def _begin_calibration_state(self):
        if self.is_predictive:
            self._set_category(self.category_index)
        count = len(self._calibration_buttons())
        self.calibration_active = True
        self.calibration_windows = [[] for _ in range(count)]
        self.calibration_head_motion = None
        self.calibration_refined = False
        self.calibration_hint = ''
        self.calibration_warning = ''
        self.calibration_last_timestamp = None
        self.dwell_engine.reset()
        self._reset_dwell()
        self.btn_calibrate.setEnabled(False)
        set_badge(self.badge, 'CALIBRATING', AMBER)
        self._start_calibration_stage('capture', list(range(count)))

    def _start_calibration_stage(self, stage, queue):
        self.calibration_stage = stage
        self.calibration_queue = list(queue)
        self.calibration_stage_total = len(self.calibration_queue)
        self._restart_calibration_window()
        self._highlight_calibration_target(0.0)
        self._show_calibration_prompt()

    def _restart_calibration_window(self):
        self.calibration_started = None
        self.calibration_pending = []
        self.calibration_pending_times = []
        self.calibration_next_check = 0.0

    def _calibration_target_button(self):
        return self._calibration_buttons()[self.calibration_queue[0]]

    def _highlight_calibration_target(self, progress):
        if not self.calibration_active or not self.calibration_queue:
            return
        target = self._calibration_target_button()
        for button in self._calibration_buttons():
            if button is target:
                button.set_dwell(progress, True)
            elif button.target or button.progress:
                button.set_dwell(0, False)

    def _show_calibration_prompt(self, status=''):
        if not self.calibration_active or not self.calibration_queue:
            return
        text = self._calibration_target_button().text()
        if self.calibration_stage == 'head':
            prompt = (f'ДВИЖЕНИЕ ГОЛОВЫ: продолжайте смотреть на «{text}» и медленно поворачивайте '
                      'голову влево-вправо, затем вверх-вниз.')
        else:
            name = {'capture': 'КАЛИБРОВКА', 'check': 'ПРОВЕРКА', 'refine': 'УТОЧНЕНИЕ'}[self.calibration_stage]
            step = self.calibration_stage_total - len(self.calibration_queue) + 1
            prompt = f'{name} {step}/{self.calibration_stage_total}: смотрите в центр «{text}».'
        if not self.frames_seen:
            status = status or 'Запуск камеры…'
        extra = status or self.calibration_hint
        if extra:
            prompt += ' ' + extra
        if self.guidance.text() != prompt:
            self.guidance.setText(prompt)

    def _collect_calibration(self, features, timestamp):
        if not self.calibration_active or not self.calibration_queue or features is None:
            return
        value = np.asarray(features, dtype=float)
        if value.shape != (FEATURE_DIM,) or not np.isfinite(value).all():
            return
        if self.calibration_last_timestamp is not None:
            if timestamp <= self.calibration_last_timestamp:
                return
            if timestamp - self.calibration_last_timestamp > CALIBRATION_GAP_SECONDS:
                self._restart_calibration_window()
        self.calibration_last_timestamp = timestamp
        if self.calibration_started is None:
            self.calibration_started = timestamp
        elapsed = timestamp - self.calibration_started
        head_step = self.calibration_stage == 'head'
        collect = HEAD_SWEEP_SECONDS if head_step else CALIBRATION_SAMPLE_SECONDS
        duration = CALIBRATION_SETTLE_SECONDS + collect
        if elapsed + 1e-7 >= CALIBRATION_SETTLE_SECONDS:
            self.calibration_pending.append(value)
            self.calibration_pending_times.append(timestamp)
            limit = HEAD_SWEEP_SECONDS if head_step else CALIBRATION_MAX_WINDOW_SECONDS
            while timestamp - self.calibration_pending_times[0] > limit:
                self.calibration_pending_times.pop(0)
                self.calibration_pending.pop(0)
        self._highlight_calibration_target(min(elapsed / duration, 1.0))
        self.dwell_display.set_value(str(int(min(elapsed, duration) * 1000)))
        if elapsed + 1e-7 < duration or timestamp + 1e-7 < self.calibration_next_check:
            self._show_calibration_prompt()
            return
        self.calibration_next_check = timestamp + 0.25
        window = np.array(self.calibration_pending)
        if head_step:
            if len(window) >= CALIBRATION_MIN_SAMPLES:
                self._accept_calibration_window(window)
            return
        span = self.calibration_pending_times[-1] - self.calibration_pending_times[0] if len(window) > 1 else 0.0
        try:
            if span + 1e-7 < 0.75 * CALIBRATION_SAMPLE_SECONDS:
                raise ValueError('Недостаточно кадров: продолжайте смотреть на кнопку.')
            _, samples = calibration_sample(window)
        except ValueError as exc:
            if elapsed < CALIBRATION_ACCEPT_SECONDS or len(window) < CALIBRATION_MIN_SAMPLES:
                self.calibration_hint = str(exc)
                self._show_calibration_prompt()
                return
            # Long, restless fixation: keep the settled end of it and move on.
            samples = best_effort_sample(window, self.calibration_pending_times)
        self._accept_calibration_window(samples)

    def _accept_calibration_window(self, samples):
        index = self.calibration_queue.pop(0)
        if self.calibration_stage == 'head':
            self.calibration_head_motion = np.array(samples, dtype=float)
        else:
            self.calibration_windows[index].append(np.array(samples, dtype=float))
        self.calibration_hint = ''
        self._restart_calibration_window()
        if self.calibration_queue:
            self._highlight_calibration_target(0.0)
            self._show_calibration_prompt()
            return
        self._finish_calibration_stage()

    def _finish_calibration_stage(self):
        count = len(self._calibration_buttons())
        if self.calibration_stage == 'capture':
            # Use the card closest to the viewport centre; no centre-column
            # card is required. Target identities stay the same on every sheet.
            centers = self._target_centers()
            card_indices = range(2, count)
            target = min(card_indices, key=lambda index: np.linalg.norm(centers[index] - [0.5, 0.5]))
            self._start_calibration_stage('head', [target])
            return
        if self.calibration_stage == 'head':
            self._start_calibration_stage('check', list(reversed(range(count))))
            return
        try:
            model = fit_gaze_classifier(self.calibration_windows, self.calibration_head_motion)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.calibration_hint = str(exc)
            self._start_calibration_stage('refine', list(range(count)))
            return
        if not self.calibration_refined:
            unresolved = inconsistent_targets(model, self.calibration_windows)
            if unresolved:
                self.calibration_refined = True
                self.calibration_hint = 'Ещё раз для точности; остальные кнопки сохранены.'
                self._start_calibration_stage('refine', unresolved)
                return
        self._complete_calibration(model)

    def _complete_calibration(self, model):
        self.calibration_model = GazeTargeting(model)
        self.calibration_active = False
        self.calibration_queue = []
        self.calibration_hint = ''
        self._restart_calibration_window()
        self.dwell_engine.reset()
        self._reset_dwell()
        self._refresh_calibration_warning()
        self.guidance.setText(('Калибровка готова. Удерживайте взгляд на слове или стрелке до '
                               'заполнения шкалы. ' + self.calibration_warning).strip())
        set_badge(self.badge, 'ACTIVE', GREEN)
        self.btn_calibrate.setEnabled(True)
        self.dwell_display.set_value('0')

    def _refresh_calibration_warning(self):
        buttons = self._calibration_buttons()
        weak = self.calibration_model.weak_pairs() if self.calibration_model is not None else []
        if weak:
            pairs = ', '.join(f'«{buttons[i].text()}» и «{buttons[j].text()}»' for i, j in weak[:3])
            self.calibration_warning = (f'Похожий взгляд у кнопок {pairs}: смотрите на них точнее '
                                        'или слегка поворачивайте голову к кнопке.')
        else:
            self.calibration_warning = ''

    # ------------------------------------------------------------------ samples

    def _poll(self):
        if self.worker is None or self.tracking_stopping:
            return
        try:
            latest, samples = self.worker.take_pending()
            now = time.perf_counter()
            if latest is not None:
                frame = latest['image']
                h, w = frame.shape[:2]
                self.tracking_video.setPixmap(QPixmap.fromImage(QImage(
                    frame.data, w, h, frame.strides[0], QImage.Format.Format_RGB888)))
                if not self.frames_seen:
                    self.frames_seen = True
                    if self.status_bar:
                        self.status_bar.update_badge('CAM', 'ACTIVE', GREEN)
                        self.status_bar.update_badge('MEDIAPIPE', 'LOADED', GREEN)
                if not samples:
                    samples = [latest]
            samples = [sample for sample in samples
                       if np.isfinite(sample['timestamp']) and (self.last_gaze_timestamp is None
                       or sample['timestamp'] > self.last_gaze_timestamp)]
            if (samples and not self.calibration_active
                    and samples[-1]['timestamp'] - samples[0]['timestamp'] > MAX_GAZE_GAP_SECONDS):
                # A stalled GUI can receive seconds of buffered fixations.
                # Only recent observations may select a word on its return.
                self._lose_gaze()
                cutoff = samples[-1]['timestamp'] - DWELL_GRACE_SECONDS
                samples = [sample for sample in samples if sample['timestamp'] >= cutoff]
            if samples:
                self.last_sample_arrival = now
            for sample in samples:
                self._process_sample(sample)
            # Child and GUI clocks are never compared; a stall is judged by
            # arrival time so no stale gaze can complete a dwell.
            if self.last_sample_arrival is not None and now - self.last_sample_arrival > 0.5:
                self.last_sample_arrival = None
                self._lose_gaze()
        except Exception as exc:
            traceback.print_exc()
            self._capture_failed(f'Ошибка обработки взгляда: {exc}')

    def _process_sample(self, sample):
        timestamp = sample['timestamp']
        if self.last_gaze_timestamp is not None and timestamp <= self.last_gaze_timestamp:
            return
        self.last_gaze_timestamp = timestamp
        gaze, features = sample.get('gaze'), sample.get('features')
        if gaze is None or features is None:
            self._clear_gaze()
            if self.calibration_active:
                self._show_calibration_prompt(sample.get('message', ''))
            else:
                self._update_selection(None, timestamp)
                self._show_gaze_guidance(sample.get('message', ''))
            return
        self.latest_gaze, self.latest_features = gaze, features
        self.gaze_x.set_value(f'{gaze[0]:.3f}')
        self.gaze_y.set_value(f'{gaze[1]:.3f}')
        if self.calibration_active:
            raw = sample.get('raw_features')
            self._collect_calibration(features if raw is None else raw, timestamp)
        else:
            self._update_selection(features, timestamp)
            self._show_gaze_guidance(sample.get('message', ''))

    def _show_gaze_guidance(self, message):
        if self.calibration_model is None:
            message = 'Нажмите «КАЛИБРОВАТЬ ВЗГЛЯД», чтобы выбирать кнопки взглядом.'
        text = ' '.join(part for part in (message, self.calibration_warning, self.speech_error) if part)
        if self.guidance.text() != text:
            self.guidance.setText(text)

    def _clear_gaze(self):
        self.latest_gaze = None
        self.latest_features = None
        self.gaze_x.set_value('---')
        self.gaze_y.set_value('---')

    def _lose_gaze(self):
        self._clear_gaze()
        self._reset_dwell()

    # --------------------------------------------------------------- selection

    def _update_selection(self, features, timestamp):
        if self.calibration_active or self.calibration_model is None:
            self._reset_dwell()
            return
        buttons = self._calibration_buttons()
        rectangles = self._target_rectangles()
        current = buttons.index(self.dwell_engine.target) if self.dwell_engine.target in buttons else None
        index = None if features is None else self.calibration_model.classify(features, current)
        point = None
        if index is not None and buttons[index] in rectangles:
            # A classified gaze targets the centre of its existing button, so
            # the 12 px entry inset and border retention are always satisfied.
            point = self._target_centers()[index] * [self.width(), self.height()]
        update = self.dwell_engine.update(point, rectangles, timestamp, missing=features is None)
        self.current_target = self.dwell_engine.target
        self.dwell_started = self.dwell_engine.started
        self.cooldown_until = self.dwell_engine.cooldown_until
        for button in buttons:
            active = button is update.target and update.triggered is None
            button.set_dwell(update.progress if active else 0, active)
        self.dwell_display.set_value(str(update.elapsed_ms if update.state == DwellState.DWELLING else 0))
        if update.triggered is not None:
            self._activate_target(update.triggered)
            update.triggered.flash()

    def _reset_dwell(self):
        if hasattr(self, 'dwell_engine'):
            self.dwell_engine.cancel_fixation()
        self.current_target = None
        self.dwell_started = None
        if hasattr(self, 'dwell_display'):
            self.dwell_display.set_value('0')
        if hasattr(self, 'card_buttons'):
            for button in [self.previous_button, self.next_button, *self.card_buttons]:
                button.set_dwell(0, False)

    def _activate_target(self, button):
        if button is self.previous_button:
            self._change_category(-1)
        elif button is self.next_button:
            self._change_category(1)
        else:
            self._select_word(self.card_buttons.index(button))

    def _change_category(self, delta):
        index = (self.category_index if self.is_predictive and delta < 0
                 else (self.category_index + delta) % len(AAC_CATEGORIES))
        self._set_category(index)
        if self.calibration_active:
            self._restart_calibration_window()
            self._highlight_calibration_target(0.0)
            self._show_calibration_prompt()
        else:
            # Manual navigation and gaze arrows both lock out selection after
            # relabeling. Use the sample clock, just like DwellEngine, and never
            # shorten the cooldown already started by a gaze-triggered arrow.
            self._context_cooldown()

    def _context_cooldown(self):
        self._reset_dwell()
        timestamp = self.dwell_engine.last_timestamp
        if timestamp is not None:
            self.dwell_engine.cooldown_until = max(
                self.dwell_engine.cooldown_until, timestamp + self.cooldown_duration)
            self.dwell_engine.state = DwellState.COOLDOWN
            self.cooldown_until = self.dwell_engine.cooldown_until

    def _set_category(self, index):
        self._reset_dwell()
        self.is_predictive = False
        self.predictive_context = None
        self.previous_button.setText('< ПРЕДЫДУЩАЯ КАТЕГОРИЯ')
        self.category_index = index % len(AAC_CATEGORIES)
        category, words = AAC_CATEGORIES[self.category_index]
        self.category_label.setText(f'КАТЕГОРИЯ [{self.category_index + 1} / {len(AAC_CATEGORIES)}]: {category}')
        for button, word in zip(self.card_buttons, words):
            button.setText(word)
        for button in self._calibration_buttons():
            button.clear_flash()
        self._refresh_calibration_warning()

    def _select_word(self, index):
        word = self.card_buttons[index].text()
        if not self.phrase_graph.has_word(word):
            return
        current = self.output.toPlainText()
        self.output.setPlainText(current + word + ' ')
        cursor = self.output.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.output.setTextCursor(cursor)
        self._speak(word)
        if self.calibration_active:
            self._reset_dwell()
            self._highlight_calibration_target(0.0)
            return
        self.is_predictive = True
        self.predictive_context = word
        self._show_predictions(word)

    def _show_predictions(self, word):
        predictions = self.phrase_graph.predict(word, n=len(self.card_buttons))
        for button, prediction in zip(self.card_buttons, predictions):
            button.setText(prediction)
        self.category_label.setText(f'ПРЕДИКТИВНЫЙ КОНТЕКСТ: [ {word} ]')
        self.previous_button.setText('[ ⟲ В КОРЕНЬ ]')
        for button in self._calibration_buttons():
            button.clear_flash()
        self._refresh_calibration_warning()
        self._context_cooldown()

    def _on_clear_clicked(self):
        self.output.clear()
        self.speech.cancel()
        self._set_category(self.category_index)
        if self.calibration_active:
            self._restart_calibration_window()
            self._highlight_calibration_target(0.0)
            self._show_calibration_prompt()
        else:
            self._context_cooldown()

    def _speak_output(self):
        text = self.output.toPlainText().strip()
        if text:
            self._speak(text)

    def _speak(self, text):
        if not self.closing:
            self.speech_error = ''
            self.speech.speak(text)
            self.speech_timer.start(50)

    def _poll_speech(self):
        errors = self.speech.take_errors()
        if errors:
            self.speech_error = errors[-1]
            self.guidance.setText(self.speech_error)

    # ---------------------------------------------------------------- shutdown

    def _capture_failed(self, message):
        self.tracking_error = message
        self.tracking_stopping = True
        self.calibration_model = None
        self.calibration_active = False
        self.calibration_queue = []
        self._reset_dwell()
        self.guidance.setText(message)
        set_badge(self.badge, 'TRACKING ERROR', RED)
        if self.status_bar:
            self.status_bar.update_badge('CAM', 'ERROR', RED)
            self.status_bar.update_badge('MEDIAPIPE', 'ERROR', RED)
        self.btn_calibrate.setEnabled(False)
        self.btn_stop_tracking.setEnabled(False)
        if self.worker is not None:
            self.worker.requestInterruption()
        self.timer.stop()

    def cleanup(self):
        self.closing = True
        self._stop_tracking()
        self.speech.close()
        self.speech_timer.stop()
        self.btn_tracking.setEnabled(False)
        self.btn_calibrate.setEnabled(False)
        self.btn_stop_tracking.setEnabled(False)
        return self.worker is None

    def _reset_session(self):
        self.calibration_model = None
        self.calibration_active = False
        self.calibration_queue = []
        self.calibration_warning = ''
        self._restart_calibration_window()
        self.last_gaze_timestamp = None
        self.last_sample_arrival = None
        self.dwell_engine.reset()
        self.cooldown_until = 0.0
        self._reset_dwell()
        self._clear_gaze()
        for button in self._calibration_buttons():
            button.clear_flash()

    def _stop_tracking(self):
        self.tracking_stopping = True
        self._reset_session()
        self.timer.stop()
        self.speech.cancel()
        self.btn_calibrate.setEnabled(False)
        self.btn_stop_tracking.setEnabled(False)
        self.tracking_video.setText('ПОСМОТРИТЕ В КАМЕРУ')
        if self.worker is not None:
            self.worker.requestInterruption()
            set_badge(self.badge, 'STOPPING', AMBER)
            if self.status_bar:
                self.status_bar.update_badge('CAM', 'STOPPING', AMBER)
        else:
            self.btn_tracking.setEnabled(not self.closing)
            set_badge(self.badge, 'IDLE', TEXT_MUTED)

    def _tracking_finished(self):
        worker = self.worker
        if worker is None:
            return
        try:
            self.timer.stop()
            self.worker = None
            worker.deleteLater()
            self._reset_session()
            self.tracking_stopping = False
            self.btn_tracking.setEnabled(not self.closing)
            self.btn_calibrate.setEnabled(False)
            self.btn_stop_tracking.setEnabled(False)
            if self.status_bar:
                self.status_bar.update_badge('CAM', 'DISCONNECTED', TEXT_MUTED)
                self.status_bar.update_badge('MEDIAPIPE', 'NOT LOADED', TEXT_MUTED)
            error = self.tracking_error or worker.error
            if error:
                self.guidance.setText(error)
                set_badge(self.badge, 'TRACKING ERROR', RED)
            elif worker.reason == 'stopped':
                self.guidance.setText('Отслеживание остановлено. Нажмите кнопку запуска для выбора слов взглядом.')
                set_badge(self.badge, 'IDLE', TEXT_MUTED)
            else:
                set_badge(self.badge, 'TRACKING ENDED', AMBER)
        finally:
            self.capture_finished.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'dwell_engine'):
            self._reset_dwell()
        # Calibration data is stored per button, not per pixel position, so a
        # layout change only restarts the window currently being measured.
        if getattr(self, 'calibration_active', False):
            self._restart_calibration_window()
            self._highlight_calibration_target(0.0)


class VoicePage(QWidget):
    capture_finished = pyqtSignal()

    def __init__(self, status_bar=None):
        super().__init__()
        self.status_bar = status_bar
        self.worker = None
        self.cancelled = False
        self.closing = False
        self.result = None
        self.source_kind = 'microphone'
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
        self.capture_area = QWidget()
        self.capture_area.setMinimumSize(0, 0)
        self.capture_layout = QVBoxLayout(self.capture_area)
        self.capture_layout.setContentsMargins(0, 0, 0, 0)
        self.capture_layout.setSpacing(8)
        self.chart_area = QWidget()
        self.chart_area.setMinimumSize(0, 0)
        self.chart_area.setLayout(graphs)
        graphs.setContentsMargins(0, 0, 0, 0)
        self.capture_layout.addWidget(self.chart_area, 1)
        self.protocol_panel = VoiceProtocolPanel()
        self.protocol_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.capture_layout.addWidget(self.protocol_panel, 1)
        self.protocol_panel.animate_in(self.capture_area)
        layout.addWidget(self.capture_area, 1)

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
        self.source_kind = 'file' if worker_type is VoiceFileCaptureWorker else 'microphone'
        self.protocol_panel.clear()
        self.start.setEnabled(False)
        self.upload.setEnabled(False)
        self.stop.setEnabled(True)
        set_badge(self.badge, 'STARTING', AMBER)
        if self.status_bar:
            source_status = {'microphone': 'STARTING', 'file': 'AUDIO FILE', 'video': 'VIDEO FILE'}[self.source_kind]
            self.status_bar.update_badge('MIC', source_status, AMBER)
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
            if self.source_kind == 'microphone':
                self.status_bar.update_badge('MIC', 'ACTIVE' if latest['state'] == 'RECORDING' else 'STANDBY', GREEN)
            else:
                self.status_bar.update_badge('MIC', 'VIDEO FILE' if self.source_kind == 'video' else 'AUDIO FILE', AMBER)

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
        if result.get('source') in ('file', 'video'):
            source_label = 'Видео' if result.get('source') == 'video' else 'Файл'
            self.detail.setText(
                f"{source_label}: {result.get('file_name', '')}; выбранный фрагмент: {result.get('segment_start_sec', 0):.2f} с; "
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
        self.protocol_panel.reveal()
        self.guidance.setText('Recording and acoustic analysis complete. See the reference guidance below.')
        set_badge(self.badge, 'COMPLETE', GREEN)

    def cleanup(self):
        self.closing = True
        self._stop()
        self.start.setEnabled(False)
        self.upload.setEnabled(False)
        self.reset.setEnabled(False)
        return self.worker is None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TecnoMedNeuro — Интеллектуальная система цифровой нейродиагностики и мониторинговых модулей")
        self.setMinimumSize(1200, 750)
        self.resize(1400, 860)
        self.setStyleSheet(f"QMainWindow {{ background: {BG}; }} * {{ border-radius: 0px; }}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        tabs = ["ОБЗОР", "ТРЕМОР", "НЕЙРОМИМИКА", "ПУЛЬС", "АЙТРЕКИНГ", "ГОЛОС", "МОТОРИКА"]
        self.nav = NavBar(tabs, self.switch_page)
        root.addWidget(self.nav)

        self.status_bar = StatusBar()
        self.session = SessionDiagnostics(self)
        self.session.stopped.connect(self._finish_close)

        self.stack = QStackedWidget()
        self.tremor_page = TremorPage(status_bar=self.status_bar)
        self._close_pending = False
        self.tremor_page.capture_finished.connect(self._finish_close)
        self.dashboard_page = DashboardPage(self.session)
        self.dashboard_page.module_requested.connect(self.switch_page)
        self.stack.addWidget(self.dashboard_page)
        self.stack.addWidget(self.tremor_page)
        self.facial_page = FacialPage(status_bar=self.status_bar)
        self.facial_page.capture_finished.connect(self._finish_close)
        self.stack.addWidget(self.facial_page)
        self.rppg_page = RPPGPage(status_bar=self.status_bar)
        self.rppg_page.capture_finished.connect(self._finish_close)
        self.stack.addWidget(self.rppg_page)
        self.aac_page = AACPage(status_bar=self.status_bar)
        self.aac_page.capture_finished.connect(self._finish_close)
        self.stack.addWidget(self.aac_page)
        self.voice_page = VoicePage(status_bar=self.status_bar)
        self.voice_page.capture_finished.connect(self._finish_close)
        self.stack.addWidget(self.voice_page)
        self.motor_page = MotorPage(status_bar=self.status_bar)
        self.motor_page.capture_finished.connect(self._finish_close)
        self.stack.addWidget(self.motor_page)
        for module, page in (('tremor', self.tremor_page), ('facial', self.facial_page),
                             ('voice', self.voice_page), ('motor', self.motor_page), ('rppg', self.rppg_page)):
            page.session = self.session
            page.protocol_panel.assessment_completed.connect(
                lambda results, key=module: self.session.publish(key, results))
            page.protocol_panel.assessment_cleared.connect(lambda key=module: self.session.clear(key))
        root.addWidget(self.stack)

        root.addWidget(self.status_bar)
        self.switch_page(0)

    def switch_page(self, index):
        if hasattr(self, 'motor_page') and index in (1, 2, 3, 4, 5, 6):
            pages = {1: self.tremor_page, 2: self.facial_page, 3: self.rppg_page,
                     4: self.aac_page, 5: self.voice_page, 6: self.motor_page}
            if any(page.worker is not None and key != index for key, page in pages.items()):
                return
        self.stack.setCurrentIndex(index)
        self.nav.set_active(index)

    def closeEvent(self, event):
        tremor_done = self.tremor_page.cleanup()
        face_done = self.facial_page.cleanup()
        rppg_done = self.rppg_page.cleanup()
        voice_done = self.voice_page.cleanup()
        aac_done = self.aac_page.cleanup()
        motor_done = self.motor_page.cleanup()
        session_done = self.session.shutdown()
        if tremor_done and face_done and rppg_done and voice_done and aac_done and motor_done and session_done:
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
