import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFrame, QGridLayout,
    QSizePolicy, QProgressBar, QSpacerItem, QSplitter, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QColor, QPalette


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


def status_badge(text, color=TEXT_MUTED):
    lbl = QLabel(text)
    lbl.setFont(f_mono(8, True))
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setFixedHeight(18)
    lbl.setStyleSheet(f"""
        background: {color};
        color: #FFFFFF;
        padding: 0 6px;
        border: none;
    """)
    return lbl


def panel(title=""):
    frame = QFrame()
    frame.setStyleSheet(f"""
        QFrame {{
            background: {BG};
            border: 1px solid {BORDER};
        }}
    """)
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


def panel_with_status(title, status_text, status_color=TEXT_MUTED):
    frame = QFrame()
    frame.setStyleSheet(f"""
        QFrame {{
            background: {BG};
            border: 1px solid {BORDER};
        }}
    """)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(10, 8, 10, 10)
    layout.setSpacing(6)
    header = QHBoxLayout()
    lbl = QLabel(title.upper())
    lbl.setFont(f_sans(9, True))
    lbl.setStyleSheet(f"color: {TEXT_DIM}; border: none; letter-spacing: 1px;")
    header.addWidget(lbl)
    header.addStretch()
    header.addWidget(status_badge(status_text, status_color))
    layout.addLayout(header)
    sep = QFrame()
    sep.setFixedHeight(1)
    sep.setStyleSheet(f"background: {BORDER_LIGHT}; border: none;")
    layout.addWidget(sep)
    return frame, layout


def feed_panel(label="VIDEO FEED"):
    frame = QFrame()
    frame.setMinimumSize(400, 300)
    frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    frame.setStyleSheet(f"""
        QFrame {{
            background: #0F172A;
            border: 1px solid {BORDER};
        }}
    """)
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
    frame.setStyleSheet(f"""
        QFrame {{
            background: {BG_ALT};
            border: 1px solid {BORDER};
        }}
    """)
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
            QPushButton {{
                background: {NAVY};
                color: #FFFFFF;
                border: 1px solid {NAVY};
                padding: 0 16px;
            }}
            QPushButton:hover {{
                background: {NAVY_MID};
            }}
        """)
    else:
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {BG};
                color: {NAVY};
                border: 1px solid {BORDER};
                padding: 0 16px;
            }}
            QPushButton:hover {{
                background: {BG_ALT};
            }}
        """)
    return btn


def thin_progress():
    p = QProgressBar()
    p.setFixedHeight(3)
    p.setValue(0)
    p.setTextVisible(False)
    p.setStyleSheet(f"""
        QProgressBar {{
            background: {BORDER_LIGHT};
            border: none;
        }}
        QProgressBar::chunk {{
            background: {NAVY};
        }}
    """)
    return p


class NavBar(QWidget):
    def __init__(self, tabs, callback):
        super().__init__()
        self.setFixedHeight(36)
        self.setStyleSheet(f"""
            QWidget {{
                background: {NAVY};
                border: none;
            }}
        """)
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
            active = i == index
            btn.setChecked(active)
            btn.setStyleSheet(self._btn_style(active))

    def _btn_style(self, active):
        if active:
            return f"""
                QPushButton {{
                    background: #1E293B;
                    color: #FFFFFF;
                    border: none;
                    border-bottom: 2px solid #3B82F6;
                    padding: 0 14px;
                    font-weight: bold;
                }}
            """
        return f"""
            QPushButton {{
                background: transparent;
                color: #94A3B8;
                border: none;
                border-bottom: 2px solid transparent;
                padding: 0 14px;
            }}
            QPushButton:hover {{
                color: #E2E8F0;
                background: #1E293B;
            }}
        """


class StatusBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(22)
        self.setStyleSheet(f"background: {BG_ALT}; border-top: 1px solid {BORDER_LIGHT};")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.setSpacing(16)

        items = [
            ("CAM", "DISCONNECTED", TEXT_MUTED),
            ("MIC", "DISCONNECTED", TEXT_MUTED),
            ("MEDIAPIPE", "NOT LOADED", TEXT_MUTED),
            ("DSP", "STANDBY", TEXT_MUTED),
        ]
        for name, state, color in items:
            row = QHBoxLayout()
            row.setSpacing(4)
            n = QLabel(name)
            n.setFont(f_mono(7, True))
            n.setStyleSheet(f"color: {TEXT_DIM}; border: none;")
            row.addWidget(n)
            row.addWidget(status_badge(state, color))
            lay.addLayout(row)

        lay.addStretch()
        ts = QLabel("MevsusLab / NDTP")
        ts.setFont(f_mono(7))
        ts.setStyleSheet(f"color: {TEXT_MUTED}; border: none;")
        lay.addWidget(ts)


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
        fields = [
            ("System", "TecnoMedNeuro Screening Suite"),
            ("Version", "0.1.0-alpha"),
            ("Backend", "MediaPipe + SciPy DSP"),
            ("Input", "USB Webcam + Microphone"),
            ("Modules", "5 active"),
            ("Framework", "PyQt6 / OpenCV"),
        ]
        for r, (k, v) in enumerate(fields):
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
            QTableWidget {{
                background: {BG};
                border: 1px solid {BORDER_LIGHT};
                gridline-color: {BORDER_LIGHT};
                font-family: Consolas;
                font-size: 10px;
            }}
            QHeaderView::section {{
                background: {BG_ALT};
                border: 1px solid {BORDER_LIGHT};
                padding: 4px;
                color: {TEXT_DIM};
            }}
        """)
        modules = [
            ("Hand Tremor", "Diagnostic", "READY"),
            ("Facial Asymmetry", "Diagnostic", "READY"),
            ("rPPG Vitals", "Monitoring", "READY"),
            ("Eye-Tracking AAC", "Assistive", "READY"),
            ("Voice Biomarkers", "Diagnostic", "READY"),
        ]
        for r, (name, typ, st) in enumerate(modules):
            table.setItem(r, 0, QTableWidgetItem(name))
            table.setItem(r, 1, QTableWidgetItem(typ))
            item = QTableWidgetItem(st)
            item.setForeground(QColor(GREEN))
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
            "-->  Diagnostic Output"
        )
        arch_text.setFont(f_mono(9))
        arch_text.setStyleSheet(f"color: {TEXT_DIM}; border: none;")
        arch_text.setWordWrap(True)
        arch_lay.addWidget(arch_text)
        layout.addWidget(arch_panel)

        bottom = QHBoxLayout()
        bottom.setSpacing(12)

        diag_panel, diag_lay = panel("Diagnostic Modules")
        for name, desc in [
            ("TREMOR", "FFT tremor classification: Parkinsonian (4-6 Hz) vs Essential (6-12 Hz)"),
            ("FACIAL", "FAST stroke screening via bilateral landmark distance asymmetry"),
            ("VOICE", "Acoustic biomarkers: F0, Jitter, Shimmer, HNR from sustained phonation"),
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
            ("AAC", "Eye-gaze controlled communication board with dwell-time activation"),
            ("rPPG", "Contactless pulse measurement via green-channel photoplethysmography"),
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
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("HAND TREMOR SCREENING")
        title.setFont(f_sans(13, True))
        title.setStyleSheet(f"color: {TEXT};")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(status_badge("IDLE", TEXT_MUTED))
        layout.addLayout(header)

        sub = QLabel("Butterworth bandpass 1-15 Hz  |  FFT power spectral density  |  Parkinsonian vs Essential classification")
        sub.setFont(f_mono(8))
        sub.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(sub)

        body = QHBoxLayout()
        body.setSpacing(8)

        left = QVBoxLayout()
        left.setSpacing(8)
        left.addWidget(feed_panel("HAND TRACKING FEED"))

        controls = QHBoxLayout()
        controls.setSpacing(6)
        controls.addWidget(cmd_button("START CAPTURE"))
        controls.addWidget(cmd_button("STOP", primary=False))
        controls.addWidget(cmd_button("RESET", primary=False))
        left.addLayout(controls)
        left.addWidget(thin_progress())
        body.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(8)

        m_panel, m_lay = panel_with_status("Analysis", "WAITING", TEXT_MUTED)
        metrics = QGridLayout()
        metrics.setSpacing(2)
        metrics.addWidget(metric_readout("Dominant Frequency", "---", "Hz"), 0, 0)
        metrics.addWidget(metric_readout("Peak Amplitude", "---", "mm"), 0, 1)
        metrics.addWidget(metric_readout("Classification", "---"), 1, 0)
        metrics.addWidget(metric_readout("Confidence", "---", "%"), 1, 1)
        m_lay.addLayout(metrics)
        right.addWidget(m_panel)

        right.addWidget(graph_area("POWER SPECTRAL DENSITY (FFT)"))
        right.addWidget(graph_area("TREMOR WAVEFORM X(t)"))
        body.addLayout(right, 2)

        layout.addLayout(body)


class FacialPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("FACIAL ASYMMETRY / FAST SCREENING")
        title.setFont(f_sans(13, True))
        title.setStyleSheet(f"color: {TEXT};")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(status_badge("IDLE", TEXT_MUTED))
        layout.addLayout(header)

        sub = QLabel("468-point face mesh  |  Bilateral Euclidean distance  |  Asymmetry index computation")
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
        controls.addWidget(cmd_button("CAPTURE RESTING"))
        controls.addWidget(cmd_button("CAPTURE SMILE", primary=False))
        controls.addWidget(cmd_button("RESET", primary=False))
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
        title = QLabel("CONTACTLESS rPPG VITALS")
        title.setFont(f_sans(13, True))
        title.setStyleSheet(f"color: {TEXT};")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(status_badge("IDLE", TEXT_MUTED))
        layout.addLayout(header)

        sub = QLabel("Green-channel ROI extraction  |  Bandpass 0.75-3.0 Hz  |  FFT peak detection  |  BPM estimation")
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
        controls.addWidget(cmd_button("START MEASUREMENT"))
        controls.addWidget(cmd_button("STOP", primary=False))
        left.addLayout(controls)
        left.addWidget(thin_progress())
        body.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(8)

        bpm_panel, bpm_lay = panel_with_status("Heart Rate", "MEASURING", TEXT_MUTED)
        bpm_lay.addWidget(metric_readout_large("BPM", "---", "bpm", NAVY))
        qual_row = QHBoxLayout()
        qual_row.addWidget(metric_readout("Signal Quality", "---"))
        qual_row.addWidget(metric_readout("SpO2 (est.)", "---", "%"))
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
        header.addWidget(status_badge("IDLE", TEXT_MUTED))
        layout.addLayout(header)

        sub = QLabel("Iris landmark tracking  |  Gaze vector estimation  |  Dwell-time activation 0.8-1.0s")
        sub.setFont(f_mono(8))
        sub.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(sub)

        top = QHBoxLayout()
        top.setSpacing(8)

        tracking_panel, tracking_lay = panel_with_status("Eye Tracking", "OFFLINE", TEXT_MUTED)
        tracking_feed = QFrame()
        tracking_feed.setMinimumSize(280, 180)
        tracking_feed.setStyleSheet(f"background: {NAVY}; border: 1px solid {BORDER};")
        tf_lay = QVBoxLayout(tracking_feed)
        tf_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tf_lbl = QLabel("IRIS TRACKING FEED")
        tf_lbl.setFont(f_mono(9))
        tf_lbl.setStyleSheet(f"color: #475569; border: none;")
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
        btn_row.addWidget(cmd_button("SPEAK"))
        btn_row.addWidget(cmd_button("CLEAR", primary=False))
        output_lay.addLayout(btn_row)
        top.addWidget(output_panel, 1)

        layout.addLayout(top)

        grid_panel, grid_lay = panel("Communication Grid")
        grid = QGridLayout()
        grid.setSpacing(4)

        cards = [
            ("WATER", 0, 0), ("DOCTOR", 0, 1), ("PAIN", 0, 2),
            ("YES", 1, 0), ("NO", 1, 1), ("HELP", 1, 2),
        ]
        for text, r, c in cards:
            btn = QPushButton(text)
            btn.setFont(f_sans(12, True))
            btn.setFixedHeight(80)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {BG_ALT};
                    color: {NAVY};
                    border: 1px solid {BORDER};
                    padding: 0 8px;
                }}
                QPushButton:hover {{
                    background: {NAVY};
                    color: #FFFFFF;
                }}
            """)
            grid.addWidget(btn, r, c)

        grid_lay.addLayout(grid)

        nav_row = QHBoxLayout()
        nav_row.addWidget(cmd_button("< PREV CATEGORY", primary=False))
        nav_row.addStretch()
        cat_label = QLabel("CATEGORY 1 / 3: BASIC NEEDS")
        cat_label.setFont(f_mono(9, True))
        cat_label.setStyleSheet(f"color: {TEXT_DIM}; border: none;")
        nav_row.addWidget(cat_label)
        nav_row.addStretch()
        nav_row.addWidget(cmd_button("NEXT CATEGORY >", primary=False))
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
        title = QLabel("VOICE BIOMARKERS / DYSARTHRIA SCREENING")
        title.setFont(f_sans(13, True))
        title.setStyleSheet(f"color: {TEXT};")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(status_badge("IDLE", TEXT_MUTED))
        layout.addLayout(header)

        sub = QLabel("Sustained phonation /a/  |  F0 + Jitter + Shimmer + HNR  |  Spectral autocorrelation")
        sub.setFont(f_mono(8))
        sub.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(sub)

        top = QHBoxLayout()
        top.setSpacing(8)

        rec_panel, rec_lay = panel_with_status("Recording", "STANDBY", AMBER)
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
        rec_lay.addWidget(cmd_button("START RECORDING"))
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
        r_row.addWidget(cmd_button("EXPORT REPORT", primary=False))
        result_lay.addLayout(r_row)
        layout.addWidget(result_panel)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TecnoMedNeuro — Neurological Screening Suite")
        self.setMinimumSize(1200, 750)
        self.resize(1400, 860)
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {BG};
            }}
            * {{
                border-radius: 0px;
            }}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        tabs = ["DASHBOARD", "TREMOR", "FACIAL", "rPPG", "AAC", "VOICE"]
        self.nav = NavBar(tabs, self.switch_page)
        root.addWidget(self.nav)

        self.stack = QStackedWidget()
        self.stack.addWidget(DashboardPage())
        self.stack.addWidget(TremorPage())
        self.stack.addWidget(FacialPage())
        self.stack.addWidget(RPPGPage())
        self.stack.addWidget(AACPage())
        self.stack.addWidget(VoicePage())
        root.addWidget(self.stack)

        self.status_bar = StatusBar()
        root.addWidget(self.status_bar)

        self.switch_page(0)

    def switch_page(self, index):
        self.stack.setCurrentIndex(index)
        self.nav.set_active(index)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
