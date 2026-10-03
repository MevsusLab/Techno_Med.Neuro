import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFrame, QGridLayout,
    QSizePolicy, QGraphicsDropShadowEffect, QProgressBar, QSpacerItem
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QColor, QIcon


ACCENT = "#3B82F6"
ACCENT_HOVER = "#2563EB"
ACCENT_LIGHT = "#EFF6FF"
BG = "#F8FAFC"
SIDEBAR_BG = "#FFFFFF"
CARD_BG = "#FFFFFF"
TEXT = "#1E293B"
TEXT_SECONDARY = "#64748B"
BORDER = "#E2E8F0"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"


def font(size=13, bold=False):
    f = QFont("Segoe UI", size)
    f.setBold(bold)
    return f


def shadow(parent, blur=20, offset=2, opacity=40):
    s = QGraphicsDropShadowEffect(parent)
    s.setBlurRadius(blur)
    s.setOffset(0, offset)
    s.setColor(QColor(0, 0, 0, opacity))
    return s


def card(title="", parent=None):
    frame = QFrame(parent)
    frame.setStyleSheet(f"""
        QFrame {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 12px;
        }}
    """)
    frame.setGraphicsEffect(shadow(frame))
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(12)
    if title:
        lbl = QLabel(title)
        lbl.setFont(font(14, True))
        lbl.setStyleSheet(f"color: {TEXT}; border: none;")
        layout.addWidget(lbl)
    return frame, layout


def camera_placeholder():
    frame = QFrame()
    frame.setMinimumSize(480, 360)
    frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    frame.setStyleSheet(f"""
        QFrame {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #F1F5F9, stop:1 #E2E8F0);
            border: 2px dashed {BORDER};
            border-radius: 12px;
        }}
    """)
    lay = QVBoxLayout(frame)
    icon = QLabel("📷")
    icon.setFont(font(32))
    icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
    icon.setStyleSheet("border: none;")
    lay.addWidget(icon)
    txt = QLabel("Camera Feed")
    txt.setFont(font(13))
    txt.setAlignment(Qt.AlignmentFlag.AlignCenter)
    txt.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
    lay.addWidget(txt)
    lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return frame


def metric_box(label, value, unit="", color=ACCENT):
    frame = QFrame()
    frame.setFixedHeight(90)
    frame.setStyleSheet(f"""
        QFrame {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 10px;
        }}
    """)
    frame.setGraphicsEffect(shadow(frame, blur=12, offset=1, opacity=25))
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(14, 10, 14, 10)
    lay.setSpacing(2)
    lbl = QLabel(label)
    lbl.setFont(font(10))
    lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
    lay.addWidget(lbl)
    val = QLabel(f"{value} {unit}".strip())
    val.setFont(font(22, True))
    val.setStyleSheet(f"color: {color}; border: none;")
    lay.addWidget(val)
    return frame


def graph_placeholder(title="Spectrum"):
    frame = QFrame()
    frame.setMinimumHeight(200)
    frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    frame.setStyleSheet(f"""
        QFrame {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #FAFBFC, stop:1 #F1F5F9);
            border: 1px solid {BORDER};
            border-radius: 10px;
        }}
    """)
    lay = QVBoxLayout(frame)
    lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
    icon = QLabel("📊")
    icon.setFont(font(28))
    icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
    icon.setStyleSheet("border: none;")
    lay.addWidget(icon)
    lbl = QLabel(title)
    lbl.setFont(font(11))
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
    lay.addWidget(lbl)
    return frame


def action_button(text, primary=True):
    btn = QPushButton(text)
    btn.setFont(font(12, True))
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedHeight(42)
    if primary:
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {ACCENT};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0 24px;
            }}
            QPushButton:hover {{
                background: {ACCENT_HOVER};
            }}
        """)
    else:
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {ACCENT_LIGHT};
                color: {ACCENT};
                border: 1px solid {ACCENT};
                border-radius: 8px;
                padding: 0 24px;
            }}
            QPushButton:hover {{
                background: {ACCENT};
                color: white;
            }}
        """)
    return btn


class SidebarButton(QPushButton):
    def __init__(self, icon_text, label, parent=None):
        super().__init__(parent)
        self.icon_text = icon_text
        self.label = label
        self.setText(f"  {icon_text}   {label}")
        self.setFont(font(12))
        self.setFixedHeight(48)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setCheckable(True)
        self.setStyleSheet(self._style(False))
        self.toggled.connect(lambda checked: self.setStyleSheet(self._style(checked)))

    def _style(self, active):
        if active:
            return f"""
                QPushButton {{
                    background: {ACCENT_LIGHT};
                    color: {ACCENT};
                    border: none;
                    border-radius: 10px;
                    text-align: left;
                    padding-left: 16px;
                    font-weight: bold;
                }}
            """
        return f"""
            QPushButton {{
                background: transparent;
                color: {TEXT_SECONDARY};
                border: none;
                border-radius: 10px;
                text-align: left;
                padding-left: 16px;
            }}
            QPushButton:hover {{
                background: #F1F5F9;
                color: {TEXT};
            }}
        """


class DashboardPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)

        header = QLabel("Dashboard")
        header.setFont(font(24, True))
        header.setStyleSheet(f"color: {TEXT};")
        layout.addWidget(header)

        subtitle = QLabel("TecnoMedNeuro — Webcam-Based Neurological Screening Suite")
        subtitle.setFont(font(13))
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY};")
        layout.addWidget(subtitle)

        grid = QGridLayout()
        grid.setSpacing(16)

        modules = [
            ("🤚", "Hand Tremor", "FFT-based tremor classification", "Parkinsonian vs Essential"),
            ("😐", "Facial Asymmetry", "FAST test for stroke screening", "468-point mesh analysis"),
            ("💓", "rPPG Vitals", "Contactless heart rate", "Green-channel spectroscopy"),
            ("👁", "Eye-Tracking AAC", "Gaze-controlled communication", "Iris landmark tracking"),
            ("🎙", "Voice Biomarkers", "Dysarthria screening", "F₀, Jitter, Shimmer, HNR"),
        ]

        for i, (icon, title, desc, tech) in enumerate(modules):
            c, cl = card()
            c.setFixedHeight(180)

            icon_lbl = QLabel(icon)
            icon_lbl.setFont(font(28))
            icon_lbl.setStyleSheet("border: none;")
            cl.addWidget(icon_lbl)

            t = QLabel(title)
            t.setFont(font(14, True))
            t.setStyleSheet(f"color: {TEXT}; border: none;")
            cl.addWidget(t)

            d = QLabel(desc)
            d.setFont(font(11))
            d.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
            d.setWordWrap(True)
            cl.addWidget(d)

            tag = QLabel(tech)
            tag.setFont(font(9))
            tag.setStyleSheet(f"""
                color: {ACCENT};
                background: {ACCENT_LIGHT};
                border: none;
                border-radius: 4px;
                padding: 2px 8px;
            """)
            tag.setFixedHeight(22)
            cl.addWidget(tag)

            cl.addStretch()
            row, col = divmod(i, 3)
            grid.addWidget(c, row, col)

        layout.addLayout(grid)

        status_card, status_lay = card("System Status")
        status_row = QHBoxLayout()
        for name, state, color in [
            ("Camera", "Ready", SUCCESS),
            ("Microphone", "Ready", SUCCESS),
            ("MediaPipe", "Not loaded", WARNING),
            ("DSP Engine", "Standby", TEXT_SECONDARY),
        ]:
            box = metric_box(name, state, color=color)
            status_row.addWidget(box)
        status_lay.addLayout(status_row)
        layout.addWidget(status_card)
        layout.addStretch()


class TremorPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        header = QLabel("Hand Tremor Screening")
        header.setFont(font(22, True))
        header.setStyleSheet(f"color: {TEXT};")
        layout.addWidget(header)

        desc = QLabel("Hold your outstretched palm in front of the camera for 15 seconds")
        desc.setFont(font(12))
        desc.setStyleSheet(f"color: {TEXT_SECONDARY};")
        layout.addWidget(desc)

        content = QHBoxLayout()
        content.setSpacing(20)

        left = QVBoxLayout()
        left.setSpacing(16)
        left.addWidget(camera_placeholder())

        btn_row = QHBoxLayout()
        btn_row.addWidget(action_button("▶  Start Recording"))
        btn_row.addWidget(action_button("⏹  Stop", primary=False))
        left.addLayout(btn_row)

        progress = QProgressBar()
        progress.setFixedHeight(6)
        progress.setValue(0)
        progress.setTextVisible(False)
        progress.setStyleSheet(f"""
            QProgressBar {{
                background: {BORDER};
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: {ACCENT};
                border-radius: 3px;
            }}
        """)
        left.addWidget(progress)
        content.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(16)

        metrics_card, metrics_lay = card("Analysis Results")
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(10)
        metrics_grid.addWidget(metric_box("Dominant Freq", "—", "Hz"), 0, 0)
        metrics_grid.addWidget(metric_box("Amplitude", "—", "mm"), 0, 1)
        metrics_grid.addWidget(metric_box("Classification", "—", color=TEXT_SECONDARY), 1, 0)
        metrics_grid.addWidget(metric_box("Confidence", "—", "%"), 1, 1)
        metrics_lay.addLayout(metrics_grid)
        right.addWidget(metrics_card)

        right.addWidget(graph_placeholder("Power Spectral Density (FFT)"))
        right.addWidget(graph_placeholder("Tremor Waveform x(t)"))
        content.addLayout(right, 2)

        layout.addLayout(content)


class FacialPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        header = QLabel("Facial Asymmetry & FAST Screening")
        header.setFont(font(22, True))
        header.setStyleSheet(f"color: {TEXT};")
        layout.addWidget(header)

        desc = QLabel("Look directly at the camera, then smile when prompted")
        desc.setFont(font(12))
        desc.setStyleSheet(f"color: {TEXT_SECONDARY};")
        layout.addWidget(desc)

        content = QHBoxLayout()
        content.setSpacing(20)

        left = QVBoxLayout()
        left.setSpacing(16)
        left.addWidget(camera_placeholder())

        btn_row = QHBoxLayout()
        btn_row.addWidget(action_button("📸  Capture Resting"))
        btn_row.addWidget(action_button("😊  Capture Smile", primary=False))
        left.addLayout(btn_row)
        content.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(16)

        idx_card, idx_lay = card("Asymmetry Index")
        big_val = QLabel("— %")
        big_val.setFont(font(48, True))
        big_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        big_val.setStyleSheet(f"color: {ACCENT}; border: none;")
        idx_lay.addWidget(big_val)
        status_lbl = QLabel("Waiting for capture...")
        status_lbl.setFont(font(12))
        status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
        idx_lay.addWidget(status_lbl)
        right.addWidget(idx_card)

        metrics_row = QGridLayout()
        metrics_row.setSpacing(10)
        metrics_row.addWidget(metric_box("Mouth Δ", "—", "px"), 0, 0)
        metrics_row.addWidget(metric_box("Eye Δ", "—", "px"), 0, 1)
        metrics_row.addWidget(metric_box("Brow Δ", "—", "px"), 1, 0)
        metrics_row.addWidget(metric_box("Nasolabial Δ", "—", "px"), 1, 1)
        right.addLayout(metrics_row)

        right.addWidget(graph_placeholder("Deviation Heatmap"))
        content.addLayout(right, 2)

        layout.addLayout(content)


class RPPGPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        header = QLabel("Contactless rPPG Vitals")
        header.setFont(font(22, True))
        header.setStyleSheet(f"color: {TEXT};")
        layout.addWidget(header)

        desc = QLabel("Sit still and face the camera — measuring pulse from skin micro-fluctuations")
        desc.setFont(font(12))
        desc.setStyleSheet(f"color: {TEXT_SECONDARY};")
        layout.addWidget(desc)

        content = QHBoxLayout()
        content.setSpacing(20)

        left = QVBoxLayout()
        left.setSpacing(16)
        left.addWidget(camera_placeholder())
        left.addWidget(action_button("💓  Start Measurement"))
        content.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(16)

        bpm_card, bpm_lay = card("Heart Rate")
        bpm_val = QLabel("— BPM")
        bpm_val.setFont(font(52, True))
        bpm_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bpm_val.setStyleSheet(f"color: {DANGER}; border: none;")
        bpm_lay.addWidget(bpm_val)
        bpm_status = QLabel("Place face in frame and hold still")
        bpm_status.setFont(font(11))
        bpm_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bpm_status.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
        bpm_lay.addWidget(bpm_status)
        right.addWidget(bpm_card)

        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(10)
        metrics_row.addWidget(metric_box("Signal Quality", "—", color=TEXT_SECONDARY))
        metrics_row.addWidget(metric_box("SpO₂ (est.)", "—", "%", color=ACCENT))
        right.addLayout(metrics_row)

        right.addWidget(graph_placeholder("Pulse Waveform (Green Channel)"))
        right.addWidget(graph_placeholder("FFT Spectrum — BPM Peak"))
        content.addLayout(right, 2)

        layout.addLayout(content)


class AACPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        header = QLabel("Eye-Tracking AAC Keyboard")
        header.setFont(font(22, True))
        header.setStyleSheet(f"color: {TEXT};")
        layout.addWidget(header)

        desc = QLabel("Look at a card for 1 second to select — gaze-controlled communication for paralyzed patients")
        desc.setFont(font(12))
        desc.setStyleSheet(f"color: {TEXT_SECONDARY};")
        layout.addWidget(desc)

        top = QHBoxLayout()
        top.setSpacing(16)

        cam_card, cam_lay = card("Eye Tracking Feed")
        cam_placeholder = QFrame()
        cam_placeholder.setMinimumSize(320, 240)
        cam_placeholder.setStyleSheet(f"""
            QFrame {{
                background: #F1F5F9;
                border: 2px dashed {BORDER};
                border-radius: 10px;
            }}
        """)
        cam_inner = QVBoxLayout(cam_placeholder)
        cam_inner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        eye_icon = QLabel("👁")
        eye_icon.setFont(font(24))
        eye_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        eye_icon.setStyleSheet("border: none;")
        cam_inner.addWidget(eye_icon)
        cam_lay.addWidget(cam_placeholder)

        gaze_row = QHBoxLayout()
        gaze_row.addWidget(metric_box("Gaze X", "—", color=ACCENT))
        gaze_row.addWidget(metric_box("Gaze Y", "—", color=ACCENT))
        gaze_row.addWidget(metric_box("Dwell", "—", "ms", color=WARNING))
        cam_lay.addLayout(gaze_row)
        top.addWidget(cam_card, 2)

        status_card, status_lay = card("Status")
        selected_lbl = QLabel("Selected:")
        selected_lbl.setFont(font(12))
        selected_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
        status_lay.addWidget(selected_lbl)
        selected_val = QLabel("—")
        selected_val.setFont(font(28, True))
        selected_val.setStyleSheet(f"color: {ACCENT}; border: none;")
        selected_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_lay.addWidget(selected_val)
        status_lay.addStretch()
        btn_row = QHBoxLayout()
        btn_row.addWidget(action_button("🔊  Speak"))
        btn_row.addWidget(action_button("🗑  Clear", primary=False))
        status_lay.addLayout(btn_row)
        top.addWidget(status_card, 1)

        layout.addLayout(top)

        grid = QGridLayout()
        grid.setSpacing(12)

        aac_cards = [
            ("💧", "Water", "#DBEAFE"),
            ("🩺", "Doctor", "#DCFCE7"),
            ("😣", "Pain", "#FEE2E2"),
            ("✅", "Yes", "#D1FAE5"),
            ("❌", "No", "#FEE2E2"),
            ("🆘", "Help", "#FEF3C7"),
        ]

        for i, (emoji, text, bg) in enumerate(aac_cards):
            btn = QPushButton(f"{emoji}\n{text}")
            btn.setFont(font(18, True))
            btn.setFixedSize(180, 140)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {bg};
                    color: {TEXT};
                    border: 2px solid {BORDER};
                    border-radius: 16px;
                }}
                QPushButton:hover {{
                    border-color: {ACCENT};
                    background: {ACCENT_LIGHT};
                }}
            """)
            row, col = divmod(i, 3)
            grid.addWidget(btn, row, col, Qt.AlignmentFlag.AlignCenter)

        nav_row = QHBoxLayout()
        for arrow, label in [("◀", "Previous"), ("▶", "Next")]:
            btn = QPushButton(f"{arrow}  {label}")
            btn.setFont(font(14, True))
            btn.setFixedSize(200, 60)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: #F1F5F9;
                    color: {TEXT};
                    border: 2px solid {BORDER};
                    border-radius: 12px;
                }}
                QPushButton:hover {{
                    border-color: {ACCENT};
                    color: {ACCENT};
                }}
            """)
            nav_row.addWidget(btn)

        layout.addLayout(grid)
        layout.addLayout(nav_row)
        layout.addStretch()


class VoicePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        header = QLabel("Voice Biomarkers & Dysarthria")
        header.setFont(font(22, True))
        header.setStyleSheet(f"color: {TEXT};")
        layout.addWidget(header)

        desc = QLabel("Sustain the vowel 'A-a-a' for 5 seconds into your microphone")
        desc.setFont(font(12))
        desc.setStyleSheet(f"color: {TEXT_SECONDARY};")
        layout.addWidget(desc)

        top = QHBoxLayout()
        top.setSpacing(20)

        rec_card, rec_lay = card("Recording")
        mic_icon = QLabel("🎙")
        mic_icon.setFont(font(48))
        mic_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mic_icon.setStyleSheet("border: none;")
        rec_lay.addWidget(mic_icon)

        timer_lbl = QLabel("0.0 / 5.0 sec")
        timer_lbl.setFont(font(16, True))
        timer_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        timer_lbl.setStyleSheet(f"color: {TEXT}; border: none;")
        rec_lay.addWidget(timer_lbl)

        progress = QProgressBar()
        progress.setFixedHeight(8)
        progress.setValue(0)
        progress.setTextVisible(False)
        progress.setStyleSheet(f"""
            QProgressBar {{
                background: {BORDER};
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: {DANGER};
                border-radius: 4px;
            }}
        """)
        rec_lay.addWidget(progress)
        rec_lay.addStretch()
        rec_lay.addWidget(action_button("🎤  Start Recording"))
        top.addWidget(rec_card, 1)

        metrics_card, metrics_lay = card("Acoustic Metrics")
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(10)
        metrics_grid.addWidget(metric_box("F₀ (Pitch)", "—", "Hz", ACCENT), 0, 0)
        metrics_grid.addWidget(metric_box("Jitter", "—", "%", WARNING), 0, 1)
        metrics_grid.addWidget(metric_box("Shimmer", "—", "%", WARNING), 1, 0)
        metrics_grid.addWidget(metric_box("HNR", "—", "dB", SUCCESS), 1, 1)
        metrics_lay.addLayout(metrics_grid)
        top.addWidget(metrics_card, 1)

        layout.addLayout(top)

        graphs = QHBoxLayout()
        graphs.setSpacing(16)
        graphs.addWidget(graph_placeholder("Audio Waveform"))
        graphs.addWidget(graph_placeholder("Frequency Spectrum (FFT)"))
        graphs.addWidget(graph_placeholder("MFCC Heatmap"))
        layout.addLayout(graphs)

        result_card, result_lay = card("Assessment")
        result_row = QHBoxLayout()
        result_lbl = QLabel("Vocal stability assessment will appear here after recording")
        result_lbl.setFont(font(12))
        result_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
        result_row.addWidget(result_lbl)
        result_row.addStretch()
        result_row.addWidget(action_button("📄  Export Report", primary=False))
        result_lay.addLayout(result_row)
        layout.addWidget(result_card)

        layout.addStretch()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TecnoMedNeuro")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)
        self.setStyleSheet(f"QMainWindow {{ background: {BG}; }}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QFrame()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet(f"""
            QFrame {{
                background: {SIDEBAR_BG};
                border-right: 1px solid {BORDER};
            }}
        """)
        sidebar_lay = QVBoxLayout(sidebar)
        sidebar_lay.setContentsMargins(16, 20, 16, 20)
        sidebar_lay.setSpacing(6)

        logo_row = QHBoxLayout()
        logo_icon = QLabel("🧠")
        logo_icon.setFont(font(22))
        logo_icon.setStyleSheet("border: none;")
        logo_row.addWidget(logo_icon)
        logo_text = QLabel("TecnoMedNeuro")
        logo_text.setFont(font(16, True))
        logo_text.setStyleSheet(f"color: {TEXT}; border: none;")
        logo_row.addWidget(logo_text)
        logo_row.addStretch()
        sidebar_lay.addLayout(logo_row)

        version = QLabel("v0.1 — Screening Suite")
        version.setFont(font(9))
        version.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
        sidebar_lay.addWidget(version)
        sidebar_lay.addSpacing(24)

        section = QLabel("NAVIGATION")
        section.setFont(font(9, True))
        section.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none; padding-left: 16px;")
        sidebar_lay.addWidget(section)
        sidebar_lay.addSpacing(4)

        self.stack = QStackedWidget()
        self.buttons = []

        pages = [
            ("📊", "Dashboard", DashboardPage()),
            ("🤚", "Hand Tremor", TremorPage()),
            ("😐", "Facial Asymmetry", FacialPage()),
            ("💓", "rPPG Vitals", RPPGPage()),
            ("👁", "Eye-Tracking AAC", AACPage()),
            ("🎙", "Voice Biomarkers", VoicePage()),
        ]

        for i, (icon, label, page) in enumerate(pages):
            btn = SidebarButton(icon, label)
            btn.clicked.connect(lambda checked, idx=i: self.switch_page(idx))
            sidebar_lay.addWidget(btn)
            self.buttons.append(btn)
            self.stack.addWidget(page)

        sidebar_lay.addStretch()

        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background: {BORDER}; border: none;")
        sidebar_lay.addWidget(divider)
        sidebar_lay.addSpacing(8)

        footer_label = QLabel("MevsusLab × NDTP")
        footer_label.setFont(font(9))
        footer_label.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
        footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_lay.addWidget(footer_label)

        root.addWidget(sidebar)

        content_area = QWidget()
        content_lay = QVBoxLayout(content_area)
        content_lay.setContentsMargins(32, 28, 32, 28)
        content_lay.addWidget(self.stack)
        root.addWidget(content_area)

        self.switch_page(0)

    def switch_page(self, index):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.buttons):
            btn.setChecked(i == index)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
