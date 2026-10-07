from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextDocument
from PyQt6.QtPrintSupport import QPrinter
from PyQt6.QtWidgets import QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget


def protocol_text(results):
    f = results['dominant_freq']
    if results['concentration'] < 50 or results['rms_px'] < 0.2:
        status = 'LOW MEASURED MOTION / INCONCLUSIVE'
        detail = 'Low pixel amplitude or diffuse spectral energy was measured. This does not establish a normal baseline or exclude tremor.'
        action = 'Repeat under consistent conditions if useful. Discuss persistent or concerning symptoms with a qualified clinician.'
    elif f is None:
        status = 'NO CLEAR PEAK / INCONCLUSIVE'
        detail = 'No distinct peak could be resolved in the usable frequency band. This does not rule out tremor or disease.'
        action = 'Check capture quality, lighting, and hand visibility before repeating.'
    elif f < 4:
        status = 'LOW-FREQUENCY MOTION / POSSIBLE DRIFT'
        detail = 'The measured peak is below 4 Hz. Arm movement, fatigue, or tracking artifacts may contribute; frequency alone does not identify the cause.'
        action = 'Support the forearm comfortably on a stable surface, maintain a consistent position, and use steady lighting and frame delivery.'
    elif f < 6.5:
        status = 'MOTION PEAK IN 4.0-6.5 Hz BAND'
        detail = "A peak was measured in this band. Frequency alone cannot identify Parkinson's disease or establish that the motion is resting tremor."
        action = 'Discuss persistent shaking with a qualified clinician for a motor examination. Document posture when repeating; this webcam protocol is not a clinical rest-tremor examination.'
    elif f <= 12.5:
        status = 'MOTION PEAK IN 6.5-12.5 Hz BAND'
        detail = 'A peak was measured in this band. It cannot distinguish essential tremor, physiological motion, or other causes.'
        action = 'Discuss persistent postural shaking with a clinician. Note caffeine, stimulant intake, fatigue, and sleep as context; do not change medication based on this result.'
    else:
        status = 'OUTSIDE EXPLORATORY FREQUENCY BANDS'
        detail = 'The peak is outside the frequency ranges summarized by this research prototype.'
        action = 'Do not assign a clinical category to this result. Discuss concerning symptoms with a qualified clinician.'
    limitation = (
        f"Analyzed band: 1-{results['max_freq']:.1f} Hz. "
        'Frequencies above this range were not assessed. Pixel RMS is not a physical amplitude in millimeters.'
    )
    return status, detail, action, limitation


class ProtocolPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.report = ''
        self.report_name = 'hand_motion_report'
        self.setObjectName('protocolPanel')
        self.setMinimumSize(0, 0)
        self.setStyleSheet('''
            QFrame#protocolPanel { background: #0B132B; border: 1px solid #1E293B; border-radius: 0px; }
            QFrame#protocolPanel QLabel { color: #E2E8F0; background: transparent; border: none; }
            QFrame#protocolPanel QWidget#protocolBody { background: #0B132B; }
            QFrame#protocolPanel QScrollArea { border: none; background: #0B132B; }
            QFrame#protocolPanel QPushButton { color: #BFDBFE; background: #0F172A; border: 1px solid #334155; border-radius: 0px; padding: 3px 6px; }
            QFrame#protocolPanel QPushButton:hover { background: #1E293B; color: #FFFFFF; }
        ''')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        header = QHBoxLayout()
        self.header_layout = header
        title = QLabel('CLINICAL PROTOCOL & SCREENING ASSESSMENT')
        self.title_label = title
        title.setFont(QFont('Consolas', 8, QFont.Weight.Bold))
        title.setWordWrap(True)
        title.setStyleSheet('color: #60A5FA;')
        header.addWidget(title, 1)
        self.export_button = QPushButton('[EXPORT PDF]')
        self.save_button = QPushButton('[SAVE REPORT]')
        for button in (self.export_button, self.save_button):
            button.setFont(QFont('Consolas', 8))
            header.addWidget(button)
        self.close_button = QPushButton('[X]')
        self.close_button.setFont(QFont('Consolas', 8))
        self.close_button.clicked.connect(self.hide)
        self.close_button.hide()
        header.addWidget(self.close_button)
        self.export_button.clicked.connect(lambda: self._export(True))
        self.save_button.clicked.connect(lambda: self._export(False))
        layout.addLayout(header)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumSize(0, 0)
        body = QWidget()
        body.setObjectName('protocolBody')
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 4, 0)
        body_layout.setSpacing(7)
        self.status_label = QLabel()
        self.status_label.setFont(QFont('Consolas', 10, QFont.Weight.Bold))
        self.detail_label = QLabel()
        self.action_label = QLabel()
        self.limitation_label = QLabel()
        for label in (self.status_label, self.detail_label, self.action_label, self.limitation_label):
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.PlainText)
            if label is not self.status_label:
                label.setFont(QFont('Segoe UI', 9))
            body_layout.addWidget(label)
        body_layout.addStretch()
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        self.disclaimer = QLabel('DISCLAIMER: Algorithmic screening prototype. Not a certified clinical diagnosis. For research use only.')
        self.disclaimer.setFont(QFont('Consolas', 7))
        self.disclaimer.setWordWrap(True)
        layout.addWidget(self.disclaimer)
        self.hide()

    def set_results(self, results):
        status, detail, action, limitation = protocol_text(results)
        self.status_label.setText(status)
        self.detail_label.setText(detail)
        self.action_label.setText('PROTOCOL: ' + action)
        self.limitation_label.setText(limitation)
        f = results['dominant_freq']
        frequency = f'{f:.2f} Hz' if f is not None else 'not resolved'
        self.report = '\n\n'.join([
            'TECNOMEDNEURO / HAND MOTION RESEARCH REPORT',
            f"Recorded: {datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"Dominant frequency: {frequency}\nMotion RMS: {results['rms_px']:.3f} px\nPeak concentration: {results['concentration']:.1f}%\nSampling rate: {results['fs']:.1f}/s\nCoverage: {results['coverage'] * 100:.1f}%\nMax frame gap: {results['max_gap_sec']:.3f}s",
            status, detail, 'PROTOCOL: ' + action, limitation, self.disclaimer.text(),
        ])

    def clear(self):
        self.report = ''
        self.hide()

    def _export(self, pdf):
        if not self.report or self.isHidden():
            return
        extension = 'pdf' if pdf else 'txt'
        path, _ = QFileDialog.getSaveFileName(self, 'Export research report', f'{self.report_name}.{extension}',
                                             'PDF (*.pdf)' if pdf else 'Text report (*.txt)')
        if not path:
            return
        output = Path(path)
        if not output.suffix:
            output = output.with_suffix('.' + extension)
        try:
            if pdf:
                printer = QPrinter(QPrinter.PrinterMode.HighResolution)
                printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
                printer.setOutputFileName(str(output))
                document = QTextDocument()
                document.setDefaultFont(QFont('Consolas', 9))
                document.setPlainText(self.report)
                document.print(printer)
                if not output.is_file() or output.stat().st_size == 0:
                    raise OSError('PDF output was not created')
            else:
                output.write_text(self.report, encoding='utf-8')
        except (OSError, RuntimeError) as exc:
            QMessageBox.warning(self, 'Report export failed', str(exc))


def facial_protocol_text(results):
    asi = results['asymmetry_index']
    smile = results['smile_imbalance']
    brow = results['brow_imbalance']
    if asi > 15 or smile > 20:
        if brow < 12:
            status = f'[REVIEW] ELEVATED LOWER-FACE MOVEMENT IMBALANCE (ASI: {asi:.2f}%)'
            detail = 'Elevated geometric error or smile imbalance was measured with less brow imbalance. This does not establish forehead sparing, central paresis, stroke, or TIA.'
        else:
            status = f'[REVIEW] ELEVATED SMILE AND BROW MEASUREMENTS (ASI: {asi:.2f}%)'
            detail = "Elevated facial measurements were recorded. Scalar imbalance scores do not show that deficits affect the same side and cannot identify Bell's palsy or exclude stroke."
        action = (
            'Sudden or newly developed facial weakness requires urgent medical assessment. '
            'For sudden droop, arm weakness, or speech difficulty, call emergency services (103 in Belarus) immediately; do not wait for a repeat scan. '
            'Note when symptoms first appeared or when the person was last known well, and follow emergency-dispatch instructions. '
            'Do not change medication or start treatment based on this report.'
        )
    elif asi < 8 and smile < 12:
        status = f'[LOW MEASURED DEVIATION] FACIAL SYMMETRY OBSERVATIONS (ASI: {asi:.2f}%)'
        detail = 'Geometric error and smile imbalance are below the selected exploratory thresholds. This does not establish normal facial function, a negative FAST result, or the absence of stroke or a facial nerve deficit.'
        action = 'Do not use a low score to dismiss symptoms. Discuss persistent concerns with a qualified clinician. Sudden facial drooping, arm weakness, or speech difficulty requires immediate emergency assessment.'
    else:
        status = f'[RETEST] MODERATE MEASURED DEVIATION (ASI: {asi:.2f}%)'
        detail = 'Measurements fall in the selected moderate observation band. Lighting, pose, landmark noise, or anatomical differences can affect them; the cause is not determined.'
        action = 'Check diffuse frontal lighting and head alignment. Repeat under consistent conditions only if there are no acute symptoms. Discuss persistent or concerning asymmetry with a qualified clinician.'
    limitation = (
        f'ASI: {asi:.2f}%; smile imbalance: {smile:.1f}%; brow imbalance: {brow:.1f}%. '
        'The 8/15% ASI and 12/20% movement bands are unvalidated research thresholds, not clinical cutoffs. '
        'This module observes only the Face component, not the full Face/Arm/Speech/Time protocol.'
    )
    return status, detail, action, limitation


class FacialProtocolPanel(ProtocolPanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.report_name = 'facial_symmetry_report'
        self.title_label.setText('CLINICAL PROTOCOL & FAST SCREENING')
        self.header_layout.removeWidget(self.save_button)
        self.header_layout.insertWidget(1, self.save_button)
        self.close_button.show()
        self.disclaimer.setText('DISCLAIMER: Computational screening utility. Not a clinical diagnosis. Does not replace urgent medical evaluation in case of acute neurological deficits.')

    def set_results(self, results):
        status, detail, action, limitation = facial_protocol_text(results)
        self.status_label.setText(status)
        self.detail_label.setText(detail)
        self.action_label.setText('PROTOCOL: ' + action)
        self.limitation_label.setText(limitation)
        quality = []
        for stage in ('rest', 'smile', 'brows'):
            trial = results[stage]
            quality.append(f"{stage.upper()}: {trial['samples']} accepted frames, {trial['rate']:.1f}/s, reference span {trial['width_px']:.1f}px")
        self.report = '\n\n'.join([
            'TECNOMEDNEURO / FACIAL SYMMETRY RESEARCH REPORT',
            f"Recorded: {datetime.now().astimezone().isoformat(timespec='seconds')}",
            status, detail, 'PROTOCOL: ' + action, limitation,
            '\n'.join(quality), self.disclaimer.text(),
        ])
