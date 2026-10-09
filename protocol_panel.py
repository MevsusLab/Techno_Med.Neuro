from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextDocument
from PyQt6.QtPrintSupport import QPrinter
from PyQt6.QtWidgets import QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget
from i18n_ru import RussianLabel as QLabel, ru


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
        self.export_button = QPushButton(ru('[EXPORT PDF]'))
        self.save_button = QPushButton(ru('[SAVE REPORT]'))
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
        path, _ = QFileDialog.getSaveFileName(self, 'Сохранить исследовательский отчёт', f'{self.report_name}.{extension}',
                                             'PDF (*.pdf)' if pdf else 'Текстовый отчёт (*.txt)')
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
                document.setPlainText(ru(self.report))
                document.print(printer)
                if not output.is_file() or output.stat().st_size == 0:
                    raise OSError('PDF output was not created')
            else:
                output.write_text(ru(self.report), encoding='utf-8')
        except (OSError, RuntimeError) as exc:
            QMessageBox.warning(self, ru('Не удалось сохранить отчёт'), ru(str(exc)))


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


def rppg_protocol_text(results):
    bpm = results['bpm']
    concentration = results['concentration']
    coverage = results['coverage'] * 100
    roi_spread = max(results['roi_bpm']) - min(results['roi_bpm'])
    if coverage < 70 or concentration < 60 or roi_spread > 5:
        status = '[UNRELIABLE] OPTICAL QUALITY / INTERPRETATION WITHHELD'
        detail = (
            'The candidate estimate does not meet the protocol panel\'s stricter optical-quality thresholds. '
            'This is not a clinical heart-rate assessment. Region discrepancy alone does not establish its cause.'
        )
        action = (
            'Support your head comfortably, use constant diffuse frontal lighting, and avoid speaking or moving. '
            'Repeat the 30-second measurement if useful and confirm with a direct pulse measurement. '
            'Do not use this candidate for clinical decisions.'
        )
    elif bpm < 60:
        status = f'[CHECK] LOWER PULSE-RATE ESTIMATE ({bpm:.0f} BPM)'
        detail = (
            'The estimated rate is below 60 BPM. This webcam result cannot diagnose bradycardia, '
            'a conduction disorder, or normal athletic adaptation.'
        )
        action = (
            'Confirm using a direct pulse measurement. Discuss persistent low readings or dizziness, '
            'fainting, or unusual fatigue with a qualified clinician. Review medication questions with '
            'a clinician; do not change medication based on this result.'
        )
    elif bpm <= 95:
        status = f'[OBSERVATION] PULSE ESTIMATE IN 60-95 BPM BAND ({bpm:.0f} BPM)'
        detail = (
            'The candidate falls within the selected observation band and passes the panel\'s quality checks. '
            'It does not establish normal circulation, autonomic balance, or absence of a medical condition.'
        )
        action = 'Compare with a direct pulse measurement under similar resting conditions. Record context and discuss persistent symptoms with a qualified clinician.'
    else:
        status = f'[CHECK] ELEVATED PULSE-RATE ESTIMATE ({bpm:.0f} BPM)'
        detail = (
            'The estimated rate exceeds the panel\'s 95 BPM boundary. This boundary is an exploratory '
            'display rule, not a tachycardia diagnosis or explanation of the cause.'
        )
        action = (
            'If comfortable and symptom-free, rest quietly for 5-10 minutes and confirm with a direct pulse measurement. '
            'Use steady light for any webcam retest. Note recent exercise, stimulants, sleep and illness as context. '
            'Discuss persistent unexplained readings with a qualified clinician.'
        )
    action += ' Seek urgent medical help for chest pain, fainting, severe breathlessness, or other acute concerning symptoms; do not wait for another scan.'
    limitation = (
        f'Spectral concentration: {concentration:.1f}%; valid coverage: {coverage:.1f}%; '
        f'all-ROI spread: {roi_spread:.1f} BPM. '
        'Panel thresholds (<70% coverage, <60% concentration, >5 BPM spread) are unvalidated display rules. '
        'All-ROI spread includes regions excluded by the engine; engine acceptance and protocol interpretation are separate.'
    )
    if results['warnings']:
        limitation += '\nCAPTURE CAUTIONS: ' + ' '.join(results['warnings'])
    return status, detail, action, limitation


class RPPGProtocolPanel(ProtocolPanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.report_name = 'rppg_pulse_report'
        self.title_label.setText('CLINICAL PROTOCOL & VITALS SCREENING')
        self.header_layout.removeWidget(self.save_button)
        self.header_layout.insertWidget(1, self.save_button)
        self.close_button.show()
        self.disclaimer.setText('DISCLAIMER: Remote photoplethysmography (rPPG) research tool. Subject to optical, camera, and lighting artifacts. Not a medical device; does not replace direct ECG or pulse oximetry.')

    def set_results(self, results):
        status, detail, action, limitation = rppg_protocol_text(results)
        self.status_label.setText(status)
        self.detail_label.setText(detail)
        self.action_label.setText('PROTOCOL: ' + action)
        self.limitation_label.setText(limitation)
        used = ', '.join(('forehead', 'right cheek', 'left cheek')[i] for i in results['used_rois'])
        metrics = (
            f"Candidate pulse rate: {results['bpm']:.1f} BPM\n"
            f"Spectral concentration: {results['concentration']:.1f}%\n"
            f"Valid coverage: {results['coverage'] * 100:.1f}%\n"
            f"ROI candidates (forehead / R cheek / L cheek): {' / '.join(f'{bpm:.1f}' for bpm in results['roi_bpm'])} BPM\n"
            f"Used regions: {used}\n"
            f"Samples: {results['samples']}; rate: {results['rate']:.1f}/s; duration: {results['duration']:.1f}s\n"
            f"Largest gap: {results['max_gap']:.3f}s; FFT spacing: {results['resolution_bpm']:.1f} BPM"
        )
        self.report = '\n\n'.join([
            'TECNOMEDNEURO / rPPG PULSE RESEARCH REPORT',
            f"Recorded: {datetime.now().astimezone().isoformat(timespec='seconds')}",
            metrics, status, detail, 'PROTOCOL: ' + action, limitation, self.disclaimer.text(),
        ])


class VoiceProtocolPanel(ProtocolPanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.report_name = 'voice_acoustic_report'
        self.title_label.setText('VOICE STABILITY / RESEARCH REFERENCES')
        self.close_button.show()
        self.disclaimer.setText('DISCLAIMER: Acoustic research prototype. A sustained vowel cannot diagnose dysarthria, dysphonia, or a neurological condition. Recording noise and pitch tracking affect these metrics.')

    def set_results(self, results):
        flagged = results['jitter_local'] > 1.04 or results['shimmer_local'] > 3.81
        status = '[REFERENCE EXCEEDED] PERTURBATION OBSERVATION' if flagged else '[OBSERVATION] ACOUSTIC METRICS AVAILABLE'
        detail = (
            'Local jitter and local shimmer are compared with legacy MDVP reference values quoted in the Praat manual '
            '(1.04% and 3.81%). These method-dependent values are not validated clinical cutoffs for this microphone or app. '
            'Below-reference results do not establish normal speech; above-reference results do not establish disease.'
        )
        action = (
            'Repeat with the same microphone distance and gain in a quiet room if useful. '
            'Discuss persistent voice or speech changes with a qualified clinician or speech-language professional. '
            'Sudden speech difficulty, facial droop, or weakness needs immediate emergency assessment; do not wait for another recording.'
        )
        limitation = (
            f"Analyzed pitch band: 75-500 Hz; voiced frames: {results['voiced_fraction'] * 100:.1f}%; "
            f"recording level: {results['level_dbfs']:.1f} dBFS. HNR has no universal clinical threshold here. "
            + ' '.join(results['warnings'])
        )
        self.status_label.setText(status)
        self.detail_label.setText(detail)
        self.action_label.setText('PROTOCOL: ' + action)
        self.limitation_label.setText(limitation)
        metrics = (
            f"Median F0: {results['f0']:.2f} Hz\nJitter local: {results['jitter_local']:.3f}%\n"
            f"Jitter RAP: {results['jitter_rap']:.3f}%\nShimmer local: {results['shimmer_local']:.3f}%\n"
            f"Shimmer APQ5: {results['shimmer_apq5']:.3f}%\nHNR (Praat autocorrelation mean): {results['hnr']:.2f} dB\n"
            f"Sample rate: {results['sample_rate']} Hz; duration: {results['duration']:.2f}s; "
            f"clipped samples: {results['clipping'] * 100:.3f}%\n"
            'MFCC mean (13 coefficients): ' + ', '.join(f'{x:.3f}' for x in results['mfcc_mean'])
        )
        if results.get('source') == 'file':
            metrics += (
                f"\nSource audio file: {results.get('file_name', '')}"
                f"\nSelected segment starts: {results.get('segment_start_sec', 0):.2f} s"
                f"\nOriginal file duration: {results.get('source_duration_sec', 0):.2f} s"
            )
        else:
            metrics += '\nSource: microphone recording'
        if results.get('source') == 'file':
            metrics += (
                f"\nSource audio file: {results.get('file_name', '')}"
                f"\nSelected segment begins at: {results.get('segment_start_sec', 0):.2f} s"
                f"\nOriginal file duration: {results.get('source_duration_sec', 0):.2f} s"
            )
        self.report = '\n\n'.join([
            'TECNOMEDNEURO / VOICE ACOUSTIC RESEARCH REPORT',
            f"Recorded: {datetime.now().astimezone().isoformat(timespec='seconds')}", metrics,
            'Algorithms: Praat via Parselmouth (pitch, periodic pulses, perturbation, harmonicity); librosa YIN, STFT, MFCC.',
            'Settings: pitch floor 75 Hz, ceiling 500 Hz; period bounds 0.8/500 to 1.25/75 seconds; period factor 1.3; amplitude factor 1.6; first/last 0.3s excluded.',
            status, detail, action, limitation,
            'References: https://praat.org/manual/Voice_2__Jitter.html ; https://praat.org/manual/Voice_3__Shimmer.html',
            self.disclaimer.text(),
        ])
