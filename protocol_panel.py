from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QObject, QEvent, QVariantAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QTextDocument
from PyQt6.QtPrintSupport import QPrinter
from PyQt6.QtWidgets import QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget
from i18n_ru import RussianLabel as QLabel, ru


JITTER_REFERENCE = 1.04
SHIMMER_REFERENCE = 3.81
FACIAL_REGION_NAMES = {'mouth': 'рот', 'eyes': 'глаза', 'brows': 'брови', 'cheeks': 'щёки'}
FACIAL_STAGE_NAMES = {'rest': 'Покой', 'smile': 'Улыбка', 'brows': 'Брови'}
ROI_NAMES = ('лоб', 'правая щека', 'левая щека')
RPPG_URGENT = ' При боли в груди, обмороке или сильной одышке срочно обратитесь за медицинской помощью.'


def _num(value, digits=2):
    return f'{value:.{digits}f}'.replace('.', ',')


def _pct(value, digits=1):
    return f'{_num(value, digits)}%'


def _timestamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def _join(*parts):
    return ' '.join(part for part in parts if part)


def protocol_text(results):
    f = results['dominant_freq']
    rms = results['rms_px']
    concentration = results['concentration']
    max_freq = results['max_freq']
    if rms < 0.2:
        status = 'СЛАБОЕ ДВИЖЕНИЕ / ВЫВОД НЕДОСТАТОЧЕН'
        detail = (
            f'Среднеквадратичное смещение кисти составило {_num(rms)} пикс., что ниже порога 0,2 пикс. '
            'При таком смещении шум камеры сопоставим с движением руки, поэтому ритм в записи не выделяется надёжно. '
            'Отсутствие результата здесь не означает отсутствие движения.'
        )
        action = 'Повторите запись с кистью, вытянутой и неподвижной в кадре, при ровном освещении.'
    elif concentration < 50:
        status = 'НЕВЫРАЖЕННЫЙ РИТМ / ВЫВОД НЕДОСТАТОЧЕН'
        detail = (
            f'Вокруг пика ±1 Гц сосредоточено {_pct(concentration)} мощности спектра при пороге 50%. '
            f'Смещение {_num(rms)} пикс. достаточно для анализа, но энергия распределена по широкой полосе без доминирующего ритма. '
            'Это характерно для нерегулярных движений или нестабильной позы.'
        )
        action = 'Опирайте предплечье на устойчивую поверхность и повторите запись при постоянном освещении.'
    elif f is None:
        status = 'ЧЁТКИЙ ПИК НЕ ВЫДЕЛЕН'
        if results.get('above_band'):
            cause = (
                f'Основная энергия находится выше границы анализа ({_num(max_freq, 1)} Гц), '
                'поэтому пик за пределами диапазона не определяется.'
            )
        else:
            cause = 'Ни один пик в полосе анализа не выделяется над фоновым уровнем спектра.'
        detail = (
            f'{cause} Концентрация мощности составила {_pct(concentration)}. '
            'Отсутствие пика не означает ни наличия, ни отсутствия ритмичного движения.'
        )
        action = 'Повторите запись с полностью видимой кистью и ровным освещением.'
    elif f < 4:
        status = f'НИЗКОЧАСТОТНОЕ ДВИЖЕНИЕ, ПИК {_num(f)} Гц'
        detail = (
            f'Доминирующая частота {_num(f)} Гц ниже 4 Гц: это медленное колебание, а не быстрое дрожание. '
            f'Смещение {_num(rms)} пикс., концентрация {_pct(concentration)}: колебание выражено, но по частоте можно сказать лишь, '
            'что оно медленное, например дрейф положения кисти или утомление предплечья.'
        )
        action = (
            'Опирайте предплечье на стол, сохраняйте положение и повторите запись, '
            'чтобы отделить медленный дрейф от ритмичного дрожания.'
        )
    elif f < 6.5:
        status = f'РИТМИЧНОЕ ДВИЖЕНИЕ 4,0–6,5 Гц, ПИК {_num(f)} Гц'
        detail = (
            f'Доминирующая частота {_num(f)} Гц находится в диапазоне 4,0–6,5 Гц. Ритм выражен: '
            f'смещение {_num(rms)} пикс., концентрация {_pct(concentration)}. '
            'Частота характеризует сам ритм, но не его причину; для интерпретации нужно сопоставить запись с позой, нагрузкой и данными осмотра.'
        )
        action = (
            'Повторите запись в той же позе с вытянутой кистью без опоры и сравните частоту. '
            'Стойкое ритмичное дрожание покажите врачу.'
        )
    elif f <= 12.5:
        status = f'РИТМИЧНОЕ ДВИЖЕНИЕ 6,5–12,5 Гц, ПИК {_num(f)} Гц'
        detail = (
            f'Доминирующая частота {_num(f)} Гц находится в диапазоне 6,5–12,5 Гц, который соответствует дрожанию при удержании позы. '
            f'Смещение {_num(rms)} пикс., концентрация {_pct(concentration)}. '
            'По частоте нельзя определить причину: возможны физиологические и другие факторы.'
        )
        action = (
            'Отметьте кофеин, стимуляторы, усталость и сон перед записью и повторите её в тех же условиях. '
            'Стойкое дрожание покажите врачу.'
        )
    else:
        status = 'ПИК ВНЕ ИССЛЕДОВАТЕЛЬСКИХ ДИАПАЗОНОВ'
        detail = (
            f'Доминирующая частота {_num(f)} Гц выше 12,5 Гц и не входит в диапазоны, которые описывает прототип. '
            'Категория по этому результату не присваивается.'
        )
        action = 'Повторите запись и при необходимости покажите врачу симптомы, которые вас беспокоят.'
    limitation = _join(
        _band_text(results),
        'Смещение дано в пикселях изображения, а не в миллиметрах, и зависит от расстояния до камеры.',
    )
    return status, detail, action, limitation


def _band_text(results):
    max_freq = results['max_freq']
    text = f'Анализируемый диапазон: 1–{_num(max_freq, 1)} Гц.'
    if max_freq >= 11.95:
        return text
    gap = results.get('max_gap_sec')
    fs = results.get('fs')
    if gap is not None and fs is not None and 0.4 / gap <= 0.45 * fs:
        cause = f'пропуском кадров {_num(gap, 3)} с'
    elif fs is not None:
        cause = f'частотой отслеживания {_num(fs, 1)}/с'
    else:
        cause = 'качеством отслеживания'
    return f'{text} Диапазон ограничен {cause}: движения быстрее {_num(max_freq, 1)} Гц не оценивались.'


class ReportReveal(QObject):
    """Animate a dock within its own viewport without moving page controls."""

    def __init__(self, container, panel):
        super().__init__(container)
        self.container = container
        self.panel = panel
        self.fraction = 0.0
        self.animation = QVariantAnimation(self)
        self.animation.setDuration(240)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.animation.valueChanged.connect(self._set_fraction)
        self.animation.finished.connect(lambda: self._set_fraction(.5))
        container.installEventFilter(self)
        panel.installEventFilter(self)

    def _set_fraction(self, fraction):
        self.fraction = float(fraction)
        layout = self.container.layout()
        margins = layout.contentsMargins()
        available = max(0, self.container.height() - margins.top() - margins.bottom()
                        - max(0, layout.spacing()))
        self.panel.setFixedHeight(round(available * self.fraction))

    def reveal(self):
        self.animation.stop()
        self._set_fraction(0.0)
        self.panel.show()
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(.5)
        self.animation.start()

    def _restore(self):
        self.animation.stop()
        self.fraction = 0.0
        self.panel.setMinimumHeight(0)
        self.panel.setMaximumHeight(16777215)

    def eventFilter(self, watched, event):
        if watched is self.panel and event.type() == QEvent.Type.Hide and self.panel.isHidden():
            self._restore()
        elif watched is self.container:
            if event.type() in (QEvent.Type.Resize, QEvent.Type.Show) and not self.panel.isHidden():
                self._set_fraction(self.fraction)
            elif event.type() == QEvent.Type.Hide:
                if self.animation.state() == QVariantAnimation.State.Running:
                    self.animation.stop()
                    self._set_fraction(.5)
        return super().eventFilter(watched, event)


class ProtocolPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.report = ''
        self.report_name = 'hand_motion_report'
        self.reveal_controller = None
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
        self.limitation_label = QLabel()
        self.action_label = QLabel()
        for label in (self.status_label, self.detail_label, self.limitation_label, self.action_label):
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.PlainText)
            if label is not self.status_label:
                label.setFont(QFont('Segoe UI', 9))
            body_layout.addWidget(label)
        body_layout.addStretch()
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        self.hide()

    def animate_in(self, container):
        self.reveal_controller = ReportReveal(container, self)

    def reveal(self):
        if self.reveal_controller is not None:
            self.reveal_controller.reveal()
        else:
            self.show()

    def _show_text(self, status, detail, action, limitation):
        self.status_label.setText(status)
        self.detail_label.setText(detail)
        self.limitation_label.setText(limitation)
        self.action_label.setText(action)

    def _compose_report(self, header, metrics, status, detail, limitation, action, source=''):
        return '\n\n'.join(part for part in [
            header,
            f'Дата и время записи: {_timestamp()}',
            source,
            metrics,
            status, detail, limitation, action,
        ] if part)

    def set_results(self, results):
        status, detail, action, limitation = protocol_text(results)
        self._show_text(status, detail, action, limitation)
        f = results['dominant_freq']
        frequency = f'{_num(f)} Гц' if f is not None else 'не выделена'
        metrics = [
            f'Доминирующая частота: {frequency}',
            f"Среднеквадратичное смещение кисти: {_num(results['rms_px'], 3)} пикс.",
            f"Концентрация мощности у пика (±1 Гц): {_pct(results['concentration'])}",
            f"Частота отслеживания: {_num(results['fs'], 1)}/с",
            f"Покрытие записи: {_pct(results['coverage'] * 100)}",
            f"Наибольший пропуск кадров: {_num(results['max_gap_sec'], 3)} с",
        ]
        if results.get('resolution_hz') is not None:
            metrics.append(f"Разрешение по частоте: {_num(results['resolution_hz'], 3)} Гц")
        if results.get('samples') is not None:
            metrics.append(f"Отсчётов: {results['samples']}")
        source = f"Источник: видеофайл {results['source_file']}" if results.get('source') == 'video' else ''
        self.report = self._compose_report(
            'TECNOMEDNEURO / HAND MOTION RESEARCH REPORT', '\n'.join(metrics),
            status, detail, limitation, action, source,
        )

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


def _dominant_region_sentence(results):
    regions = (results.get('smile') or {}).get('regions')
    if not regions:
        return ''
    name, value = max(regions.items(), key=lambda item: item[1])
    return (
        f'Наибольшее геометрическое отклонение при улыбке — в области «{FACIAL_REGION_NAMES.get(name, name)}»: '
        f'{_pct(value)} от расстояния между внешними уголками глаз.'
    )


FACIAL_URGENT_ACTION = (
    'При внезапной асимметрии лица, слабости руки или нарушении речи немедленно звоните 103 — не ждите повторного теста.'
)


def facial_protocol_text(results):
    asi = results['asymmetry_index']
    smile = results['smile_imbalance']
    brow = results['brow_imbalance']
    region = _dominant_region_sentence(results)
    if asi > 15 or smile > 20:
        if brow < 12:
            status = f'[ПРОВЕРИТЬ] ПОВЫШЕННАЯ АСИММЕТРИЯ НИЖНЕЙ ЧАСТИ ЛИЦА (ИНДЕКС {_pct(asi)})'
            detail = _join(
                f'Индекс асимметрии {_pct(asi)} выше порога 15% или дисбаланс улыбки {_pct(smile)} выше порога 20%. '
                f'Дисбаланс бровей {_pct(brow)} ниже порога 12%: движения лба сопоставимы между сторонами.',
                'Измерение не показывает, с какой стороны асимметрия, и не определяет её причину.',
                region,
            )
        else:
            status = f'[ПРОВЕРИТЬ] ПОВЫШЕННЫЕ ПОКАЗАТЕЛИ УЛЫБКИ И БРОВЕЙ (ИНДЕКС {_pct(asi)})'
            detail = _join(
                f'Дисбаланс улыбки {_pct(smile)} и дисбаланс бровей {_pct(brow)}: асимметрия есть и в нижней части лица, и в движении бровей.',
                'Измерение описывает различие сторон, но не причину, сторону и степень нарушения.',
                region,
            )
        action = FACIAL_URGENT_ACTION + ' Для повторного измерения сидите прямо, смотрите в камеру и используйте ровный рассеянный свет.'
    elif asi < 8 and smile < 12:
        status = f'[НИЗКОЕ ОТКЛОНЕНИЕ] СИММЕТРИЧНЫЕ ПОКАЗАТЕЛИ (ИНДЕКС {_pct(asi)})'
        detail = (
            f'Индекс асимметрии {_pct(asi)} ниже порога 8%, дисбаланс улыбки {_pct(smile)} ниже порога 12%: '
            'движения сторон лица при улыбке и подъёме бровей различаются мало. '
            'Это отсутствие заметной геометрической асимметрии в данном измерении; оно не заменяет осмотр и не подтверждает нормальную функцию лица.'
        )
        action = (
            'Повторите тест при ровном освещении. При внезапных симптомах требуется немедленная экстренная оценка: звоните 103. '
            'Стойкие жалобы обсудите с врачом.'
        )
    else:
        status = f'[ПОВТОРИТЬ] УМЕРЕННОЕ ОТКЛОНЕНИЕ (ИНДЕКС {_pct(asi)})'
        detail = _join(
            f'Индекс асимметрии {_pct(asi)}, дисбаланс улыбки {_pct(smile)} и дисбаланс бровей {_pct(brow)} попадают в промежуточную зону '
            'между порогами наблюдения. Освещение, поза, точность ориентиров или особенности лица могут давать такие значения; '
            'по этой записи причину установить нельзя.',
            region,
        )
        action = (
            'Проверьте равномерность освещения и положение головы, затем повторите тест в тех же условиях. '
            'Стойкую асимметрию обсудите с врачом.'
        )
    limitation = (
        'Пороги наблюдения: индекс асимметрии 8% и 15%, дисбаланс 12% и 20% — непроверенные исследовательские значения, '
        'а не клинические границы. Модуль оценивает только компонент «лицо» теста FAST, без рук, речи и времени.'
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

    def set_results(self, results):
        status, detail, action, limitation = facial_protocol_text(results)
        self._show_text(status, detail, action, limitation)
        metrics = [
            f"Индекс асимметрии: {_pct(results['asymmetry_index'])}",
            f"Дисбаланс улыбки: {_pct(results['smile_imbalance'])}",
            f"Дисбаланс бровей: {_pct(results['brow_imbalance'])}",
        ]
        for stage in ('rest', 'smile', 'brows'):
            trial = results[stage]
            metrics.append(
                f"{FACIAL_STAGE_NAMES[stage]}: {trial['samples']} принятых кадров, {_num(trial['rate'], 1)}/с, "
                f"расстояние между внешними уголками глаз {_num(trial['width_px'], 1)} пикс."
            )
        source = f"Источник: видеофайл {results['source_file']}" if results.get('source') == 'video' else ''
        self.report = self._compose_report(
            'TECNOMEDNEURO / FACIAL SYMMETRY RESEARCH REPORT', '\n'.join(metrics),
            status, detail, limitation, action, source,
        )


def rppg_protocol_text(results):
    bpm = results['bpm']
    concentration = results['concentration']
    coverage = results['coverage'] * 100
    rois = results['roi_bpm']
    spread = max(rois) - min(rois)
    roi_text = ' / '.join(f'{_num(value, 1)}' for value in rois)
    failures = []
    if coverage < 70:
        failures.append(f'покрытие {_pct(coverage, 0)} при минимуме 70%')
    if concentration < 60:
        failures.append(f'концентрация спектра {_pct(concentration)} при минимуме 60%')
    if spread > 5:
        failures.append(f'разброс участков {_num(spread, 1)} уд/мин при максимуме 5 уд/мин')
    quality = (
        f'покрытие {_pct(coverage, 0)}, концентрация спектра {_pct(concentration)}, '
        f'разброс участков {_num(spread, 1)} уд/мин'
    )
    if failures:
        status = '[НЕНАДЁЖНО] КАЧЕСТВО СИГНАЛА НЕДОСТАТОЧНО, ОЦЕНКА НЕ ВЫВОДИТСЯ'
        detail = (
            f'Кандидатное значение {bpm:.0f} уд/мин не прошло проверки качества: {"; ".join(failures)}. '
            f'Участки лоб / правая щека / левая щека дают {roi_text} уд/мин. '
            'Пульсовой сигнал на коже выражен слабо или искажён движением и освещением, поэтому значение нельзя считать результатом.'
        )
        action = 'Сядьте прямо, держите лицо неподвижно при ровном рассеянном свете и повторите 30-секундную запись.'
    elif bpm < 60:
        status = f'[ПРОВЕРИТЬ] ПУЛЬС НИЖЕ 60 УД/МИН: {bpm:.0f}'
        detail = (
            f'Оценка {bpm:.0f} уд/мин ниже 60. Качество записи достаточно: {quality}. '
            'Участки кожи согласованы, поэтому значение отражает общий сигнал, а не один участок. '
            'Камерная оценка не показывает причину низкой частоты.'
        )
        action = 'Проверьте пульс прямым измерением в спокойном состоянии; при стойких низких значениях или головокружении обратитесь к врачу.'
    elif bpm <= 95:
        status = f'[НАБЛЮДЕНИЕ] ОЦЕНКА {bpm:.0f} УД/МИН В ДИАПАЗОНЕ 60–95'
        detail = (
            f'Оценка {bpm:.0f} уд/мин попадает в выбранный диапазон наблюдения и проходит проверки качества: {quality}. '
            'Участки кожи дают согласованный результат. Оценка описывает частоту пульса в момент записи, '
            'но не даёт общей картины состояния сердечно-сосудистой системы.'
        )
        action = 'Сравните с прямым измерением пульса в тех же условиях покоя.'
    else:
        status = f'[ПРОВЕРИТЬ] ПУЛЬС ВЫШЕ 95 УД/МИН: {bpm:.0f}'
        detail = (
            f'Оценка {bpm:.0f} уд/мин выше 95 уд/мин. Качество записи проходит проверки: {quality}. '
            'Частота могла повыситься из-за нагрузки, волнения, кофеина или жары; камерная оценка не определяет причину.'
        )
        action = 'Отдохните 5–10 минут и повторите измерение при ровном свете; сравните с прямым измерением пульса и укажите недавнюю нагрузку.'
    action += RPPG_URGENT
    limitation = _join(
        f'Покрытие {_pct(coverage, 0)}; концентрация спектра {_pct(concentration)}. '
        f'Разброс всех ROI: {_num(spread, 1)} уд/мин (учитываются все три участка, включая исключённые движком).',
        'Пороги панели (покрытие 70%, концентрация 60%, разброс 5 уд/мин) — исследовательские правила отображения, а не клинические границы.',
        f"Разрешение спектра: {_num(results['resolution_bpm'], 1)} уд/мин." if results.get('resolution_bpm') is not None else '',
        ('Предупреждения при записи: ' + ' '.join(results['warnings'])) if results.get('warnings') else '',
    )
    return status, detail, action, limitation


class RPPGProtocolPanel(ProtocolPanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.report_name = 'rppg_pulse_report'
        self.title_label.setText('CLINICAL PROTOCOL & VITALS SCREENING')
        self.header_layout.removeWidget(self.save_button)
        self.header_layout.insertWidget(1, self.save_button)
        self.close_button.show()

    def set_results(self, results):
        status, detail, action, limitation = rppg_protocol_text(results)
        self._show_text(status, detail, action, limitation)
        used = ', '.join(ROI_NAMES[i] for i in results.get('used_rois', []))
        metrics = [
            f"Источник: видеофайл {results['source_file']}" if results.get('source') == 'video' else '',
            f"Предварительная оценка пульса: {_num(results['bpm'], 1)} уд/мин",
            f"Спектральная концентрация: {_pct(results['concentration'])}",
            f"Покрытие: {_pct(results['coverage'] * 100)}",
            f"Участки (лоб / правая щека / левая щека): {' / '.join(_num(v, 1) for v in results['roi_bpm'])} уд/мин",
            f"Использованные участки: {used}" if used else '',
            f"Отсчёты: {results.get('samples', '—')}; частота {_num(results['rate'], 1)}/с; длительность {_num(results['duration'], 1)} с",
            f"Наибольший пропуск: {_num(results['max_gap'], 3)} с; разрешение спектра: {_num(results['resolution_bpm'], 1)} уд/мин",
        ]
        self.report = self._compose_report(
            'TECNOMEDNEURO / rPPG PULSE RESEARCH REPORT', '\n'.join(line for line in metrics if line),
            status, detail, limitation, action,
        )


def voice_protocol_text(results):
    f0 = results['f0']
    jitter = results['jitter_local']
    shimmer = results['shimmer_local']
    hnr = results['hnr']
    voiced = results['voiced_fraction'] * 100
    jitter_high = jitter > JITTER_REFERENCE
    shimmer_high = shimmer > SHIMMER_REFERENCE
    if jitter_high and shimmer_high:
        status = '[ПРЕВЫШЕН ОРИЕНТИР] ПОВЫШЕНЫ ДЖИТТЕР И ШИММЕР'
        pattern = 'Превышены оба ориентира: периоды голоса колеблются и по длительности, и по амплитуде сильнее эталонных значений.'
        action = (
            'Повторите запись в тихой комнате на том же расстоянии до микрофона и с тем же усилением. '
            'Стойкие изменения голоса или речи обсудите с врачом или логопедом; при внезапном нарушении речи или слабости звоните 103.'
        )
    elif jitter_high:
        status = '[ПРЕВЫШЕН ОРИЕНТИР] ПОВЫШЕН ДЖИТТЕР'
        pattern = 'Превышен только джиттер: длительность периодов колеблется сильнее эталона, амплитуда в пределах ориентира.'
        action = (
            'Повторите запись в тихой комнате на том же расстоянии до микрофона и с тем же усилением. '
            'Стойкие изменения голоса или речи обсудите с врачом или логопедом; при внезапном нарушении речи или слабости звоните 103.'
        )
    elif shimmer_high:
        status = '[ПРЕВЫШЕН ОРИЕНТИР] ПОВЫШЕН ШИММЕР'
        pattern = 'Превышен только шиммер: амплитуда периодов колеблется сильнее эталона, длительность периодов в пределах ориентира.'
        action = (
            'Повторите запись в тихой комнате на том же расстоянии до микрофона и с тем же усилением. '
            'Стойкие изменения голоса или речи обсудите с врачом или логопедом; при внезапном нарушении речи или слабости звоните 103.'
        )
    else:
        status = '[НАБЛЮДЕНИЕ] ПОКАЗАТЕЛИ В ПРЕДЕЛАХ ОРИЕНТИРОВ'
        pattern = 'Оба показателя ниже ориентиров: периоды голоса и их амплитуда колеблются в пределах, принятых для этой методики.'
        action = 'Для проверки стабильности повторите запись в тех же условиях; стойкие изменения голоса или речи обсудите с врачом или логопедом.'
    jitter_word = 'выше' if jitter_high else 'не выше'
    shimmer_word = 'выше' if shimmer_high else 'не выше'
    detail = _join(
        pattern,
        f'Основной тон (медиана F0) — {_num(f0, 1)} Гц.',
        f'Локальный джиттер (колебание длительности периодов) — {_pct(jitter, 2)}; ориентир {_num(JITTER_REFERENCE, 2)}%: {jitter_word}.',
        f'Локальный шиммер (колебание амплитуды периодов) — {_pct(shimmer, 2)}; ориентир {_num(SHIMMER_REFERENCE, 2)}%: {shimmer_word}.',
        f'Гармоничность (HNR) — {_num(hnr, 1)} дБ: чем выше значение, тем больше периодической составляющей и меньше шума в звуке.',
        f'Периодические участки найдены в {_pct(voiced, 0)} кадров.',
        'Ориентиры MDVP относятся к классической методике Praat и зависят от микрофона и условий записи; '
        'превышение означает необходимость повторной записи в тех же условиях, а не диагноз.',
    )
    limitation = _join(
        f"Диапазон высоты голоса при анализе: 75–500 Гц. Проанализировано {_num(results['duration'], 2)} с при частоте дискретизации "
        f"{results['sample_rate']} Гц; уровень записи {_num(results['level_dbfs'], 1)} дБFS; "
        f"пересвеченных отсчётов {_pct(results['clipping'] * 100, 3)}.",
        'Для HNR здесь нет универсального клинического порога.',
        ' '.join(results.get('warnings', [])),
    )
    return status, detail, action, limitation


class VoiceProtocolPanel(ProtocolPanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.report_name = 'voice_acoustic_report'
        self.title_label.setText('VOICE STABILITY / RESEARCH REFERENCES')
        self.close_button.show()

    def set_results(self, results):
        status, detail, action, limitation = voice_protocol_text(results)
        self._show_text(status, detail, action, limitation)
        metrics = (
            f"Медиана F0: {_num(results['f0'], 2)} Гц\n"
            f"Джиттер локальный: {_pct(results['jitter_local'], 3)}\n"
            f"Джиттер RAP: {_pct(results['jitter_rap'], 3)}\n"
            f"Шиммер локальный: {_pct(results['shimmer_local'], 3)}\n"
            f"Шиммер APQ5: {_pct(results['shimmer_apq5'], 3)}\n"
            f"HNR (среднее по Praat): {_num(results['hnr'], 2)} дБ\n"
            f"Частота дискретизации: {results['sample_rate']} Гц; длительность: {_num(results['duration'], 2)} с; "
            f"отсчётов с клиппингом: {_pct(results['clipping'] * 100, 3)}\n"
            'MFCC (среднее, 13 коэффициентов): ' + ', '.join(_num(x, 3) for x in results['mfcc_mean'])
        )
        if results.get('source') == 'file':
            metrics += (
                f"\nИсходный аудиофайл: {results.get('file_name', '')}"
                f"\nНачало выбранного фрагмента: {_num(results.get('segment_start_sec', 0), 2)} с"
                f"\nДлительность исходного файла: {_num(results.get('source_duration_sec', 0), 2)} с"
            )
        elif results.get('source') == 'video':
            metrics += (
                f"\nВидео: {results.get('file_name', '')}"
                f"\nНачало выбранного аудиофрагмента: {_num(results.get('segment_start_sec', 0), 2)} с"
                f"\nДлительность исходного видео: {_num(results.get('source_duration_sec', 0), 2)} с"
            )
        else:
            metrics += '\nИсточник: запись с микрофона'
        self.report = '\n\n'.join([
            'TECNOMEDNEURO / VOICE ACOUSTIC RESEARCH REPORT',
            f'Дата и время записи: {_timestamp()}',
            metrics,
            'Алгоритмы: Praat через Parselmouth (высота тона, периоды, вариации, гармоничность); librosa YIN, STFT, MFCC.',
            'Настройки: диапазон F0 75–500 Гц; границы периодов 0,8/500 и 1,25/75 с; коэффициент периодов 1,3; коэффициент амплитуды 1,6; первые и последние 0,3 с исключены.',
            'Справочные материалы Praat: https://praat.org/manual/Voice_2__Jitter.html ; https://praat.org/manual/Voice_3__Shimmer.html',
            status, detail, limitation, action,
        ])
