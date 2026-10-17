# формирование протокола оценки моторики

from motor_engine import motor_band
from protocol_panel import ProtocolPanel, _num, _pct


# дисклеймер
motor_disclaimer = (
    'ДИСКЛЕЙМЕР: Программный расчет на базе компьютерного зрения. Не является медицинским диагнозом '
    'и не заменяет очный осмотр невролога по шкале UPDRS.'
)
MOTOR_DISCLAIMER = motor_disclaimer


def motor_protocol_text(results):
    band = motor_band(results)
    f, dec, cv = results['tap_frequency'], results['amplitude_decrement'], results['rhythm_cv']
    values = (
        f'За {_num(results["duration"], 1)} с зарегистрировано {results["total_taps"]} полных циклов. '
        f'Средний темп — {_num(f)} Гц, изменение амплитуды по тренду — {_pct(dec)}, вариабельность интервалов — {_pct(cv)}.'
    )
    if band == 'stable':
        title = '[НАБЛЮДЕНИЕ] СТАБИЛЬНЫЕ ТЕМП И АМПЛИТУДА'
        analysis = values + ' Темп не ниже 2,8 Гц, снижение амплитуды меньше 15%, CV не выше 20%: выбранные признаки замедления не зарегистрированы в этой пробе.'
        action = 'Для сравнения выполните такую же пробу второй рукой, сохраняя положение и расстояние до камеры.'
    elif band == 'borderline':
        title = '[ПОГРАНИЧНО] СНИЖЕНИЕ ТЕМПА, АМПЛИТУДЫ ИЛИ РИТМИЧНОСТИ'
        reasons = []
        if -25 <= dec <= -15:
            reasons.append('амплитуда снижается на 15–25%')
        if 2 <= f < 2.8:
            reasons.append('темп находится между 2,0 и 2,8 Гц')
        if 20 < cv <= 35:
            reasons.append('CV выше 20%, интервалы менее равномерны')
        analysis = values + ' Пограничные признаки: ' + '; '.join(reasons) + '. Причина по одной записи не определяется; возможен вклад усталости или условий выполнения.'
        action = 'Повторите тест после двух минут отдыха. Сравните результаты второй руки в тех же условиях.'
    else:
        title = '[ВНИМАНИЕ] ВЫРАЖЕННОЕ ЗАМЕДЛЕНИЕ ИЛИ ДЕКРЕМЕНТ / НУЖНА ОЧНАЯ ОЦЕНКА'
        reasons = []
        if dec < -25:
            reasons.append('снижение амплитуды превышает 25%')
        if f < 2:
            reasons.append('темп ниже 2,0 Гц')
        if cv > 35:
            reasons.append('CV выше 35%, интервалы существенно различаются')
        analysis = values + ' Выраженные признаки: ' + '; '.join(reasons) + '. Они требуют проверки, но не устанавливают брадикинезию, её причину или заболевание.'
        action = (
            'Покажите результат неврологу. Для сопоставления сохраните запись второй руки и результаты модуля «ТРЕМОР»; '
            'другие двигательные пробы выполняйте по указанию специалиста.'
        )
    quality = (
        f'Средняя максимальная дистанция — {_num(results["mean_amplitude"])} длины ладони (0–9). '
        f'Принятых кадров: {results["samples"]}; частота {_num(results["rate"], 1)}/с; '
        f'покрытие {_pct(results["coverage"] * 100)}; максимальный интервал {_num(results["max_gap"], 3)} с. '
        'Положительное изменение означает рост амплитуды, отрицательное — снижение. '
        'Пороги 2,0/2,8 Гц, −15/−25% и CV 20/35% — правила наблюдения приложения, не баллы UPDRS 3.4. '
        'Нормирование уменьшает влияние расстояния до камеры, но не устраняет влияние поворота ладони.'
    )
    if results.get('warnings'):
        quality += ' ' + ' '.join(results['warnings'])
    return title, analysis, action, quality


class MotorProtocolPanel(ProtocolPanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.report_name = 'motor_finger_tapping_report'
        self.title_label.setText('ПРОТОКОЛ / МОТОРИКА ПАЛЬЦЕВ (UPDRS 3.4)')
        self.header_layout.removeWidget(self.save_button)
        self.header_layout.insertWidget(1, self.save_button)
        self.close_button.show()

    def set_results(self, results):
        title, analysis, action, quality = motor_protocol_text(results)
        self._show_text(title, analysis, action, quality + '\n\n' + motor_disclaimer)
        hand = {'Left': 'левая', 'Right': 'правая'}.get(results['hand'], results['hand'])
        metrics = '\n'.join([
            f'Рука (метка трекера): {hand}',
            f'Частота тапов: {_num(results["tap_frequency"])} Гц',
            f'Изменение амплитуды: {_pct(results["amplitude_decrement"])}',
            f'Полных циклов: {results["total_taps"]}',
            f'CV межтаповых интервалов: {_pct(results["rhythm_cv"])}',
            f'Средняя пиковая амплитуда: {_num(results["mean_amplitude"])} норм. ед.',
            f'Тренд: {_num(results["initial_amplitude"])} → {_num(results["final_amplitude"])} норм. ед.',
            'Метод: расстояние 4–8 / расстояние 0–9; пики scipy.signal.find_peaks; линейная регрессия максимумов.',
        ])
        source = ('Источник: видеофайл ' + results['source_file'] if results.get('source') == 'video'
                  else 'Источник: веб-камера')
        self.report = self._compose_report(
            'TechnoMedNeuro / МОТОРИКА ПАЛЬЦЕВ / ТЕСТ UPDRS 3.4', metrics,
            title, analysis, quality + '\n\n' + motor_disclaimer, action, source,
        )
        self.assessment_completed.emit(results)
