"""Finger-tapping kinematics; observation thresholds are not UPDRS scores."""

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def finger_separation(points):
    """Image-plane tip distance in palm units, using pixel-space landmarks."""
    points = np.asarray(points, dtype=float)
    if points.shape != (21, 2) or not np.isfinite(points).all():
        raise ValueError('Ориентиры кисти отсутствуют или некорректны.')
    palm = float(np.linalg.norm(points[0] - points[9]))
    if palm < 25:
        raise ValueError('Приблизьте кисть: расстояние запястье–основание среднего пальца меньше 25 пикселей.')
    distance = float(np.linalg.norm(points[4] - points[8]) / palm)
    if distance > 4:
        raise ValueError('Геометрия кисти нестабильна: держите ладонь лицом к камере.')
    return distance, palm


def motor_band(results):
    """Severe flags win; the unspecified 20–35% CV band is borderline."""
    if results['amplitude_decrement'] < -25 or results['tap_frequency'] < 2 or results['rhythm_cv'] > 35:
        return 'marked'
    if (results['amplitude_decrement'] > -15 and results['tap_frequency'] >= 2.8
            and results['rhythm_cv'] <= 20):
        return 'stable'
    return 'borderline'


def analyze_tapping(samples, record_start, record_end):
    if record_start is None or record_end is None or not np.isfinite([record_start, record_end]).all():
        raise ValueError('Временные границы записи отсутствуют.')
    duration = record_end - record_start
    if duration < 9.9 or duration > 15.1 or len(samples) < 200:
        raise ValueError('Требуется завершённая запись 10–15 секунд и не менее 200 кадров.')
    timestamps = np.asarray([s['timestamp'] for s in samples], dtype=float)
    if (not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0)
            or timestamps[0] < record_start - .01 or timestamps[-1] > record_end + .01):
        raise ValueError('Время кадров некорректно или выходит за границы записи.')
    rate = float(1 / np.median(np.diff(timestamps)))
    if rate < 24.5:
        raise ValueError('Для оценки быстрого таппинга требуется не менее 25 кадров/с; рекомендуется 30 и выше.')
    valid = [s for s in samples if s.get('distance') is not None]
    coverage = len(valid) / len(samples)
    if coverage < .85 or len(valid) < 200:
        raise ValueError(f'Кисть отслеживалась только в {coverage * 100:.1f}% кадров; требуется минимум 85%.')
    t = np.asarray([s['timestamp'] for s in valid], dtype=float)
    d = np.asarray([s['distance'] for s in valid], dtype=float)
    palm = np.asarray([s['palm_scale'] for s in valid], dtype=float)
    if (not np.isfinite(d).all() or not np.isfinite(palm).all()
            or np.any(d < 0) or np.any(d > 4) or np.any(palm < 25)):
        raise ValueError('Расстояния пальцев или масштаб ладони некорректны.')
    gap = float(np.diff(t).max())
    if gap > .15 or t[0] - record_start > .15 or record_end - t[-1] > .15:
        raise ValueError('Отслеживание кисти прерывалось более чем на 0,15 с; циклы через пропуски не учитываются.')
    if np.percentile(palm, 95) / np.percentile(palm, 5) > 1.35:
        raise ValueError('Масштаб ладони сильно изменился. Сохраняйте расстояние и положение кисти относительно камеры.')
    hands = {s.get('hand') for s in valid if s.get('hand')}
    if len(hands) > 1:
        raise ValueError('Во время записи сменилась отслеживаемая рука. Выполните отдельную пробу для каждой руки.')

    relative = t - record_start
    time = np.arange(0, relative[-1] - relative[0] + .1 / rate, 1 / rate) + relative[0]
    raw = np.interp(time, relative, d)
    window = max(3, int(round(rate * .07)) | 1)
    smooth = savgol_filter(raw, window, 2)
    excursion = float(np.percentile(smooth, 95) - np.percentile(smooth, 5))
    prominence = max(.08, .1 * excursion)
    peaks, _ = find_peaks(smooth, distance=max(1, round(rate * .12)), prominence=prominence)
    valleys, _ = find_peaks(-smooth, distance=max(1, round(rate * .06)))
    # Boundary closures can complete a cycle; boundary openings cannot.
    valleys = np.r_[0, valleys, len(smooth) - 1]
    accepted = []
    closure_limit = .45  # fingertip contact/near-contact in palm units
    for peak in peaks:
        before = valleys[valleys < peak]
        after = valleys[valleys > peak]
        if not len(before) or not len(after):
            continue
        left, right = before[-1], after[0]
        if (smooth[left] <= closure_limit and smooth[right] <= closure_limit
                and smooth[peak] - max(smooth[left], smooth[right]) >= prominence):
            accepted.append(peak)
    if len(accepted) < 3:
        raise ValueError('Недостаточно полных циклов раскрытия–смыкания. Соединяйте большой и указательный пальцы после каждого раскрытия.')
    peaks = np.asarray(accepted, dtype=int)
    peak_times = time[peaks]
    peak_amp = smooth[peaks]
    intervals = np.diff(peak_times)
    frequency = float(1 / intervals.mean())
    cv = float(intervals.std() / intervals.mean() * 100)
    # Centering avoids numerical loss when capture timestamps are absolute.
    slope, initial = np.polyfit(peak_times - peak_times[0], peak_amp, 1)
    final = float(initial + slope * (peak_times[-1] - peak_times[0]))
    if initial <= .08 or final <= 0:
        raise ValueError('Линейный тренд амплитуды нестабилен; повторите пробу с отчётливым раскрытием пальцев.')
    decrement = float((final - initial) / initial * 100)
    warnings = []
    if coverage < .98:
        warnings.append('Короткие пропуски отслеживания интерполированы; оценка ритма требует осторожности.')
    if gap > 1.8 / rate:
        warnings.append('В записи есть задержки кадров, способные повлиять на число быстрых циклов.')
    return {
        'tap_frequency': frequency, 'total_taps': len(peaks), 'rhythm_cv': cv,
        'amplitude_decrement': decrement, 'mean_amplitude': float(peak_amp.mean()),
        'amplitude_slope': float(slope), 'initial_amplitude': float(initial), 'final_amplitude': final,
        'time': time, 'distance': raw, 'smoothed': smooth, 'peak_indices': peaks,
        'peak_times': peak_times, 'peak_amplitudes': peak_amp,
        'trend': initial + slope * (peak_times - peak_times[0]),
        'rate': rate, 'coverage': coverage, 'max_gap': gap, 'duration': duration,
        'samples': len(valid), 'hand': next(iter(hands), 'не определена'), 'warnings': warnings,
    }
