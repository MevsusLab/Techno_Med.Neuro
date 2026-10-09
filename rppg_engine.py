import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, detrend, sosfiltfilt


def power_spectrum(signal, fs):
    window = np.hanning(len(signal))
    power = np.abs(rfft(signal * window)) ** 2 / (fs * np.sum(window ** 2))
    power[1:] *= 2
    if len(signal) % 2 == 0:
        power[-1] /= 2
    return rfftfreq(len(signal), 1 / fs), power


def spectral_peak(freqs, power):
    band = (freqs >= 0.75) & (freqs <= 3.0)
    indices = np.flatnonzero(band)
    index = indices[np.argmax(power[band])]
    frequency = freqs[index]
    if 0 < index < len(power) - 1:
        left, middle, right = np.log(np.maximum(power[index - 1:index + 2], np.finfo(float).tiny))
        denominator = left - 2 * middle + right
        if denominator < -1e-12:
            offset = np.clip(0.5 * (left - right) / denominator, -0.5, 0.5)
            frequency += offset * (freqs[1] - freqs[0])
    near = band & (np.abs(freqs - frequency) <= 0.15)
    total = np.sum(power[band])
    concentration = np.sum(power[near]) / total if total > 0 else 0.0
    floor = max(float(np.median(power[band])), np.finfo(float).tiny)
    ratio = float(power[index] / floor)
    return float(np.clip(frequency, 0.75, 3.0)), float(concentration), ratio


def assess_motion(timestamps, geometry):
    motion_span = np.percentile(geometry, 95, axis=0) - np.percentile(geometry, 5, axis=0)
    previous = np.searchsorted(timestamps, timestamps - 1.0)
    usable = timestamps - timestamps[previous] >= 0.8
    local_change = np.percentile(np.abs(geometry[usable] - geometry[previous[usable]]), 95, axis=0)
    if max(motion_span[:2]) > 0.5 or motion_span[2] > 0.25 or max(motion_span[3:]) > 20:
        raise ValueError('Large sustained head-position or distance change; keep the face within a consistent camera view.')
    if max(local_change[:2]) > 0.12 or local_change[2] > 0.08 or max(local_change[3:]) > 8:
        raise ValueError(
            f'Rapid repeated head movement: position {max(local_change[:2]):.3f} eye spans, '
            f'scale {local_change[2] * 100:.1f}%, pose {max(local_change[3:]):.1f} degrees per second. '
            'Remain still during measurement.'
        )
    warnings = []
    if max(motion_span[:2]) > 0.08 or motion_span[2] > 0.06 or max(motion_span[3:]) > 8:
        warnings.append('Slow head-position drift was observed. Estimate accepted only after pulse consistency and motion-artifact checks; confirm with a reference sensor.')
    if np.any(np.ptp(geometry, axis=0) - motion_span > np.array([0.04, 0.04, 0.03, 4, 4, 4])):
        warnings.append('Brief pose or tracking outliers were observed; treat the estimate cautiously.')
    return warnings, motion_span, local_change


class RPPGEngine:
    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = []
        self.last_error = ''
        self.last_quality = {}

    def add_sample(self, sample):
        self.samples.append(sample)

    def analyze(self, record_start, record_end, frame_count):
        self.last_error = ''
        self.last_quality = {'samples': len(self.samples)}
        try:
            return self._analyze(record_start, record_end, frame_count)
        except ValueError as exc:
            self.last_error = str(exc)
            return None

    def _analyze(self, record_start, record_end, frame_count):
        if record_start is None or record_end is None or not np.isfinite([record_start, record_end]).all():
            raise ValueError('Recording time bounds are unavailable.')
        duration = record_end - record_start
        if duration < 20 or len(self.samples) < 300:
            raise ValueError('At least 20 seconds and 300 valid skin-region samples are required.')
        t = np.asarray([sample['timestamp'] for sample in self.samples], dtype=float)
        rgb = np.asarray([sample['rgb'] for sample in self.samples], dtype=float)
        geometry = np.asarray([sample['geometry'] for sample in self.samples], dtype=float)
        background = np.asarray([sample['background'] for sample in self.samples], dtype=float)
        if rgb.shape != (len(t), 3, 3) or geometry.shape != (len(t), 6) or background.shape != (len(t),):
            raise ValueError('Skin-region sample data has an invalid shape.')
        if not all(np.isfinite(values).all() for values in (t, rgb, geometry, background)):
            raise ValueError('Capture contains invalid color, pose, or timing values.')
        intervals = np.diff(t)
        if np.any(intervals <= 0) or t[0] < record_start - 0.001 or t[-1] > record_end + 0.001:
            raise ValueError('Capture timestamps are invalid or outside the recording interval.')
        fs = 1 / np.median(intervals)
        coverage = min(1.0, len(t) / max(frame_count, 1), len(t) / (duration * fs))
        self.last_quality.update({'rate': float(fs), 'coverage': float(coverage), 'max_gap': float(intervals.max())})
        if fs < 15 or coverage < 0.85 or intervals.max() > 0.25 or t[0] - record_start > 0.25 or record_end - t[-1] > 0.25:
            raise ValueError(f'Unreliable sampling: {fs:.1f}/s, {coverage * 100:.0f}% valid coverage, {intervals.max():.3f}s largest gap. Keep the face visible.')
        warnings, motion_span, local_change = assess_motion(t, geometry)
        self.last_quality.update({'motion_span': motion_span.tolist(), 'local_motion': local_change.tolist(), 'warnings': warnings})
        if np.any(rgb.mean(axis=0) < 30) or np.any(rgb.mean(axis=0) > 240):
            raise ValueError('Skin regions are too dark or overexposed.')
        if np.max(np.abs(np.diff(rgb[:, :, 1], axis=0)) / np.mean(rgb[:, :, 1], axis=0)) > 0.03:
            raise ValueError('An abrupt exposure or lighting change interrupted the pulse signal.')

        time = np.arange(0, t[-1] - t[0], 1 / fs)
        green = np.column_stack([np.interp(time, t - t[0], rgb[:, i, 1]) for i in range(3)])
        fractional = 100 * (green / np.mean(green, axis=0) - 1)
        raw = detrend(fractional, axis=0)
        sos = butter(4, [0.75, 3.0], btype='bandpass', fs=fs, output='sos')
        filtered = sosfiltfilt(sos, raw, axis=0)
        interior = (time >= 1) & (time <= time[-1] - 1)
        signals = filtered[interior]
        freqs, _ = power_spectrum(signals[:, 0], fs)
        powers = np.column_stack([power_spectrum(signals[:, i], fs)[1] for i in range(3)])
        peaks = [spectral_peak(freqs, powers[:, i]) for i in range(3)]
        adequate = [i for i, (_, concentration, ratio) in enumerate(peaks)
                    if concentration >= 0.45 and ratio >= 8 and np.std(signals[:, i]) >= 0.005]
        agreeing = [i for i in adequate if sum(abs(peaks[i][0] - peaks[j][0]) <= 0.10 for j in adequate) >= 2]
        if len(agreeing) < 2:
            raise ValueError('No consistent pulse peak across at least two skin regions. Improve lighting, remain still, and retry.')
        target = float(np.median([peaks[i][0] for i in agreeing]))
        agreeing = [i for i in agreeing if abs(peaks[i][0] - target) <= 0.10]
        if len(agreeing) < 2:
            raise ValueError('Skin-region peaks disagree; BPM is withheld.')
        normalized_power = powers[:, agreeing] / np.maximum(np.sum(powers[:, agreeing], axis=0), 1e-12)
        spectrum = np.mean(normalized_power, axis=1)
        frequency, concentration, ratio = spectral_peak(freqs, spectrum)
        if concentration < 0.45 or ratio < 8 or abs(frequency - target) > 0.10:
            raise ValueError('Combined spectrum is not consistent with the individual skin regions; BPM is withheld.')

        half = len(signals) // 2
        halves = []
        for segment in (signals[:half], signals[half:]):
            sf, _ = power_spectrum(segment[:, 0], fs)
            sp = np.mean([power_spectrum(segment[:, i], fs)[1] for i in agreeing], axis=0)
            halves.append(spectral_peak(sf, sp)[0])
        if max(abs(np.asarray(halves) - frequency)) > 0.15:
            raise ValueError('Pulse estimate changes between recording halves. Repeat a steady resting measurement.')

        nuisance = np.column_stack([
            *[np.interp(time, t - t[0], geometry[:, i]) for i in range(6)],
            np.interp(time, t - t[0], background),
        ])
        nuisance = sosfiltfilt(sos, detrend(nuisance, axis=0), axis=0)[interior]
        pulse = np.mean(signals[:, agreeing], axis=1)
        if np.std(pulse) < 0.005:
            raise ValueError('Pulse modulation is too small to measure reliably.')
        for i in range(nuisance.shape[1]):
            threshold = 0.0005 if i < 3 else 0.05
            if np.std(nuisance[:, i]) > threshold and abs(np.corrcoef(pulse, nuisance[:, i])[0, 1]) > 0.8:
                raise ValueError('Candidate pulse strongly follows head movement or background lighting; BPM is withheld.')
        return {
            'bpm': frequency * 60, 'frequency': frequency, 'concentration': concentration * 100,
            'peak_ratio': ratio, 'rate': float(fs), 'coverage': float(coverage),
            'max_gap': float(intervals.max()), 'samples': len(t), 'duration': duration,
            'resolution_bpm': float((freqs[1] - freqs[0]) * 60),
            'roi_bpm': [p[0] * 60 for p in peaks], 'used_rois': agreeing,
            'time': time[interior], 'pulse': pulse, 'roi_signals': signals,
            'frequencies': freqs, 'spectrum': spectrum,
            'warnings': warnings, 'motion_span': motion_span.tolist(),
        }
