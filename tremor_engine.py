import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, detrend, sosfiltfilt, welch


class TremorEngine:
    def __init__(self):
        self.reset()

    def reset(self):
        self.timestamps = []
        self.x = []
        self.y = []
        self.last_error = ""
        self.last_quality = {}

    def add_sample(self, x, y, timestamp):
        self.timestamps.append(timestamp)
        self.x.append(x)
        self.y.append(y)

    @property
    def duration(self):
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    @property
    def count(self):
        return len(self.timestamps)

    def analyze(self, record_start=None, record_end=None):
        self.last_error = ""
        self.last_quality = {"samples": self.count, "duration_sec": self.duration}
        if self.count < 2:
            self.last_error = "No usable hand recording. Keep one hand fully visible."
            return None

        t = np.asarray(self.timestamps, dtype=np.float64)
        x = np.asarray(self.x, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.float64)
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            self.last_error = "Invalid landmark coordinates or timestamps."
            return None
        intervals = np.diff(t)
        if np.any(intervals <= 0):
            self.last_error = "Frame timestamps are not strictly increasing."
            return None
        fs = 1.0 / np.median(intervals)
        capture_duration = record_end - record_start if record_start is not None and record_end is not None else self.duration
        coverage = min(1.0, self.count / (capture_duration * fs)) if capture_duration > 0 else 0.0
        max_gap = float(np.max(intervals))
        jitter = float(np.percentile(np.abs(intervals * fs - 1), 90))
        self.last_quality.update({
            "fs": float(fs), "coverage": float(coverage), "max_gap_sec": max_gap,
            "timing_jitter": jitter,
        })
        if self.count < 200 or self.duration < 10:
            self.last_error = f"Only {self.duration:.1f}s / {self.count} hand samples. Record at least 10 seconds."
            return None
        if record_start is not None and record_end is not None:
            if not np.isfinite(record_start) or not np.isfinite(record_end) or record_end <= record_start:
                self.last_error = "Invalid recording time bounds."
                return None
            if record_end - record_start < 10:
                self.last_error = "Record at least 10 seconds with a visible hand."
                return None
            if t[0] < record_start - 0.02 or t[-1] > record_end + 0.02:
                self.last_error = "Landmarks fall outside the recording interval."
                return None
            if t[0] - record_start > 0.2 or record_end - t[-1] > 0.2:
                self.last_error = (
                    f"Hand tracking was interrupted at the start or end "
                    f"(start: {max(0, t[0] - record_start):.2f}s; end: {max(0, record_end - t[-1]):.2f}s). "
                    "Keep the hand visible until recording finishes."
                )
                return None

        if fs < 25:
            self.last_error = (
                f"Tracking rate is {fs:.1f} samples/s; at least 25 are required. "
                "Improve lighting and close other camera or CPU-heavy applications."
            )
            return None
        if max_gap > 0.2 or coverage < 0.8:
            self.last_error = (
                f"Hand tracking was interrupted (largest gap: {max_gap:.3f}s; "
                f"limit: 0.200s; coverage: {coverage * 100:.0f}%). "
                "Keep the hand visible and avoid sudden movement."
            )
            return None
        high = min(12.0, 0.45 * fs, 0.4 / max_gap)
        if high < 3:
            self.last_error = (
                f"Largest tracking gap ({max_gap:.3f}s) leaves less than 3 Hz of usable bandwidth. "
                "Keep the hand visible and reduce camera or CPU load."
            )
            return None

        time = np.arange(0, self.duration, 1 / fs)
        x = np.interp(time, t - t[0], x)
        y = np.interp(time, t - t[0], y)
        raw_x = detrend(x)
        raw_y = detrend(y)
        sos = butter(4, [1.0, high], btype="bandpass", fs=fs, output="sos")
        filtered_x = sosfiltfilt(sos, raw_x)
        filtered_y = sosfiltfilt(sos, raw_y)
        stable = (time >= 1.0) & (time <= self.duration - 1.0)
        stable_x = filtered_x[stable]
        stable_y = filtered_y[stable]

        nperseg = min(256, len(stable_x))
        freqs, px = welch(stable_x, fs=fs, nperseg=nperseg, window="hamming")
        _, py = welch(stable_y, fs=fs, nperseg=nperseg, window="hamming")
        psd = px + py
        band = (freqs >= 1) & (freqs <= high)
        band_freqs = freqs[band]
        band_psd = psd[band]
        peak_index = np.argmax(band_psd)
        peak = band_freqs[peak_index]
        resolution = float(freqs[1] - freqs[0])
        _, raw_px = welch(raw_x[stable], fs=fs, nperseg=nperseg, window="hamming")
        _, raw_py = welch(raw_y[stable], fs=fs, nperseg=nperseg, window="hamming")
        full_band = (freqs >= 1) & (freqs <= min(12.0, 0.45 * fs))
        unfiltered_peak = freqs[full_band][np.argmax((raw_px + raw_py)[full_band])]
        above_band = unfiltered_peak >= high - resolution
        total_power = np.sum(band_psd)
        concentration = 100 * np.sum(band_psd[np.abs(band_freqs - peak) <= 1]) / total_power if total_power > 0 else 0.0
        peak_ratio = band_psd[peak_index] / np.median(band_psd) if np.median(band_psd) > 0 else 0.0
        rms = np.sqrt(np.mean(stable_x ** 2 + stable_y ** 2))

        window = np.hamming(len(stable_x))
        fft_freqs = rfftfreq(len(stable_x), 1 / fs)
        fft_mag = np.sqrt(np.abs(rfft(stable_x * window)) ** 2 + np.abs(rfft(stable_y * window)) ** 2) * 2 / np.sum(window)
        fft_mag[0] /= 2
        if len(stable_x) % 2 == 0:
            fft_mag[-1] /= 2

        return {
            "dominant_freq": float(peak) if rms > 1e-8 and peak_ratio >= 4 and concentration >= 30 and peak < high - resolution and not above_band else None,
            "above_band": bool(above_band),
            "rms_px": float(rms),
            "concentration": float(concentration),
            "fs": float(fs),
            "max_freq": float(high),
            "samples": self.count,
            "duration_sec": self.duration,
            "resolution_hz": resolution,
            "coverage": float(coverage),
            "max_gap_sec": float(np.max(intervals)),
            "time": time,
            "raw_x": raw_x,
            "raw_y": raw_y,
            "filtered_y": filtered_y,
            "filtered_x": filtered_x,
            "fft_freqs": fft_freqs,
            "fft_mag": fft_mag,
            "psd_freqs": freqs,
            "psd": psd,
        }
