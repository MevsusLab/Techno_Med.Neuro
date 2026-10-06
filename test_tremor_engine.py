import unittest

import numpy as np

from tremor_engine import TremorEngine


class TremorEngineTests(unittest.TestCase):
    def record(self, t, x, y):
        engine = TremorEngine()
        for ti, xi, yi in zip(t, x, y):
            engine.add_sample(float(xi), float(yi), float(ti))
        return engine

    def test_horizontal_motion_is_detected(self):
        t = np.arange(450) / 30
        engine = self.record(t, 10 * np.sin(2 * np.pi * 5 * t), np.zeros_like(t))
        result = engine.analyze()
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["dominant_freq"], 5, delta=0.2)
        self.assertAlmostEqual(result["rms_px"], 10 / np.sqrt(2), delta=0.5)

    def test_two_axis_motion_uses_vector_rms(self):
        t = np.arange(450) / 30
        engine = self.record(t, 10 * np.cos(2 * np.pi * 5 * t), 10 * np.sin(2 * np.pi * 5 * t))
        result = engine.analyze()
        self.assertAlmostEqual(result["dominant_freq"], 5, delta=0.2)
        self.assertAlmostEqual(result["rms_px"], 10, delta=0.6)

    def test_short_frame_jitter_is_resampled(self):
        t = np.arange(450) / 30
        t += 0.004 * np.sin(2 * np.pi * 0.7 * t)
        engine = self.record(t, np.zeros_like(t), 8 * np.sin(2 * np.pi * 7 * t))
        self.assertAlmostEqual(engine.analyze()["dominant_freq"], 7, delta=0.2)

    def test_long_tracking_gap_is_rejected(self):
        t = np.arange(450) / 30
        t = t[(t < 5) | (t >= 8)]
        engine = self.record(t, np.zeros_like(t), 10 * np.sin(2 * np.pi * 5 * t))
        self.assertIsNone(engine.analyze())
        self.assertIn("interrupted", engine.last_error)

    def test_missing_hand_at_start_or_end_is_rejected(self):
        t = np.arange(450) / 30
        for kept in (t[t >= 1], t[t < 14]):
            engine = self.record(kept, np.zeros_like(kept), 10 * np.sin(2 * np.pi * 5 * kept))
            self.assertIsNone(engine.analyze(record_start=0, record_end=15))
            self.assertIn("start or end", engine.last_error)

    def test_complete_recording_bounds_are_accepted(self):
        t = np.arange(450) / 30
        engine = self.record(t, np.zeros_like(t), 10 * np.sin(2 * np.pi * 5 * t))
        self.assertAlmostEqual(engine.analyze(record_start=0, record_end=15)["dominant_freq"], 5, delta=0.2)

    def test_stopped_early_is_rejected_even_with_enough_samples(self):
        t = np.arange(450) / 30
        engine = self.record(t, np.zeros_like(t), 10 * np.sin(2 * np.pi * 5 * t))
        self.assertIsNone(engine.analyze(record_start=0, record_end=9))

    def test_low_frame_rate_is_rejected(self):
        t = np.arange(300) / 20
        engine = self.record(t, np.zeros_like(t), 10 * np.sin(2 * np.pi * 5 * t))
        self.assertIsNone(engine.analyze())
        self.assertIn("rate", engine.last_error)

    def test_flat_signal_has_no_frequency(self):
        t = np.arange(450) / 30
        engine = self.record(t, np.zeros_like(t), np.zeros_like(t))
        result = engine.analyze()
        self.assertIsNone(result["dominant_freq"])
        self.assertEqual(result["rms_px"], 0)

    def test_short_capture_is_rejected_and_reset_clears_error(self):
        t = np.arange(150) / 30
        engine = self.record(t, np.zeros_like(t), np.zeros_like(t))
        self.assertIsNone(engine.analyze())
        self.assertIn("10 seconds", engine.last_error)
        engine.reset()
        self.assertEqual(engine.count, 0)
        self.assertEqual(engine.last_error, "")

    def test_duplicate_timestamps_are_rejected(self):
        t = np.arange(450) / 30
        t[100] = t[99]
        engine = self.record(t, np.zeros_like(t), 10 * np.sin(2 * np.pi * 5 * t))
        self.assertIsNone(engine.analyze())
        self.assertIn("timestamps", engine.last_error)

    def test_nonfinite_coordinates_are_rejected(self):
        t = np.arange(450) / 30
        y = np.sin(2 * np.pi * 5 * t)
        y[100] = np.nan
        engine = self.record(t, np.zeros_like(t), y)
        self.assertIsNone(engine.analyze())
        self.assertIn("Invalid", engine.last_error)

    def test_irregular_timing_reduces_bandwidth(self):
        intervals = np.tile([0.02, 0.04], 300)
        t = np.cumsum(intervals)
        engine = self.record(t, np.zeros_like(t), np.sin(2 * np.pi * 5 * t))
        result = engine.analyze()
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['max_freq'], 10, delta=0.01)
        self.assertAlmostEqual(result['dominant_freq'], 5, delta=0.2)

    def test_reported_webcam_gap_keeps_only_supported_band(self):
        t = np.arange(497) / 33.1
        keep = np.ones(len(t), dtype=bool)
        keep[20::11] = False
        t = t[keep]
        t[100] += 0.021
        self.assertAlmostEqual(np.max(np.diff(t)), 0.081, delta=0.004)
        engine = self.record(t, 10 * np.sin(2 * np.pi * 4 * t), np.zeros_like(t))
        result = engine.analyze(record_start=t[0], record_end=t[-1])
        self.assertIsNotNone(result)
        self.assertLess(result['max_freq'], 5)
        self.assertGreater(result['coverage'], 0.8)
        self.assertAlmostEqual(result['dominant_freq'], 4, delta=0.2)

    def test_frequency_above_limited_band_is_not_reported(self):
        t = np.arange(497) / 33.1
        keep = np.ones(len(t), dtype=bool)
        keep[20::11] = False
        t = t[keep]
        t[100] += 0.021
        engine = self.record(t, 10 * np.sin(2 * np.pi * 5 * t), np.zeros_like(t))
        result = engine.analyze()
        self.assertIsNotNone(result)
        self.assertIsNone(result['dominant_freq'])

    def test_rms_scales_with_motion_not_direction(self):
        t = np.arange(900) / 60
        a = 6 * np.sin(2 * np.pi * 8 * t)
        vertical = self.record(t, np.zeros_like(t), a).analyze()
        diagonal = self.record(t, a / np.sqrt(2), a / np.sqrt(2)).analyze()
        self.assertAlmostEqual(vertical["rms_px"], diagonal["rms_px"], places=6)
        self.assertAlmostEqual(vertical["rms_px"], 6 / np.sqrt(2), delta=0.1)
        self.assertEqual(vertical["dominant_freq"], diagonal["dominant_freq"])

    def test_broadband_noise_is_not_labeled_as_clear_peak(self):
        t = np.arange(450) / 30
        rng = np.random.default_rng(17)
        engine = self.record(t, rng.normal(size=len(t)), rng.normal(size=len(t)))
        self.assertIsNone(engine.analyze()["dominant_freq"])


if __name__ == "__main__":
    unittest.main()
