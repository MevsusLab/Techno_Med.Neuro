import unittest

import numpy as np
from rppg_engine import RPPGEngine


def pulse_samples(bpm=72, rate=30, duration=30, noise=0.01):
    t = np.arange(int(rate * duration)) / rate
    rng = np.random.default_rng(13)
    modulation = np.sin(2 * np.pi * bpm / 60 * t)
    rgb = np.zeros((len(t), 3, 3))
    for region in range(3):
        rgb[:, region] = [130, 110, 90]
        rgb[:, region, 1] += (0.3 + region * 0.05) * modulation + rng.normal(0, noise, len(t))
    return [{'timestamp': float(ti), 'rgb': color, 'background': 80.0,
             'geometry': np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])} for ti, color in zip(t, rgb)]


class RPPGEngineTests(unittest.TestCase):
    def analyze(self, samples, frames=900, end=30):
        engine = RPPGEngine()
        for sample in samples:
            engine.add_sample(sample)
        result = engine.analyze(0, end, frames)
        return engine, result

    def test_known_rates_are_recovered(self):
        for bpm in (45, 60, 72, 90, 120, 150, 180):
            with self.subTest(bpm=bpm):
                engine, result = self.analyze(pulse_samples(bpm))
                self.assertIsNotNone(result, engine.last_error)
                self.assertAlmostEqual(result['bpm'], bpm, delta=2)
                self.assertEqual(len(result['used_rois']), 3)

    def test_mild_timing_jitter_preserves_bpm(self):
        samples = pulse_samples()
        for sample in samples:
            sample['timestamp'] += 0.002 * np.sin(0.4 * sample['timestamp'])
        engine, result = self.analyze(samples)
        self.assertIsNotNone(result, engine.last_error)
        self.assertAlmostEqual(result['bpm'], 72, delta=2)

    def test_long_gap_and_bad_timestamps_rejected(self):
        for samples in ([s for s in pulse_samples() if not 10 < s['timestamp'] < 12], pulse_samples()):
            if len(samples) == 900:
                samples[100]['timestamp'] = samples[99]['timestamp']
            engine, result = self.analyze(samples)
            self.assertIsNone(result)
            self.assertTrue(engine.last_error)

    def test_flat_and_noisy_signals_are_rejected(self):
        rng = np.random.default_rng(21)
        for noisy in (False, True):
            samples = pulse_samples()
            for sample in samples:
                sample['rgb'][:, 1] = 110 + (rng.normal(0, 0.05, 3) if noisy else 0)
            engine, result = self.analyze(samples)
            self.assertIsNone(result)
            self.assertIn('consistent pulse', engine.last_error)

    def test_region_disagreement_withholds_estimate(self):
        samples = pulse_samples()
        for sample in samples:
            sample['rgb'][:, 1] = 110 + 0.3 * np.sin(2 * np.pi * np.array([1.0, 1.5, 2.0]) * sample['timestamp'])
        _, result = self.analyze(samples)
        self.assertIsNone(result)

    def test_lighting_step_is_rejected(self):
        samples = pulse_samples()
        for sample in samples:
            if sample['timestamp'] >= 15:
                sample['rgb'][:, 1] += 10
        engine, result = self.analyze(samples)
        self.assertIsNone(result)
        self.assertIn('exposure', engine.last_error)

    def test_motion_and_background_correlated_peaks_are_rejected(self):
        for source in ('geometry', 'background'):
            samples = pulse_samples()
            for sample in samples:
                wave = np.sin(2 * np.pi * 1.2 * sample['timestamp'])
                if source == 'geometry':
                    sample[source][0] += 0.01 * wave
                else:
                    sample[source] += 0.3 * wave
            engine, result = self.analyze(samples)
            self.assertIsNone(result)
            self.assertIn('strongly follows', engine.last_error)

    def test_changed_rate_between_halves_is_rejected(self):
        samples = pulse_samples()
        for sample in samples:
            t = sample['timestamp']
            sample['rgb'][:, 1] = 110 + 0.3 * np.sin(2 * np.pi * (1.0 if t < 15 else 1.8) * t)
        _, result = self.analyze(samples)
        self.assertIsNone(result)

    def test_reset_clears_session(self):
        engine, _ = self.analyze(pulse_samples())
        engine.reset()
        self.assertEqual(engine.samples, [])
        self.assertEqual(engine.last_error, '')

    def test_out_of_band_oscillation_is_not_reported_as_pulse(self):
        for bpm in (30, 210):
            with self.subTest(bpm=bpm):
                engine, result = self.analyze(pulse_samples(bpm))
                self.assertIsNone(result)
                self.assertTrue(engine.last_error)

    def test_one_failed_region_does_not_invalidate_two_agreeing_regions(self):
        samples = pulse_samples()
        rng = np.random.default_rng(41)
        for sample in samples:
            sample['rgb'][0, 1] = 110 + rng.normal(0, 0.05)
        engine, result = self.analyze(samples)
        self.assertIsNotNone(result, engine.last_error)
        self.assertEqual(result['used_rois'], [1, 2])

    def test_gradual_head_drift_preserves_consistent_pulse_with_caution(self):
        samples = pulse_samples()
        for sample in samples:
            fraction = sample['timestamp'] / 30
            sample['geometry'][0] += 0.18 * fraction
            sample['geometry'][2] += 0.10 * fraction
            sample['geometry'][3] += 10 * fraction
            sample['rgb'][:, 1] += 1.5 * fraction
        engine, result = self.analyze(samples)
        self.assertIsNotNone(result, engine.last_error)
        self.assertAlmostEqual(result['bpm'], 72, delta=2)
        self.assertTrue(result['warnings'])
        self.assertIn('Slow head-position drift', result['warnings'][0])

    def test_single_pose_outlier_does_not_reject_clean_pulse(self):
        samples = pulse_samples()
        samples[400]['geometry'][0] += 0.12
        samples[400]['geometry'][4] += 10
        engine, result = self.analyze(samples)
        self.assertIsNotNone(result, engine.last_error)
        self.assertAlmostEqual(result['bpm'], 72, delta=2)
        self.assertIn('outliers', ' '.join(result['warnings']))

    def test_large_sustained_or_repeated_fast_motion_is_rejected(self):
        for fast in (False, True):
            samples = pulse_samples()
            for sample in samples:
                t = sample['timestamp']
                sample['geometry'][0] += 0.14 * np.sin(2 * np.pi * 0.4 * t) if fast else t / 30
            engine, result = self.analyze(samples)
            self.assertIsNone(result)
            self.assertIn('head', engine.last_error)

    def test_head_drift_without_consistent_pulse_still_rejects(self):
        samples = pulse_samples()
        for sample in samples:
            sample['geometry'][0] += 0.18 * sample['timestamp'] / 30
            sample['rgb'][:, 1] = 110 + 0.05 * sample['timestamp']
        _, result = self.analyze(samples)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
