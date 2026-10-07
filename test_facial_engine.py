import unittest
import numpy as np
from facial_engine import normalize_face, analyze_stage, compare_stages, PAIRS


def symmetric_face():
    points = np.zeros((478, 3))
    points[:, :2] = [320, 200]
    for i, y in ((9, 100), (168, 180), (1, 230), (152, 350)):
        points[i, :2] = [320, y]
    for pairs in PAIRS.values():
        for right, left in pairs:
            points[right, :2] = [260, 240]
            points[left, :2] = [380, 240]
    for i, xy in {33: (220, 180), 263: (420, 180), 133: (290, 180), 362: (350, 180),
                  159: (255, 170), 145: (255, 190), 386: (385, 170), 374: (385, 190),
                  70: (255, 140), 300: (385, 140)}.items():
        points[i, :2] = xy
    return points


class FacialEngineTests(unittest.TestCase):
    def test_symmetric_geometry_is_zero(self):
        result = normalize_face(symmetric_face(), (0, 0, 0))
        self.assertAlmostEqual(result['aggregate'], 0, places=7)

    def test_translation_scale_and_roll_preserve_score(self):
        points = symmetric_face()
        points[61, 1] += 10
        expected = normalize_face(points, (0, 0, 0))['aggregate']
        theta = np.deg2rad(15)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        points[:, :2] = points[:, :2] @ rotation.T * 1.3 + [30, 50]
        self.assertAlmostEqual(normalize_face(points, (0, 0, 15))['aggregate'], expected, places=6)

    def test_lowered_mouth_corner_is_detected(self):
        points = symmetric_face()
        points[61, 1] += 20
        self.assertGreater(normalize_face(points, (0, 0, 0))['regions']['mouth'], 4)

    def test_head_turn_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_face(symmetric_face(), (25, 0, 0))

    def test_closed_eyes_and_nonfinite_landmarks_are_rejected(self):
        points = symmetric_face()
        points[159] = points[145]
        with self.assertRaises(ValueError):
            normalize_face(points, (0, 0, 0))
        points = symmetric_face()
        points[100, 0] = np.nan
        with self.assertRaises(ValueError):
            normalize_face(points, (0, 0, 0))

    def test_interrupted_stage_is_rejected(self):
        samples = [{'timestamp': float(t), 'landmarks': symmetric_face(), 'pose': (0, 0, 0)} for t in np.arange(0, 3, 0.05) if not 1 < t < 2]
        with self.assertRaises(ValueError):
            analyze_stage(samples, 0, 3)

    def test_small_expression_is_not_high_imbalance(self):
        rest = analyze_stage([{'timestamp': float(t), 'landmarks': symmetric_face(), 'pose': (0, 0, 0)} for t in np.arange(0, 3, 0.05)], 0, 3)
        with self.assertRaises(ValueError):
            compare_stages(rest, rest, rest)

    def test_movement_imbalance_has_defined_percentage(self):
        samples = [{'timestamp': float(t), 'landmarks': symmetric_face(), 'pose': (0, 0, 0)} for t in np.arange(0, 3, 0.05)]
        rest = analyze_stage(samples, 0, 3)
        smile = dict(rest, mouth_position=rest['mouth_position'] + [[-0.10, -0.05], [0.05, -0.025]])
        brows = dict(rest, brow=np.asarray(rest['brow']) + 0.1)
        result = compare_stages(rest, smile, brows)
        self.assertGreater(result['smile_imbalance'], 0)
        self.assertAlmostEqual(result['brow_imbalance'], 0)

    def test_vertical_smile_movement_is_measured(self):
        samples = [{'timestamp': float(t), 'landmarks': symmetric_face(), 'pose': (0, 0, 0)} for t in np.arange(0, 3, 0.05)]
        rest = analyze_stage(samples, 0, 3)
        smile = dict(rest, mouth_position=rest['mouth_position'] + [[0, -0.1], [0, -0.1]])
        brows = dict(rest, brow=np.asarray(rest['brow']) + 0.1)
        self.assertAlmostEqual(compare_stages(rest, smile, brows)['smile_imbalance'], 0)


if __name__ == '__main__':
    unittest.main()
