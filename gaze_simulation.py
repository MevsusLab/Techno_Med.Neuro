"""Landmark-level virtual AAC user for hardware-free end-to-end gaze tests.

A 3D head (two 12 mm eyeballs, canthi, lids, nose tip) is rotated by head pose,
the irises by eye-in-head rotation, then perspective-projected into a mirrored
640x480 image with per-landmark pixel noise. Samples are produced exactly as
gaze_capture produces them: telemetry ratios and tracking features, EMA-smoothed.
"""

import numpy as np

from gaze_engine import GazeSignalFilter, gaze_features, tracking_features

LANDMARKS = 478
FIXATE_MESSAGE = 'Фиксируйте взгляд на слове или стрелке до заполнения шкалы.'
NO_FACE_MESSAGE = 'Держите лицо в кадре и смотрите на экран.'


def screen_angles(point):
    """Normalized window position -> gaze yaw/pitch in degrees (60 cm viewing)."""
    return np.array([(point[0] - 0.5) * 30.0, (point[1] - 0.5) * 20.0])


def _rotation(yaw, pitch):
    a, b = np.radians(yaw), np.radians(pitch)
    around_y = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    around_x = np.array([[1, 0, 0], [0, np.cos(b), np.sin(b)], [0, -np.sin(b), np.cos(b)]])
    return around_x @ around_y


def face_landmarks(gaze=(0.0, 0.0), head=(0.0, 0.0), blink=False, iris_gain=1.0, nose_depth=30.0,
                   noise=0.0, rng=None, distance=600.0):
    points = np.zeros((LANDMARKS, 3))
    eyes = ((-32, (468, 133, 33, 159, 145), (160, 158, 153, 144)),
            (32, (473, 362, 263, 386, 374), (385, 387, 373, 380)))
    eye_yaw, eye_pitch = np.radians(np.subtract(gaze, head))
    direction = np.array([np.sin(eye_yaw) * np.cos(eye_pitch), np.sin(eye_pitch),
                          np.cos(eye_yaw) * np.cos(eye_pitch)])
    for x, (iris, inner, outer, upper, lower), contour in eyes:
        side = np.sign(x)
        points[inner] = [x - side * 15, 0, 8]
        points[outer] = [x + side * 15, 0, 6]
        opening = 0.4 if blink else 5.0
        points[upper] = [x, -opening, 10]
        points[lower] = [x, opening, 10]
        for index, offset in zip(contour, (-7, 7, 7, -7)):
            points[index] = [x + offset, -opening if index in (160, 158, 385, 387) else opening, 9]
        points[iris] = np.array([x, 0, 0]) + 12 * iris_gain * (direction - [0, 0, 1]) + [0, 0, 12]
        points[iris + 1:iris + 5] = points[iris] + [[5, 0, 0], [0, -5, 0], [-5, 0, 0], [0, 5, 0]]
    points[1] = [0, 35, nose_depth]
    pivot = np.array([0, 0, -90.0])
    points = (points - pivot) @ _rotation(*head).T + pivot
    depth = distance - points[:, 2]
    image = np.column_stack((320 + 700 * points[:, 0] / depth, 240 + 700 * points[:, 1] / depth))
    if noise:
        image = image + rng.normal(0, noise, image.shape)
    return image


class VirtualUser:
    """Looks at window points with natural head participation and drift."""

    def __init__(self, seed=0, head_gain=0.3, noise=0.25, iris_gain=1.0, nose_depth=30.0,
                 micro_motion=0.15, fps=30.0, start=1000.0):
        self.rng = np.random.default_rng(seed)
        self.head_gain = head_gain
        self.noise = noise
        self.iris_gain = iris_gain
        self.nose_depth = nose_depth
        self.micro_motion = micro_motion
        self.fps = fps
        self.time = start
        self.head_offset = np.zeros(2)
        self.head_walk = np.zeros(2)
        self.signal_filter = GazeSignalFilter()

    def drift(self, degrees):
        self.head_offset = self.head_offset + self.rng.normal(0, degrees, 2)

    def sample(self, point, sweep=False, face=True, blink=False, iris_spike=None, roll=0.0):
        self.time += 1.0 / self.fps
        gaze_angle = screen_angles(point) + self.rng.normal(0, 0.2, 2)
        self.head_walk = 0.98 * self.head_walk + self.rng.normal(0, self.micro_motion, 2)
        head = self.head_gain * screen_angles(point) + self.head_offset + self.head_walk
        if sweep:
            head = head + [5.0 * np.sin(2 * np.pi * self.time / 2.0), 4.0 * np.sin(2 * np.pi * self.time / 1.4)]
        ratios = features = None
        message = NO_FACE_MESSAGE
        if face:
            pixels = face_landmarks(gaze_angle, head, blink, self.iris_gain, self.nose_depth,
                                    self.noise, self.rng)
            if iris_spike is not None:
                pixels[[468, 473]] += np.asarray(iris_spike, dtype=float)
            if roll:
                angle = np.deg2rad(roll)
                rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                pixels = (pixels - [320, 240]) @ rotation.T + [320, 240]
            try:
                ratios, features = gaze_features(pixels), tracking_features(pixels)
                message = FIXATE_MESSAGE
            except ValueError as exc:
                message = str(exc)
        ratios, smoothed, features = self.signal_filter.update(ratios, features, self.time)
        return {'gaze': ratios, 'features': smoothed,
                'raw_features': features, 'message': message, 'timestamp': self.time}


class PageDriver:
    """Feeds a virtual user's samples into an AACPage as the GUI poll does."""

    def __init__(self, page, user):
        self.page = page
        self.user = user
        self.badges = []
        self.prompts = []

    def centers(self):
        return self.page._target_centers()

    def frame(self, point, **kwargs):
        self.page._process_sample(self.user.sample(point, **kwargs))
        badge = self.page.badge.text()
        if not self.badges or self.badges[-1] != badge:
            self.badges.append(badge)

    def calibrate(self, max_seconds=120.0, blink_every=0, misdirect=None):
        """Follow the highlighted button. misdirect: {(stage, index): other_index}."""
        start = self.user.time
        frame = 0
        while self.page.calibration_active:
            if self.user.time - start > max_seconds:
                raise AssertionError('calibration did not finish: ' + self.page.guidance.text())
            index = self.page.calibration_queue[0]
            stage = self.page.calibration_stage
            if misdirect and (stage, index) in misdirect:
                index = misdirect[(stage, index)]
            blink = bool(blink_every) and frame % blink_every in (0, 1, 2, 3)
            self.frame(self.centers()[index], sweep=stage == 'head', blink=blink)
            prompt = self.page.guidance.text()
            if not self.prompts or self.prompts[-1] != prompt:
                self.prompts.append(prompt)
            frame += 1
        return self.user.time - start

    def look(self, point, seconds, **kwargs):
        for _ in range(int(round(seconds * self.user.fps))):
            self.frame(point, **kwargs)
