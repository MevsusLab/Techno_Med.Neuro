"""Gaze features, per-button calibration classifier and dwell state machine.

Pipeline: MediaPipe landmarks -> iris ratios (telemetry) and a 4-D tracking
feature (iris position in a roll-corrected eye-corner frame plus head-pose
proxies) -> EMA smoothing -> classifier trained on per-button calibration
fixations -> dwell state machine over the existing button rectangles.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import chi2


EMA_ALPHA = 0.25
DWELL_THRESHOLD_MS = 900
COOLDOWN_MS = 500
DEADZONE_PX = 12
# Frame gaps up to this length (a blink, a dropped frame) keep an active dwell.
MAX_GAZE_GAP_SECONDS = 0.35
# A momentarily unclassified frame keeps the current dwell for this long.
DWELL_GRACE_SECONDS = 0.2

CALIBRATION_SETTLE_SECONDS = 0.7
CALIBRATION_SAMPLE_SECONDS = 1.0
CALIBRATION_MIN_SAMPLES = 16
CALIBRATION_MAX_WINDOW_SECONDS = 3.0
# After this long on one button with a visible face, the best recent samples
# are accepted so calibration always progresses instead of failing.
CALIBRATION_ACCEPT_SECONDS = 6.0
CALIBRATION_GAP_SECONDS = 0.5

EYE_OPEN_RATIO = 0.07
ENTRY_POSTERIOR = 0.75
RETAIN_POSTERIOR = 0.35
# Boundary band between neighbouring buttons (fraction of their spacing):
# entry needs <=40% of the way to any neighbour, retention holds to the midpoint.
ENTRY_FRACTION = 0.4
RETAIN_FRACTION = 0.5
WEAK_SEPARATION = 2.5
FEATURE_DIM = 4
# Minimum per-feature standard deviation: iris x/y, head x/y (eye-width / IOD units).
FEATURE_NOISE_FLOOR = np.array([0.004, 0.003, 0.004, 0.004])

# Head rotation with fixed gaze moves the iris against the head (vestibulo-
# ocular reflex). The classifier tolerates this much head-pose change, using
# the user's own head-movement step or, failing that, a geometric prior
# (eyeball radius 12 mm, nose tip ~30 mm in front of the canthi).
HEAD_DRIFT_DEGREES = 1.5
HEAD_PER_DEGREE = 0.0055              # nose-tip offset / inter-ocular distance per degree
IRIS_PER_HEAD_DEGREE = np.array([0.0034, 0.0036])
MIN_HEAD_SWEEP_DEGREES = 1.0
HEAD_SWEEP_SECONDS = 3.0
# Gaze farther than one neighbour spacing from every button is off the grid.
OUTSIDE_SEPARATION = 1.0
# Slow online correction of common drift from confident, well-inside frames.
DRIFT_RATE = 0.02
DRIFT_CONFIDENCE = 0.9
DRIFT_FRACTION = 0.25
DRIFT_BOUND = 0.35
# A large, isolated feature step must recur on the next frame before it can
# influence calibration or EMA. This rejects iris/landmark spikes without
# widening target boundaries or accepting gaze on the gaps between buttons.
FEATURE_JUMP_LIMIT = np.array([0.06, 0.04, 0.05, 0.05])

# iris, inner canthus, outer canthus, upper lid, lower lid
LEFT_EYE = (468, 133, 33, 159, 145)
RIGHT_EYE = (473, 362, 263, 386, 374)
NOSE_TIP = 1


def _noise_floor(dim):
    return FEATURE_NOISE_FLOOR if dim == FEATURE_DIM else np.full(dim, 0.003)


def gaze_features(landmarks, pose=None):
    """Average normalized iris ratios of both eyes (telemetry X/Y).

    ratio_x = (iris_x - inner_x) / (outer_x - inner_x)
    ratio_y = (iris_y - upper_y) / (lower_y - upper_y)
    The inner-to-outer axis has opposite signs in the two eyes; the reversed
    eye is reflected before averaging so horizontal motion cannot cancel out.
    Pixel or normalized coordinates may be supplied.
    """
    points = np.asarray(landmarks, dtype=float)
    if points.ndim != 2 or points.shape[0] < 478 or points.shape[1] < 2 or not np.isfinite(points).all():
        raise ValueError('Надёжное положение радужки не найдено.')
    if pose is not None:
        if len(pose) != 3 or not np.isfinite(pose).all():
            raise ValueError('Оценка положения головы недоступна.')
        yaw, pitch, roll = pose
        if abs(yaw) > 18 or abs(pitch) > 15 or abs(roll) > 20:
            raise ValueError('Поверните голову прямо к экрану и держите её ровно.')
    horizontal, vertical = [], []
    for iris, inner, outer, upper, lower in (LEFT_EYE, RIGHT_EYE):
        corners = points[[inner, outer], :2]
        left, right = corners[np.argsort(corners[:, 0])]
        axis, normal, width = _corner_frame(left, right)
        height = float(np.dot(points[lower, :2] - points[upper, :2], normal))
        if height <= width * EYE_OPEN_RATIO:
            raise ValueError('Глаза закрыты или слишком малы для отслеживания взгляда.')
        ratio_x = float(np.dot(points[iris, :2] - left, axis) / width)
        ratio_y = float(np.dot(points[iris, :2] - points[upper, :2], normal) / height)
        ratios = np.array([ratio_x, ratio_y])
        if not np.isfinite(ratios).all() or np.any(ratios < -0.25) or np.any(ratios > 1.25):
            raise ValueError('Взгляд не определяется стабильно; поправьте освещение и положение головы.')
        horizontal.append(ratio_x)
        vertical.append(ratio_y)
    return np.clip([np.mean(horizontal), np.mean(vertical)], 0.0, 1.0)


def _corner_frame(left, right):
    axis = right - left
    width = np.linalg.norm(axis)
    if width < 1e-8:
        raise ValueError('Не удалось определить ось глаза.')
    axis = axis / width
    return axis, np.array([-axis[1], axis[0]]), width


def tracking_features(landmarks):
    """4-D feature for screen targeting; supply image-pixel coordinates.

    [0, 1]: iris position in each eye's corner frame (along the canthus axis,
            and perpendicular displacement), averaged over both eyes. Eyelids
            move with vertical gaze, so they are not used as the reference.
    [2, 3]: nose-tip offset from the eye-corner midpoint in a roll-corrected
            inter-ocular frame, a proxy for head yaw and pitch.
    All components are invariant to image translation, scale and roll. The
    classifier learns how a user combines eye and head movement per button.
    """
    points = np.asarray(landmarks, dtype=float)
    gaze_features(points)  # finite values, open eyes, plausible iris position
    eyes = []
    for iris, inner, outer, _, _ in (LEFT_EYE, RIGHT_EYE):
        corners = points[[inner, outer], :2]
        left, right = corners[np.argsort(corners[:, 0])]
        axis, normal, width = _corner_frame(left, right)
        iris_point = points[iris, :2]
        eyes.append([np.dot(iris_point - left, axis) / width,
                     0.5 + np.dot(iris_point - (left + right) / 2, normal) / width])
    outer = points[[LEFT_EYE[2], RIGHT_EYE[2]], :2]
    left, right = outer[np.argsort(outer[:, 0])]
    axis, normal, iod = _corner_frame(left, right)
    middle = points[[LEFT_EYE[1], LEFT_EYE[2], RIGHT_EYE[1], RIGHT_EYE[2]], :2].mean(axis=0)
    nose = points[NOSE_TIP, :2] - middle
    head = [np.dot(nose, axis) / iod, np.dot(nose, normal) / iod]
    features = np.concatenate([np.mean(eyes, axis=0), head])
    if not np.isfinite(features).all():
        raise ValueError('Надёжное положение радужки не найдено.')
    return features


class GazeSmoother:
    """EMA with optional frame-rate independence and brief missing-frame grace.

    Missing observations always return None. Keeping history for a short blink
    smooths reacquisition without emitting held gaze that could trigger dwell.
    """

    def __init__(self, alpha=EMA_ALPHA, time_aware=False, missing_grace=0.0):
        self.alpha = alpha
        self.time_aware = time_aware
        self.missing_grace = missing_grace
        self.reset()

    def reset(self):
        self.previous = None
        self.timestamp = None
        self.last_valid_timestamp = None

    def update(self, gaze, timestamp):
        if gaze is None:
            if (self.missing_grace > 0 and np.isfinite(timestamp)
                    and self.last_valid_timestamp is not None
                    and 0 <= timestamp - self.last_valid_timestamp <= self.missing_grace):
                if self.timestamp is None or timestamp > self.timestamp:
                    self.timestamp = timestamp
                return None
            self.reset()
            return None
        value = np.asarray(gaze, dtype=float)
        if value.ndim != 1 or not np.isfinite(value).all() or not np.isfinite(timestamp):
            self.reset()
            return None
        if self.previous is not None and value.shape != self.previous.shape:
            self.reset()
        if self.timestamp is not None and timestamp <= self.timestamp:
            return self.previous.copy()
        if self.previous is None or timestamp - self.timestamp > MAX_GAZE_GAP_SECONDS:
            self.previous = value.copy()
        else:
            alpha = (1 - (1 - self.alpha) ** ((timestamp - self.timestamp) * 30)
                     if self.time_aware else self.alpha)
            self.previous = alpha * value + (1.0 - alpha) * self.previous
        self.timestamp = timestamp
        self.last_valid_timestamp = timestamp
        return self.previous.copy()


class GazeSignalFilter:
    """Shared capture/simulation filter for blink recovery and isolated spikes."""

    def __init__(self):
        self.gaze_smoother = GazeSmoother(time_aware=True, missing_grace=DWELL_GRACE_SECONDS)
        self.feature_smoother = GazeSmoother(time_aware=True, missing_grace=DWELL_GRACE_SECONDS)
        self.reset()

    def reset(self):
        self.gaze_smoother.reset()
        self.feature_smoother.reset()
        self.last_raw = None
        self.last_valid_timestamp = None
        self.last_timestamp = None
        self.candidate = None

    def update(self, gaze, features, timestamp):
        if not np.isfinite(timestamp):
            self.reset()
            return None, None, None
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            return None, None, None
        self.last_timestamp = timestamp
        if (self.last_valid_timestamp is not None
                and timestamp - self.last_valid_timestamp > MAX_GAZE_GAP_SECONDS):
            self.last_raw = self.candidate = None
        valid = gaze is not None and features is not None
        if valid:
            gaze, features = np.asarray(gaze, dtype=float), np.asarray(features, dtype=float)
            valid = (gaze.shape == (2,) and features.shape == (FEATURE_DIM,)
                     and np.isfinite(gaze).all() and np.isfinite(features).all())
        if valid and self.last_raw is not None:
            jumped = np.any(np.abs(features - self.last_raw) > FEATURE_JUMP_LIMIT)
            confirmed = (self.candidate is not None
                         and np.all(np.abs(features - self.candidate) <= FEATURE_JUMP_LIMIT * .5))
            if jumped and not confirmed:
                self.candidate = features.copy()
                valid = False
            else:
                if jumped:
                    # A confirmed saccade must not carry the old button's EMA
                    # through dwell completion, especially immediately after a blink.
                    self.gaze_smoother.reset()
                    self.feature_smoother.reset()
                self.candidate = None
        elif not valid:
            self.candidate = None
        if not valid:
            self.gaze_smoother.update(None, timestamp)
            self.feature_smoother.update(None, timestamp)
            return None, None, None
        self.last_raw = features.copy()
        self.last_valid_timestamp = timestamp
        return (self.gaze_smoother.update(gaze, timestamp),
                self.feature_smoother.update(features, timestamp), features.copy())


def ema_series(values, alpha=EMA_ALPHA):
    values = np.asarray(values, dtype=float)
    smoothed = np.empty_like(values)
    previous = values[0]
    for index, value in enumerate(values):
        previous = value if index == 0 else alpha * value + (1 - alpha) * previous
        smoothed[index] = previous
    return smoothed


def _trim(values, scale):
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center), axis=0)
    tolerance = np.maximum(5 * 1.4826 * mad, scale)
    return values[np.all(np.abs(values - center) <= tolerance, axis=1)]


def calibration_sample(values):
    """Validate one time-ordered fixation window and return its inliers.

    Isolated outliers (blink frames, landmark glitches) are trimmed. A window
    is rejected when too few samples remain, or when its two halves differ by
    much more than their own jitter (the eyes jumped to another place).
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < CALIBRATION_MIN_SAMPLES:
        raise ValueError('Недостаточно кадров: продолжайте смотреть на кнопку.')
    if not np.isfinite(values).all():
        raise ValueError('Положение радужки определяется нестабильно.')
    floor = _noise_floor(values.shape[1])
    inliers = _trim(values, 3 * floor)
    if len(inliers) < CALIBRATION_MIN_SAMPLES or len(inliers) < 0.6 * len(values):
        raise ValueError('Взгляд нестабилен: удерживайте глаза на кнопке.')
    first, second = np.array_split(inliers, 2)
    jitter = 1.4826 * np.mean([np.median(np.abs(part - np.median(part, axis=0)), axis=0)
                               for part in (first, second)], axis=0)
    step = np.abs(np.median(first, axis=0) - np.median(second, axis=0))
    if np.any(step > np.maximum(6 * jitter, 5 * floor)):
        raise ValueError('Взгляд сместился: смотрите в центр кнопки.')
    return np.median(inliers, axis=0), inliers


def best_effort_sample(values, timestamps):
    """Most recent settled second of a long, unstable window, outliers trimmed."""
    values = np.asarray(values, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    recent = values[timestamps >= timestamps[-1] - CALIBRATION_SAMPLE_SECONDS]
    if len(recent) < CALIBRATION_MIN_SAMPLES:
        recent = values[-CALIBRATION_MIN_SAMPLES:]
    trimmed = _trim(recent, 3 * _noise_floor(values.shape[1]))
    return trimmed if len(trimmed) >= max(4, len(recent) // 2) else recent


@dataclass(frozen=True)
class GazeClassifier:
    """Shared-covariance Gaussian classifier over calibrated button fixations.

    Each button has a mean feature vector; one pooled covariance captures
    EMA-smoothed jitter, drift between calibration passes and head rotation
    with fixed gaze. A button is entered only with high posterior and kept
    with a lower posterior (hysteresis). Gaze between buttons or away from
    the calibrated grid returns None and never builds dwell.
    """

    means: np.ndarray
    precision: np.ndarray
    outside: float
    separation: np.ndarray
    learned_head_motion: bool = False

    def distances(self, features):
        value = np.asarray(features, dtype=float)
        if value.shape != (self.means.shape[1],) or not np.isfinite(value).all():
            return None
        difference = self.means - value
        return np.einsum('kd,de,ke->k', difference, self.precision, difference)

    def posteriors(self, features):
        distances = self.distances(features)
        if distances is None:
            return None
        weights = np.exp(-0.5 * (distances - distances.min()))
        return weights / weights.sum()

    def nearest_separation(self):
        separation = self.separation + np.diag(np.full(len(self.means), np.inf))
        return float(np.median(separation.min(axis=1)))

    def boundary_fraction(self, distances, index):
        """Largest fraction of the way from button `index` toward any other button.

        0 is the button's calibrated centre, 0.5 the boundary with a neighbour.
        Measured in the classifier metric, it is the feature-space analogue of
        the pixel inset: entry and retention bands scale with button spacing.
        """
        squared = np.maximum(self.separation[index] ** 2, 1e-12)
        fractions = (distances[index] - distances + squared) / (2 * squared)
        fractions[index] = -np.inf
        return float(fractions.max())

    def assess(self, features, current=None):
        """Return (button or None, posterior, boundary fraction) for one frame."""
        distances = self.distances(features)
        if distances is None:
            return None, 0.0, 1.0
        weights = np.exp(-0.5 * (distances - distances.min()))
        posterior = weights / weights.sum()
        if current is not None and 0 <= current < len(distances) and distances[current] <= self.outside:
            fraction = self.boundary_fraction(distances, current)
            if posterior[current] >= RETAIN_POSTERIOR and fraction <= RETAIN_FRACTION:
                return int(current), float(posterior[current]), fraction
        best = int(np.argmin(distances))
        fraction = self.boundary_fraction(distances, best)
        if distances[best] <= self.outside and posterior[best] >= ENTRY_POSTERIOR and fraction <= ENTRY_FRACTION:
            return best, float(posterior[best]), fraction
        return None, float(posterior[best]), fraction

    def classify(self, features, current=None):
        return self.assess(features, current)[0]

    def weak_pairs(self, threshold=WEAK_SEPARATION):
        count = len(self.means)
        return [(i, j) for i in range(count) for j in range(i + 1, count)
                if self.separation[i, j] < threshold]


class GazeTargeting:
    """Runtime wrapper that slowly removes common drift (posture, lighting).

    Only frames that are confident and well inside a button (at most a quarter
    of the way to any neighbour) nudge a shared offset toward that button, so
    gaze resting on a gap between buttons can never be pulled onto one. The
    offset is bounded to a third of the neighbour spacing.
    """

    def __init__(self, model):
        self.model = model
        self.offset = np.zeros(model.means.shape[1])
        self.bound = DRIFT_BOUND * model.nearest_separation()

    def classify(self, features, current=None):
        value = np.asarray(features, dtype=float)
        if value.shape != self.offset.shape or not np.isfinite(value).all():
            return None
        shifted = value - self.offset
        index, posterior, fraction = self.model.assess(shifted, current)
        if index is not None and posterior >= DRIFT_CONFIDENCE and fraction <= DRIFT_FRACTION:
            self.offset = self.offset + DRIFT_RATE * (shifted - self.model.means[index])
            norm = float(np.sqrt(self.offset @ self.model.precision @ self.offset))
            if norm > self.bound:
                self.offset = self.offset * (self.bound / norm)
        return index

    def weak_pairs(self, threshold=WEAK_SEPARATION):
        return self.model.weak_pairs(threshold)


def head_motion_covariance(head_motion, alpha=EMA_ALPHA):
    """Feature covariance of a HEAD_DRIFT_DEGREES head rotation at fixed gaze.

    Learned from the calibration step where the user keeps looking at one
    button while turning the head; scaled by the nose-offset amplitude. Falls
    back to an anatomical prior if the head barely moved.
    """
    reference = (HEAD_DRIFT_DEGREES * HEAD_PER_DEGREE) ** 2
    if head_motion is not None:
        values = np.asarray(head_motion, dtype=float)
        if (values.ndim == 2 and values.shape[1] == FEATURE_DIM
                and len(values) >= CALIBRATION_MIN_SAMPLES and np.isfinite(values).all()):
            smoothed = ema_series(values, alpha)[min(8, len(values) // 3):]
            smoothed = _trim(smoothed, 3 * FEATURE_NOISE_FLOOR)
            if len(smoothed) >= 4:
                covariance = np.cov(smoothed.T)
                head_variance = float(np.mean(np.diag(covariance)[2:]))
                if head_variance >= (MIN_HEAD_SWEEP_DEGREES * HEAD_PER_DEGREE) ** 2:
                    return covariance * (reference / head_variance), True
    yaw = np.array([-IRIS_PER_HEAD_DEGREE[0], 0.0, HEAD_PER_DEGREE, 0.0])
    pitch = np.array([0.0, -IRIS_PER_HEAD_DEGREE[1], 0.0, HEAD_PER_DEGREE])
    return HEAD_DRIFT_DEGREES ** 2 * (np.outer(yaw, yaw) + np.outer(pitch, pitch)), False


def fit_gaze_classifier(windows, head_motion=None, alpha=EMA_ALPHA):
    """Fit from per-button lists of time-ordered raw feature windows (passes).

    Raises ValueError only for missing or non-finite data; covariance floors
    and shrinkage keep the fit well-conditioned for any finite input.
    """
    if len(windows) < 2:
        raise ValueError('Для калибровки нужны как минимум две кнопки.')
    dimension = None
    means, pass_centers, residuals, smoothed_windows = [], [], [], []
    for passes in windows:
        if not passes:
            raise ValueError('Нет измерений для одной из кнопок.')
        centers, smoothed_target = [], []
        for window in passes:
            values = np.asarray(window, dtype=float)
            if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
                raise ValueError('Измерения взгляда повреждены.')
            if dimension is None:
                dimension = values.shape[1]
            elif values.shape[1] != dimension:
                raise ValueError('Измерения взгляда имеют разный формат.')
            centers.append(np.median(values, axis=0))
            smoothed = ema_series(values, alpha)[min(8, len(values) // 3):]
            residuals.append(smoothed - np.median(smoothed, axis=0))
            smoothed_target.append(smoothed)
        centers = np.array(centers)
        means.append(np.median(centers, axis=0) if len(centers) >= 3 else centers.mean(axis=0))
        pass_centers.append(centers)
        smoothed_windows.append(np.concatenate(smoothed_target))
    means = np.array(means)
    residual = np.concatenate(residuals)
    covariance = residual.T @ residual / max(len(residual) - 1, 1)
    deviations = [centers - mean for centers, mean in zip(pass_centers, means) if len(centers) >= 2]
    freedom = sum(len(centers) - 1 for centers in pass_centers if len(centers) >= 2)
    if freedom:
        deviations = np.concatenate(deviations)
        covariance = covariance + deviations.T @ deviations / freedom
    floor = np.maximum(_noise_floor(dimension), 0.05 * np.ptp(means, axis=0))
    np.fill_diagonal(covariance, np.maximum(np.diag(covariance), floor ** 2))
    covariance = 0.85 * covariance + 0.15 * np.diag(np.diag(covariance))
    learned = False
    if dimension == FEATURE_DIM:
        head, learned = head_motion_covariance(head_motion, alpha)
        covariance = covariance + head
    precision = np.linalg.inv(covariance)
    difference = means[:, None] - means[None, :]
    separation = np.sqrt(np.maximum(np.einsum('ijd,de,ije->ij', difference, precision, difference), 0))
    own = []
    for mean, smoothed in zip(means, smoothed_windows):
        offset = smoothed - mean
        own.append(np.einsum('nd,de,ne->n', offset, precision, offset))
    nearest = separation + np.diag(np.full(len(means), np.inf))
    outside = max(float(chi2.ppf(0.999, dimension)), 2.0 * float(np.quantile(np.concatenate(own), 0.99)),
                  (OUTSIDE_SEPARATION * float(np.median(nearest.min(axis=1)))) ** 2)
    return GazeClassifier(means, precision, outside, separation, learned)


def inconsistent_targets(model, windows):
    """Buttons whose individual calibration windows are not recognized as themselves."""
    problems = []
    for index, passes in enumerate(windows):
        for window in passes:
            posterior = model.posteriors(np.median(np.asarray(window, dtype=float), axis=0))
            if posterior is None or int(np.argmax(posterior)) != index or posterior[index] < ENTRY_POSTERIOR:
                problems.append(index)
                break
    return problems


class DwellState(str, Enum):
    IDLE = 'IDLE'
    DWELLING = 'DWELLING'
    TRIGGERED = 'TRIGGERED'
    COOLDOWN = 'COOLDOWN'


@dataclass(frozen=True)
class DwellUpdate:
    state: DwellState
    target: object = None
    elapsed_ms: int = 0
    progress: float = 0.0
    triggered: object = None


class DwellEngine:
    """Timestamp-driven fixation over target rectangles.

    IDLE -> DWELLING (entry needs a point 12 px inside a target; the full
    border retains it) -> TRIGGERED at 900 ms -> COOLDOWN for 500 ms -> IDLE.
    A frame without gaze data (missing=True: blink, landmark dropout) keeps the
    dwell for up to 200 ms but cannot trigger. Gaze outside every target, or
    on a different target, resets it immediately.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.target = None
        self.started = None
        self.last_hit = None
        self.cooldown_until = 0.0
        self.last_timestamp = None
        self.state = DwellState.IDLE

    def cancel_fixation(self):
        self.target = None
        self.started = None
        self.last_hit = None
        self.state = DwellState.COOLDOWN if self.state == DwellState.COOLDOWN else DwellState.IDLE

    @staticmethod
    def _contains(point, rect, inset=0):
        left, top, right, bottom = rect
        return left + inset <= point[0] <= right - inset and top + inset <= point[1] <= bottom - inset

    def _progress(self, timestamp):
        elapsed = max(0.0, (timestamp - self.started) * 1000)
        return elapsed, min(elapsed / DWELL_THRESHOLD_MS, 1.0)

    def update(self, position, targets, timestamp, missing=False):
        if not np.isfinite(timestamp):
            self.cancel_fixation()
            return DwellUpdate(self.state)
        if self.last_timestamp is not None:
            if timestamp <= self.last_timestamp:
                return DwellUpdate(self.state, self.target)
            if timestamp - self.last_timestamp > MAX_GAZE_GAP_SECONDS:
                self.cancel_fixation()
        self.last_timestamp = timestamp
        if timestamp < self.cooldown_until:
            self.cancel_fixation()
            self.state = DwellState.COOLDOWN
            return DwellUpdate(self.state)
        if self.state == DwellState.COOLDOWN:
            self.state = DwellState.IDLE
        target = None
        if position is not None and np.asarray(position).shape == (2,) and np.isfinite(position).all():
            if self.target in targets and self._contains(position, targets[self.target]):
                target = self.target
            else:
                target = next((key for key, rect in targets.items()
                               if self._contains(position, rect, DEADZONE_PX)), None)
        if target is None:
            if (missing and position is None and self.target in targets and self.last_hit is not None
                    and timestamp - self.last_hit <= DWELL_GRACE_SECONDS):
                elapsed, progress = self._progress(timestamp)
                return DwellUpdate(DwellState.DWELLING, self.target, int(elapsed + 1e-7), progress)
            self.cancel_fixation()
            return DwellUpdate(self.state)
        if target != self.target:
            self.target, self.started = target, timestamp
        self.last_hit = timestamp
        elapsed, progress = self._progress(timestamp)
        if elapsed + 1e-7 >= DWELL_THRESHOLD_MS:
            self.cooldown_until = timestamp + COOLDOWN_MS / 1000
            self.target = self.started = self.last_hit = None
            self.state = DwellState.COOLDOWN
            return DwellUpdate(DwellState.TRIGGERED, target, DWELL_THRESHOLD_MS, 1.0, target)
        self.state = DwellState.DWELLING
        return DwellUpdate(self.state, target, int(elapsed + 1e-7), progress)
