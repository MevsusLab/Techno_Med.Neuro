# модуль трекинга взгляда и фиксации (dwell)

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import chi2


ema_alpha = 0.25
dwell_threshold_ms = 900
cooldown_ms = 500
deadzone_px = 12
# задержка на моргание
max_gaze_gap_seconds = 0.35
# держим удержание при потере кадра
dwell_grace_seconds = 0.2

calib_settle_sec = 0.7
calib_sample_sec = 1.0
calib_min_samples = 16
calib_max_window_sec = 3.0
# если долго сидим на кнопке, принимаем лучшие кадры
calib_accept_sec = 6.0
calib_gap_sec = 0.5

eye_open_ratio = 0.07
entry_posterior = 0.75
retain_posterior = 0.35
# границы между соседними кнопками
entry_fraction = 0.4
retain_fraction = 0.5
weak_separation = 2.5
feature_dim = 4
# порог шума признаков
feature_noise_floor = np.array([0.004, 0.003, 0.004, 0.004])

# компенсация вестибуло-окулярного рефлекса
head_drift_degrees = 1.5
head_per_degree = 0.0055
iris_per_head_degree = np.array([0.0034, 0.0036])
min_head_sweep_degrees = 1.0
head_sweep_seconds = 3.0
# выход за границы сетки
outside_separation = 1.0
# плавная подстройка дрейфа
drift_rate = 0.02
drift_confidence = 0.9
drift_fraction = 0.25
drift_bound = 0.35
# подавление резких скачков фичей
feature_jump_limit = np.array([0.06, 0.04, 0.05, 0.05])

# индексы точек левого и правого глаза
left_eye = (468, 133, 33, 159, 145)
right_eye = (473, 362, 263, 386, 374)
nose_tip = 1

# алиасы для совместимости с тестами
EMA_ALPHA = ema_alpha
DWELL_THRESHOLD_MS = dwell_threshold_ms
COOLDOWN_MS = cooldown_ms
DEADZONE_PX = deadzone_px
MAX_GAZE_GAP_SECONDS = max_gaze_gap_seconds
DWELL_GRACE_SECONDS = dwell_grace_seconds
CALIBRATION_SETTLE_SECONDS = calib_settle_sec
CALIBRATION_SAMPLE_SECONDS = calib_sample_sec
CALIBRATION_MIN_SAMPLES = calib_min_samples
CALIBRATION_MAX_WINDOW_SECONDS = calib_max_window_sec
CALIBRATION_ACCEPT_SECONDS = calib_accept_sec
CALIBRATION_GAP_SECONDS = calib_gap_sec
EYE_OPEN_RATIO = eye_open_ratio
ENTRY_POSTERIOR = entry_posterior
RETAIN_POSTERIOR = retain_posterior
ENTRY_FRACTION = entry_fraction
RETAIN_FRACTION = retain_fraction
WEAK_SEPARATION = weak_separation
FEATURE_DIM = feature_dim
FEATURE_NOISE_FLOOR = feature_noise_floor
HEAD_DRIFT_DEGREES = head_drift_degrees
HEAD_PER_DEGREE = head_per_degree
IRIS_PER_HEAD_DEGREE = iris_per_head_degree
MIN_HEAD_SWEEP_DEGREES = min_head_sweep_degrees
HEAD_SWEEP_SECONDS = head_sweep_seconds
OUTSIDE_SEPARATION = outside_separation
DRIFT_RATE = drift_rate
DRIFT_CONFIDENCE = drift_confidence
DRIFT_FRACTION = drift_fraction
DRIFT_BOUND = drift_bound
FEATURE_JUMP_LIMIT = feature_jump_limit
LEFT_EYE = left_eye
RIGHT_EYE = right_eye
NOSE_TIP = nose_tip


def _noise_floor(dim):
    return feature_noise_floor if dim == feature_dim else np.full(dim, 0.003)


def gaze_features(landmarks, pose=None):
    # нормализованные координаты зрачка относительно углов глаза
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
    for iris, inner, outer, upper, lower in (left_eye, right_eye):
        corners = points[[inner, outer], :2]
        left, right = corners[np.argsort(corners[:, 0])]
        axis, normal, width = _corner_frame(left, right)
        height = float(np.dot(points[lower, :2] - points[upper, :2], normal))
        if height <= width * eye_open_ratio:
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
    # 4d вектор признаков: координаты зрачка и наклон носа
    points = np.asarray(landmarks, dtype=float)
    gaze_features(points)
    eyes = []
    for iris, inner, outer, _, _ in (left_eye, right_eye):
        corners = points[[inner, outer], :2]
        left, right = corners[np.argsort(corners[:, 0])]
        axis, normal, width = _corner_frame(left, right)
        iris_point = points[iris, :2]
        eyes.append([np.dot(iris_point - left, axis) / width,
                     0.5 + np.dot(iris_point - (left + right) / 2, normal) / width])
    outer = points[[left_eye[2], right_eye[2]], :2]
    left, right = outer[np.argsort(outer[:, 0])]
    axis, normal, iod = _corner_frame(left, right)
    middle = points[[left_eye[1], left_eye[2], right_eye[1], right_eye[2]], :2].mean(axis=0)
    nose = points[nose_tip, :2] - middle
    head = [np.dot(nose, axis) / iod, np.dot(nose, normal) / iod]
    features = np.concatenate([np.mean(eyes, axis=0), head])
    if not np.isfinite(features).all():
        raise ValueError('Надёжное положение радужки не найдено.')
    return features


class GazeSmoother:
    # сглаживание фильтром экспоненциального скользящего среднего

    def __init__(self, alpha=ema_alpha, time_aware=False, missing_grace=0.0):
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
        if self.previous is None or timestamp - self.timestamp > max_gaze_gap_seconds:
            self.previous = value.copy()
        else:
            alpha = (1 - (1 - self.alpha) ** ((timestamp - self.timestamp) * 30)
                     if self.time_aware else self.alpha)
            self.previous = alpha * value + (1.0 - alpha) * self.previous
        self.timestamp = timestamp
        self.last_valid_timestamp = timestamp
        return self.previous.copy()


class GazeSignalFilter:
    # общий фильтр против морганий и выбросов трекера

    def __init__(self):
        self.gaze_smoother = GazeSmoother(time_aware=True, missing_grace=dwell_grace_seconds)
        self.feature_smoother = GazeSmoother(time_aware=True, missing_grace=dwell_grace_seconds)
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
                and timestamp - self.last_valid_timestamp > max_gaze_gap_seconds):
            self.last_raw = self.candidate = None
        valid = gaze is not None and features is not None
        if valid:
            gaze, features = np.asarray(gaze, dtype=float), np.asarray(features, dtype=float)
            valid = (gaze.shape == (2,) and features.shape == (feature_dim,)
                     and np.isfinite(gaze).all() and np.isfinite(features).all())
        if valid and self.last_raw is not None:
            jumped = np.any(np.abs(features - self.last_raw) > feature_jump_limit)
            confirmed = (self.candidate is not None
                         and np.all(np.abs(features - self.candidate) <= feature_jump_limit * .5))
            if jumped and not confirmed:
                self.candidate = features.copy()
                valid = False
            else:
                if jumped:
                    # сброс фильтра при скачке фичей (саккада)
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


def ema_series(values, alpha=ema_alpha):
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
    # проверка стабильности фиксации в окне
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < calib_min_samples:
        raise ValueError('Недостаточно кадров: продолжайте смотреть на кнопку.')
    if not np.isfinite(values).all():
        raise ValueError('Положение радужки определяется нестабильно.')
    floor = _noise_floor(values.shape[1])
    inliers = _trim(values, 3 * floor)
    if len(inliers) < calib_min_samples or len(inliers) < 0.6 * len(values):
        raise ValueError('Взгляд нестабилен: удерживайте глаза на кнопке.')
    first, second = np.array_split(inliers, 2)
    jitter = 1.4826 * np.mean([np.median(np.abs(part - np.median(part, axis=0)), axis=0)
                                for part in (first, second)], axis=0)
    step = np.abs(np.median(first, axis=0) - np.median(second, axis=0))
    if np.any(step > np.maximum(6 * jitter, 5 * floor)):
        raise ValueError('Взгляд сместился: смотрите в центр кнопки.')
    return np.median(inliers, axis=0), inliers


def best_effort_sample(values, timestamps):
    # выборка отсчетов с отсевом выбросов
    values = np.asarray(values, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    recent = values[timestamps >= timestamps[-1] - calib_sample_sec]
    if len(recent) < calib_min_samples:
        recent = values[-calib_min_samples:]
    trimmed = _trim(recent, 3 * _noise_floor(values.shape[1]))
    return trimmed if len(trimmed) >= max(4, len(recent) // 2) else recent


@dataclass(frozen=True)
class GazeClassifier:
    # байесовский классификатор по центрам кнопок

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
        squared = np.maximum(self.separation[index] ** 2, 1e-12)
        fractions = (distances[index] - distances + squared) / (2 * squared)
        fractions[index] = -np.inf
        return float(fractions.max())

    def assess(self, features, current=None):
        # проверка к какой кнопке ближе взгляд
        distances = self.distances(features)
        if distances is None:
            return None, 0.0, 1.0
        weights = np.exp(-0.5 * (distances - distances.min()))
        posterior = weights / weights.sum()
        if current is not None and 0 <= current < len(distances) and distances[current] <= self.outside:
            fraction = self.boundary_fraction(distances, current)
            if posterior[current] >= retain_posterior and fraction <= retain_fraction:
                return int(current), float(posterior[current]), fraction
        best = int(np.argmin(distances))
        fraction = self.boundary_fraction(distances, best)
        if distances[best] <= self.outside and posterior[best] >= entry_posterior and fraction <= entry_fraction:
            return best, float(posterior[best]), fraction
        return None, float(posterior[best]), fraction

    def classify(self, features, current=None):
        return self.assess(features, current)[0]

    def weak_pairs(self, threshold=weak_separation):
        count = len(self.means)
        return [(i, j) for i in range(count) for j in range(i + 1, count)
                if self.separation[i, j] < threshold]


class GazeTargeting:
    # компенсация медленного дрейфа взгляда

    def __init__(self, model):
        self.model = model
        self.offset = np.zeros(model.means.shape[1])
        self.bound = drift_bound * model.nearest_separation()

    def classify(self, features, current=None):
        value = np.asarray(features, dtype=float)
        if value.shape != self.offset.shape or not np.isfinite(value).all():
            return None
        shifted = value - self.offset
        index, posterior, fraction = self.model.assess(shifted, current)
        if index is not None and posterior >= drift_confidence and fraction <= drift_fraction:
            self.offset = self.offset + drift_rate * (shifted - self.model.means[index])
            norm = float(np.sqrt(self.offset @ self.model.precision @ self.offset))
            if norm > self.bound:
                self.offset = self.offset * (self.bound / norm)
        return index

    def weak_pairs(self, threshold=weak_separation):
        return self.model.weak_pairs(threshold)


def head_motion_covariance(head_motion, alpha=ema_alpha):
    # расчет матрицы ковариации при повороте головы
    reference = (head_drift_degrees * head_per_degree) ** 2
    if head_motion is not None:
        values = np.asarray(head_motion, dtype=float)
        if (values.ndim == 2 and values.shape[1] == feature_dim
                and len(values) >= calib_min_samples and np.isfinite(values).all()):
            smoothed = ema_series(values, alpha)[min(8, len(values) // 3):]
            smoothed = _trim(smoothed, 3 * feature_noise_floor)
            if len(smoothed) >= 4:
                covariance = np.cov(smoothed.T)
                head_variance = float(np.mean(np.diag(covariance)[2:]))
                if head_variance >= (min_head_sweep_degrees * head_per_degree) ** 2:
                    return covariance * (reference / head_variance), True
    yaw = np.array([-iris_per_head_degree[0], 0.0, head_per_degree, 0.0])
    pitch = np.array([0.0, -iris_per_head_degree[1], 0.0, head_per_degree])
    return head_drift_degrees ** 2 * (np.outer(yaw, yaw) + np.outer(pitch, pitch)), False


def fit_gaze_classifier(windows, head_motion=None, alpha=ema_alpha):
    # подгонка классификатора по точкам калибровки
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
    if dimension == feature_dim:
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
                  (outside_separation * float(np.median(nearest.min(axis=1)))) ** 2)
    return GazeClassifier(means, precision, outside, separation, learned)


def inconsistent_targets(model, windows):
    # поиск кнопок с конфликтной калибровкой
    problems = []
    for index, passes in enumerate(windows):
        for window in passes:
            posterior = model.posteriors(np.median(np.asarray(window, dtype=float), axis=0))
            if posterior is None or int(np.argmax(posterior)) != index or posterior[index] < entry_posterior:
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
    # автомат состояний удержания взгляда (dwell)

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
        return elapsed, min(elapsed / dwell_threshold_ms, 1.0)

    def update(self, position, targets, timestamp, missing=False):
        if not np.isfinite(timestamp):
            self.cancel_fixation()
            return DwellUpdate(self.state)
        if self.last_timestamp is not None:
            if timestamp <= self.last_timestamp:
                return DwellUpdate(self.state, self.target)
            if timestamp - self.last_timestamp > max_gaze_gap_seconds:
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
                               if self._contains(position, rect, deadzone_px)), None)
        if target is None:
            if (missing and position is None and self.target in targets and self.last_hit is not None
                    and timestamp - self.last_hit <= dwell_grace_seconds):
                elapsed, progress = self._progress(timestamp)
                return DwellUpdate(DwellState.DWELLING, self.target, int(elapsed + 1e-7), progress)
            self.cancel_fixation()
            return DwellUpdate(self.state)
        if target != self.target:
            self.target, self.started = target, timestamp
        self.last_hit = timestamp
        elapsed, progress = self._progress(timestamp)
        if elapsed + 1e-7 >= dwell_threshold_ms:
            self.cooldown_until = timestamp + cooldown_ms / 1000
            self.target = self.started = self.last_hit = None
            self.state = DwellState.COOLDOWN
            return DwellUpdate(DwellState.TRIGGERED, target, dwell_threshold_ms, 1.0, target)
        self.state = DwellState.DWELLING
        return DwellUpdate(self.state, target, int(elapsed + 1e-7), progress)
