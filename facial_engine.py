# пары симметричных точек лица
import numpy as np

pairs = {
    'mouth': [(61, 291), (78, 308)],
    'eyes': [(159, 386), (145, 374), (33, 263)],
    'brows': [(70, 300), (105, 334)],
    'cheeks': [(98, 327), (205, 425)],
}
PAIRS = pairs


def normalize_face(landmarks, pose):
    points = np.asarray(landmarks, dtype=float)
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] < 2 or not np.isfinite(points).all():
        raise ValueError('Face landmarks are missing or invalid.')
    if pose is None or len(pose) != 3 or not np.isfinite(pose).all():
        raise ValueError('Head pose is unavailable.')
    yaw, pitch, roll = pose
    if abs(yaw) > 12 or abs(pitch) > 12 or abs(roll) > 25:
        raise ValueError('Face the camera directly; head rotation exceeds the measurement limit.')
    eyes = points[[33, 263], :2]
    width = np.linalg.norm(eyes[1] - eyes[0])
    if width < 90:
        raise ValueError('Move closer to the camera; the face is too small.')
    center = eyes.mean(axis=0)
    horizontal = (eyes[1] - eyes[0]) / width
    vertical = np.array([-horizontal[1], horizontal[0]])
    if np.dot(points[152, :2] - center, vertical) < 0:
        vertical = -vertical
    aligned = np.column_stack(((points[:, :2] - center) @ horizontal, (points[:, :2] - center) @ vertical)) / width
    if not np.isfinite(aligned).all():
        raise ValueError('Face alignment failed.')
    middle = aligned[[9, 168, 1, 152]]
    if np.ptp(middle[:, 1]) < 0.25:
        raise ValueError('Face midline is degenerate.')
    axis_slope, axis_offset = np.polyfit(middle[:, 1], middle[:, 0], 1)
    if abs(axis_slope) > 0.25:
        raise ValueError('Face midline is unstable; retake the frame facing the camera.')
    normal = np.array([1.0, -axis_slope]) / np.hypot(1, axis_slope)
    tangent = np.array([axis_slope, 1.0]) / np.hypot(1, axis_slope)
    centered = aligned - [axis_offset, 0]
    side = centered @ normal
    along_axis = centered @ tangent
    if np.any(side[[61, 33]] >= 0) or np.any(side[[291, 263]] <= 0):
        raise ValueError('Face orientation or landmark ordering is unreliable.')
    regions = {}
    for region, pair_list in pairs.items():
        errors = []
        for right, left in pair_list:
            errors.append(100 * np.hypot(abs(side[right]) - abs(side[left]), along_axis[right] - along_axis[left]))
        regions[region] = float(np.median(errors))
    eye_widths = (np.linalg.norm(aligned[33] - aligned[133]), np.linalg.norm(aligned[263] - aligned[362]))
    if min(eye_widths) < 0.08:
        raise ValueError('Eye landmarks are unreliable.')
    eye_right = np.linalg.norm(aligned[159] - aligned[145]) / eye_widths[0]
    eye_left = np.linalg.norm(aligned[386] - aligned[374]) / eye_widths[1]
    brow_right = (aligned[33, 1] - aligned[70, 1])
    brow_left = (aligned[263, 1] - aligned[300, 1])
    if not np.isfinite([eye_right, eye_left, brow_right, brow_left]).all():
        raise ValueError('Eye or eyebrow measurements failed.')
    if min(eye_right, eye_left) < 0.10:
        raise ValueError('Eyes are closed or too occluded to measure; repeat with open eyes.')
    return {
        'points': aligned, 'width_px': float(width),
        'axis': (float(axis_slope), float(axis_offset)), 'regions': regions,
        'aggregate': float(np.mean(list(regions.values()))),
        'mouth': (float(abs(side[61])), float(abs(side[291]))),
        'mouth_position': np.column_stack((side[[61, 291]], along_axis[[61, 291]])),
        'eye': (float(eye_right), float(eye_left)),
        'brow': (float(brow_right), float(brow_left)),
        'cheek': (float(abs(side[205])), float(abs(side[425]))),
    }


def analyze_stage(samples, start, end):
    if not np.isfinite([start, end]).all() or end - start < 2 or len(samples) < 20:
        raise ValueError('Insufficient tracked face frames; repeat this expression.')
    t = np.asarray([sample['timestamp'] for sample in samples], dtype=float)
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or t[0] < start - 0.001 or t[-1] > end + 0.001:
        raise ValueError('Stage timestamps are invalid or outside the recording interval.')
    if t[0] - start > 0.25 or end - t[-1] > 0.25 or np.max(np.diff(t)) > 0.3:
        raise ValueError(
            f'Face tracking was interrupted (start gap {max(0, t[0] - start):.2f}s, '
            f'end gap {max(0, end - t[-1]):.2f}s, largest gap {np.max(np.diff(t)):.2f}s); repeat this expression.'
        )
    analyses = [normalize_face(sample['landmarks'], sample['pose']) for sample in samples]
    poses = np.asarray([sample['pose'] for sample in samples])
    if np.max(np.ptp(poses, axis=0)) > 8:
        raise ValueError('Head pose changed during the expression; repeat facing the camera.')
    if np.std([value['width_px'] for value in analyses]) / np.median([value['width_px'] for value in analyses]) > 0.08:
        raise ValueError('Camera distance or head pose changed during capture.')
    return {
        'regions': {key: float(np.median([item['regions'][key] for item in analyses])) for key in pairs},
        'aggregate': float(np.median([item['aggregate'] for item in analyses])),
        'points': np.median([item['points'] for item in analyses], axis=0),
        'mouth': tuple(np.median([item['mouth'] for item in analyses], axis=0)),
        'mouth_position': np.median([item['mouth_position'] for item in analyses], axis=0),
        'eye': tuple(np.median([item['eye'] for item in analyses], axis=0)),
        'brow': tuple(np.median([item['brow'] for item in analyses], axis=0)),
        'cheek': tuple(np.median([item['cheek'] for item in analyses], axis=0)),
        'samples': len(samples),
        'pose': np.median(poses, axis=0),
        'width_px': float(np.median([item['width_px'] for item in analyses])),
        'axis': np.median([item['axis'] for item in analyses], axis=0),
        'rate': float((len(t) - 1) / (t[-1] - t[0])),
    }


def compare_stages(rest, smile, brows):
    for expression in (smile, brows):
        if np.max(np.abs(expression['pose'] - rest['pose'])) > 5 or abs(expression['width_px'] / rest['width_px'] - 1) > 0.08:
            raise ValueError('Между этапами изменилось положение головы или расстояние до камеры. Повторите весь тест.')
    smile_motion = np.linalg.norm(smile['mouth_position'] - rest['mouth_position'], axis=1)
    brow_motion = np.abs(np.asarray(brows['brow']) - rest['brow'])
    if sum(smile_motion) < 0.02 or sum(brow_motion) < 0.025:
        missing = []
        if sum(smile_motion) < 0.02:
            missing.append(f'движение уголков рта {sum(smile_motion) * 100:.2f}% (минимум 2,00%)')
        if sum(brow_motion) < 0.025:
            missing.append(f'движение бровей {sum(brow_motion) * 100:.2f}% (минимум 2,50%)')
        raise ValueError('Мимическое движение слишком мало: ' + '; '.join(missing) + '. Следуйте подсказкам для улыбки и поднятия бровей. Это технические пределы измерения, а не клинические пороги.')
    imbalance_smile = 100 * abs(smile_motion[0] - smile_motion[1]) / sum(smile_motion)
    imbalance_brows = 100 * abs(brow_motion[0] - brow_motion[1]) / sum(brow_motion)
    return {
        'rest': rest, 'smile': smile, 'brows': brows,
        'smile_imbalance': float(imbalance_smile), 'brow_imbalance': float(imbalance_brows),
        'asymmetry_index': float(np.mean([rest['aggregate'], smile['aggregate'], brows['aggregate']])),
    }
