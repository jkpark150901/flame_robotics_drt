"""
calibration/solver.py
=====================
순수 수학/IO 함수만 포함. NatNet, rbpodo, asyncio 의존 없음.

모델 컨벤션:
  T_base_tcp_i = T_base_motive @ T_motive_rb_i @ T_rb_tcp
"""

import csv
import json
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

# =============================================================================
# 변환 유틸리티
# =============================================================================

def quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=float)
    n = x*x + y*y + z*z + w*w
    if n <= 0.0:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    return np.array([
        [1.0-(yy+zz), xy-wz,     xz+wy    ],
        [xy+wz,       1.0-(xx+zz), yz-wx  ],
        [xz-wy,       yz+wx,     1.0-(xx+yy)],
    ])


def average_quaternions_xyzw(quats: list[np.ndarray]) -> np.ndarray:
    q0 = np.asarray(quats[0], dtype=float)
    q0 /= np.linalg.norm(q0)
    aligned = []
    for q in quats:
        q = np.asarray(q, dtype=float) / np.linalg.norm(q)
        if np.dot(q, q0) < 0:
            q = -q
        aligned.append(q)
    q_avg = np.mean(aligned, axis=0)
    return q_avg / np.linalg.norm(q_avg)


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=float)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.eye(3)
    axis = rotvec / angle
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)


def rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2*np.sin(angle))
    return axis * angle


def average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    quats = []
    for R in rotations:
        tr = np.trace(R)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2.0
            q = np.array([(R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s, 0.25*s])
        else:
            i = int(np.argmax(np.diag(R)))
            if i == 0:
                s = np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2
                q = np.array([0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s])
            elif i == 1:
                s = np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2
                q = np.array([(R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s])
            else:
                s = np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2
                q = np.array([(R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s, (R[1,0]-R[0,1])/s])
        q /= np.linalg.norm(q)
        if quats and np.dot(q, quats[0]) < 0:
            q = -q
        quats.append(q)
    q_avg = np.mean(quats, axis=0)
    return quat_xyzw_to_matrix(q_avg / np.linalg.norm(q_avg))


def average_transforms(T_list: list[np.ndarray]) -> np.ndarray:
    T_avg = np.eye(4)
    T_avg[:3, :3] = average_rotations([T[:3, :3] for T in T_list])
    T_avg[:3, 3] = np.mean([T[:3, 3] for T in T_list], axis=0)
    return T_avg


def invert_transform(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def se3_from_vec(x: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = x[:3]
    T[:3, :3] = matrix_from_rotvec(x[3:6])
    return T


def vec_from_se3(T: np.ndarray) -> np.ndarray:
    x = np.zeros(6)
    x[:3] = T[:3, 3]
    x[3:6] = rotvec_from_matrix(T[:3, :3])
    return x


def tcp_raw_to_matrix(tcp_raw: np.ndarray, orientation_type: str = 'zyx_euler_deg') -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(tcp_raw[:3], dtype=float) / 1000.0
    r = np.asarray(tcp_raw[3:6], dtype=float)
    if orientation_type == 'zyx_euler_deg':
        rx, ry, rz = np.radians(r)
        T[:3, :3] = _rot_z(rz) @ _rot_y(ry) @ _rot_x(rx)
    elif orientation_type == 'xyz_euler_deg':
        rx, ry, rz = np.radians(r)
        T[:3, :3] = _rot_x(rx) @ _rot_y(ry) @ _rot_z(rz)
    elif orientation_type == 'rotvec_deg':
        T[:3, :3] = matrix_from_rotvec(np.radians(r))
    elif orientation_type == 'rotvec_rad':
        T[:3, :3] = matrix_from_rotvec(r)
    else:
        raise ValueError(f'unknown orientation_type: {orientation_type}')
    return T


def rb_pose_to_matrix(pos_m: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(pos_m, dtype=float)
    T[:3, :3] = quat_xyzw_to_matrix(quat_xyzw)
    return T


# =============================================================================
# SVD (point cloud) 캘리브레이션
# =============================================================================

def _compute_svd_transform(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    cA, cB = np.mean(A, axis=0), np.mean(B, axis=0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cB - R @ cA
    return T


def _rmse_mm(T: np.ndarray, pts_src: np.ndarray, pts_dst: np.ndarray) -> float:
    aligned = (T[:3, :3] @ pts_src.T).T + T[:3, 3]
    return float(np.sqrt(np.mean(np.linalg.norm(pts_dst - aligned, axis=1)**2)) * 1000.0)


def _residuals_mm(T: np.ndarray, pts_src: np.ndarray, pts_dst: np.ndarray) -> np.ndarray:
    aligned = (T[:3, :3] @ pts_src.T).T + T[:3, 3]
    return np.linalg.norm(pts_dst - aligned, axis=1) * 1000.0


def compute_T_align_svd(points_motive: np.ndarray, points_tcp: np.ndarray) -> np.ndarray:
    return _compute_svd_transform(points_motive, points_tcp)


def compute_T_align_with_rb_offset(
    points_motive: np.ndarray,
    quat_motive: np.ndarray,
    points_tcp: np.ndarray,
    max_iter: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_motive = np.asarray(points_motive, dtype=float)
    points_tcp = np.asarray(points_tcp, dtype=float)
    rotations = np.array([quat_xyzw_to_matrix(q) for q in quat_motive])

    T_align = _compute_svd_transform(points_motive, points_tcp)
    rb_to_tcp_offset = np.zeros(3)
    corrected = points_motive.copy()

    for _ in range(max_iter):
        R_a, t_a = T_align[:3, :3], T_align[:3, 3]
        lhs = [R_a @ R for R in rotations]
        rhs = [p_tcp - (R_a @ p + t_a) for p, p_tcp in zip(points_motive, points_tcp)]
        rb_to_tcp_offset, *_ = np.linalg.lstsq(np.vstack(lhs), np.hstack(rhs), rcond=None)
        corrected = np.array([p + R @ rb_to_tcp_offset for p, R in zip(points_motive, rotations)])
        T_align = _compute_svd_transform(corrected, points_tcp)

    return T_align, rb_to_tcp_offset, corrected


def solve_calibration(
    points_motive: np.ndarray,
    quat_motive: np.ndarray,
    points_tcp: np.ndarray,
    use_rb_offset: bool,
    outlier_threshold_mm: float,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    if use_rb_offset:
        T_align, rb_to_tcp_offset, corrected = compute_T_align_with_rb_offset(points_motive, quat_motive, points_tcp)
    else:
        T_align = compute_T_align_svd(points_motive, points_tcp)
        rb_to_tcp_offset = None
        corrected = points_motive

    residuals = _residuals_mm(T_align, corrected, points_tcp)
    inlier_mask = np.ones(len(points_motive), dtype=bool)
    if outlier_threshold_mm > 0.0:
        inlier_mask = residuals <= outlier_threshold_mm
        for idx in np.where(~inlier_mask)[0]:
            log.warning("캘리브레이션 outlier 제외: %s residual=%.1f mm", labels[idx], residuals[idx])
        if np.count_nonzero(inlier_mask) >= 6 and np.any(~inlier_mask):
            if use_rb_offset:
                T_align, rb_to_tcp_offset, corrected = compute_T_align_with_rb_offset(
                    points_motive[inlier_mask], quat_motive[inlier_mask], points_tcp[inlier_mask])
            else:
                T_align = compute_T_align_svd(points_motive[inlier_mask], points_tcp[inlier_mask])
                rb_to_tcp_offset = None
                corrected = points_motive[inlier_mask]
        elif np.count_nonzero(inlier_mask) < 6:
            log.warning("outlier 제외 후 inlier가 %d개뿐이라 전체 샘플로 계산합니다.", np.count_nonzero(inlier_mask))
            inlier_mask[:] = True

    return T_align, rb_to_tcp_offset, corrected, inlier_mask


# =============================================================================
# Hand-eye 캘리브레이션
# =============================================================================

OPENCV_HANDEYE_METHODS: dict[str, int] = {
    'tsai':       cv2.CALIB_HAND_EYE_TSAI,
    'park':       cv2.CALIB_HAND_EYE_PARK,
    'horaud':     cv2.CALIB_HAND_EYE_HORAUD,
    'andreff':    cv2.CALIB_HAND_EYE_ANDREFF,
    'daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def solve_handeye_opencv(
    T_motive_rb_list: list[np.ndarray],
    T_base_tcp_list: list[np.ndarray],
    method: int = cv2.CALIB_HAND_EYE_PARK,
) -> tuple[np.ndarray, np.ndarray]:
    """
    OpenCV calibrateHandEye 매핑:
      base=M(mocap world), gripper=RB, camera=TCP, target=B(robot base)
    반환: (T_base_motive, T_rb_tcp)
    """
    if len(T_motive_rb_list) < 4:
        raise ValueError("hand-eye calibration에는 최소 4개 pose가 필요합니다.")

    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for T_rb, T_tcp in zip(T_motive_rb_list, T_base_tcp_list):
        R_g2b.append(T_rb[:3, :3].astype(np.float64))
        t_g2b.append(T_rb[:3, 3].reshape(3, 1).astype(np.float64))
        T_inv = invert_transform(T_tcp)
        R_t2c.append(T_inv[:3, :3].astype(np.float64))
        t_t2c.append(T_inv[:3, 3].reshape(3, 1).astype(np.float64))

    R_rb_tcp, t_rb_tcp = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
    T_rb_tcp = np.eye(4)
    T_rb_tcp[:3, :3] = R_rb_tcp
    T_rb_tcp[:3, 3] = t_rb_tcp.reshape(3)

    T_base_motive = average_transforms([
        T_tcp @ invert_transform(T_rb @ T_rb_tcp)
        for T_rb, T_tcp in zip(T_motive_rb_list, T_base_tcp_list)
    ])
    return T_base_motive, T_rb_tcp


def handeye_residuals(
    T_base_motive: np.ndarray,
    T_rb_tcp: np.ndarray,
    T_motive_rb_list: list[np.ndarray],
    T_base_tcp_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    pos_res, rot_res = [], []
    for T_rb, T_tcp in zip(T_motive_rb_list, T_base_tcp_list):
        T_err = invert_transform(T_tcp) @ (T_base_motive @ T_rb @ T_rb_tcp)
        pos_res.append(np.linalg.norm(T_err[:3, 3]) * 1000.0)
        rot_res.append(np.linalg.norm(rotvec_from_matrix(T_err[:3, :3])) * 180.0 / np.pi)
    return np.asarray(pos_res), np.asarray(rot_res)


def handeye_absolute_residual(
    x: np.ndarray,
    T_motive_rb_list: list[np.ndarray],
    T_base_tcp_list: list[np.ndarray],
    rot_weight: float = 0.1,
) -> np.ndarray:
    T_bm = se3_from_vec(x[:6])
    T_rt = se3_from_vec(x[6:12])
    res = []
    for T_rb, T_tcp in zip(T_motive_rb_list, T_base_tcp_list):
        T_err = invert_transform(T_tcp) @ (T_bm @ T_rb @ T_rt)
        res.extend(T_err[:3, 3])
        res.extend(rot_weight * rotvec_from_matrix(T_err[:3, :3]))
    return np.asarray(res)


def solve_handeye_absolute_ls(
    T_motive_rb_list: list[np.ndarray],
    T_base_tcp_list: list[np.ndarray],
    T_base_motive_init: np.ndarray,
    T_rb_tcp_init: np.ndarray,
    rot_weight: float = 0.1,
    f_scale: float = 0.01,
):
    from scipy.optimize import least_squares
    x0 = np.concatenate([vec_from_se3(T_base_motive_init), vec_from_se3(T_rb_tcp_init)])
    result = least_squares(
        handeye_absolute_residual, x0,
        args=(T_motive_rb_list, T_base_tcp_list, rot_weight),
        loss='huber', f_scale=f_scale, max_nfev=2000,
    )
    return se3_from_vec(result.x[:6]), se3_from_vec(result.x[6:12]), result


# =============================================================================
# 저장 / 로드
# =============================================================================

def save_calibration(T_align: np.ndarray, path: str, rb_to_tcp_offset: np.ndarray | None = None):
    payload = {
        'T_base_motive': T_align.tolist(),
        'T_align': T_align.tolist(),
        'convention': {
            'T_base_tcp': '^B T_TCP',
            'T_motive_rb': '^M T_RB',
            'T_base_motive': '^B T_M',
            'T_rb_tcp': '^RB T_TCP',
            'model': 'T_base_tcp = T_base_motive @ T_motive_rb @ T_rb_tcp',
        },
    }
    if rb_to_tcp_offset is not None:
        rb_to_tcp_offset = np.asarray(rb_to_tcp_offset, dtype=float)
        if rb_to_tcp_offset.shape == (4, 4):
            payload['T_rb_tcp'] = rb_to_tcp_offset.tolist()
            payload['T_rigidbody_tcp'] = rb_to_tcp_offset.tolist()
            payload['rb_to_tcp_offset_m'] = rb_to_tcp_offset[:3, 3].tolist()
        else:
            payload['rb_to_tcp_offset_m'] = rb_to_tcp_offset.tolist()
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    log.info("캘리브레이션 저장 완료 → %s", path)


def apply_calibration_and_save(
    raw_records: list[dict],
    T_align: np.ndarray,
    path: str,
    rb_to_tcp_offset: np.ndarray | None = None,
    tcp_orientation_type: str = 'zyx_euler_deg',
):
    if not raw_records:
        return

    errors, inlier_errors, final_records = [], [], []
    for row in raw_records:
        rb_raw = np.array([row['rb_raw_x_m'], row['rb_raw_y_m'], row['rb_raw_z_m']])
        T_tcp = tcp_raw_to_matrix(np.array([
            row['tcp_x_mm'], row['tcp_y_mm'], row['tcp_z_mm'],
            row['tcp_rx_deg'], row['tcp_ry_deg'], row['tcp_rz_deg'],
        ]), tcp_orientation_type)

        T_pred = None
        if rb_to_tcp_offset is not None:
            rb_rot = np.array([row['rb_qx'], row['rb_qy'], row['rb_qz'], row['rb_qw']])
            if np.asarray(rb_to_tcp_offset).shape == (4, 4):
                T_rb = rb_pose_to_matrix(rb_raw, rb_rot)
                T_motive_tcp = T_rb @ rb_to_tcp_offset
                rb_corrected = T_motive_tcp[:3, 3]
                T_pred = T_align @ T_motive_tcp
            else:
                rb_corrected = rb_raw + quat_xyzw_to_matrix(rb_rot) @ rb_to_tcp_offset
        else:
            rb_corrected = rb_raw

        rb_aligned = (T_align @ np.append(rb_corrected, 1.0))[:3]
        rot_error_deg = ''
        if T_pred is not None:
            rot_error_deg = round(float(np.linalg.norm(rotvec_from_matrix(T_tcp[:3,:3].T @ T_pred[:3,:3])) * 180/np.pi), 4)

        tcp_pos_m = np.array([row['tcp_x_mm'], row['tcp_y_mm'], row['tcp_z_mm']]) / 1000.0
        err = np.linalg.norm(tcp_pos_m - rb_aligned)
        errors.append(err)
        if int(row.get('calibration_inlier', 1)):
            inlier_errors.append(err)

        final_records.append({
            'elapsed_s': row['elapsed_s'],
            'pose_label': row.get('pose_label', ''),
            'joint_0_deg': row.get('joint_0_deg', ''), 'joint_1_deg': row.get('joint_1_deg', ''),
            'joint_2_deg': row.get('joint_2_deg', ''), 'joint_3_deg': row.get('joint_3_deg', ''),
            'joint_4_deg': row.get('joint_4_deg', ''), 'joint_5_deg': row.get('joint_5_deg', ''),
            'calibration_inlier': row.get('calibration_inlier', ''),
            'tcp_x_m': round(tcp_pos_m[0], 6), 'tcp_y_m': round(tcp_pos_m[1], 6), 'tcp_z_m': round(tcp_pos_m[2], 6),
            'tcp_rx_deg': row.get('tcp_rx_deg', ''), 'tcp_ry_deg': row.get('tcp_ry_deg', ''), 'tcp_rz_deg': row.get('tcp_rz_deg', ''),
            'rb_raw_x_m': row['rb_raw_x_m'], 'rb_raw_y_m': row['rb_raw_y_m'], 'rb_raw_z_m': row['rb_raw_z_m'],
            'rb_qx': row.get('rb_qx', ''), 'rb_qy': row.get('rb_qy', ''),
            'rb_qz': row.get('rb_qz', ''), 'rb_qw': row.get('rb_qw', ''),
            'mocap_valid_samples': row.get('mocap_valid_samples', ''),
            'tcp_step_mm': row.get('tcp_step_mm', ''), 'rb_step_mm': row.get('rb_step_mm', ''),
            'step_delta_mm': row.get('step_delta_mm', ''), 'step_ratio': row.get('step_ratio', ''),
            'rb_corrected_x_m': round(rb_corrected[0], 6), 'rb_corrected_y_m': round(rb_corrected[1], 6), 'rb_corrected_z_m': round(rb_corrected[2], 6),
            'rb_aligned_x_m': round(rb_aligned[0], 6), 'rb_aligned_y_m': round(rb_aligned[1], 6), 'rb_aligned_z_m': round(rb_aligned[2], 6),
            'error_mm': round(err * 1000.0, 4),
            'rotation_error_deg': rot_error_deg,
        })

    rmse = np.sqrt(np.mean(np.array(errors)**2)) * 1000.0
    log.info("=" * 40)
    if inlier_errors and len(inlier_errors) != len(errors):
        log.info("캘리브레이션 RMSE: inlier %.3f mm / all %.3f mm",
                 np.sqrt(np.mean(np.array(inlier_errors)**2)) * 1000.0, rmse)
    else:
        log.info("캘리브레이션 RMSE: %.3f mm", rmse)
    log.info("=" * 40)

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=final_records[0].keys())
        writer.writeheader()
        writer.writerows(final_records)
    log.info("결과 CSV 저장 완료 → %s", path)


def _get_float(row: dict, key: str, default: float | None = None) -> float:
    v = row.get(key, '')
    if v == '' or v is None:
        if default is None:
            raise ValueError(f'CSV row에 {key} 값이 없습니다.')
        return default
    return float(v)


def load_calibration_samples_csv(path: str):
    raw_records, points_motive, quat_motive, points_tcp_m = [], [], [], []
    has_quat = True
    prev_tcp, prev_rb = None, None

    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    for idx, row in enumerate(rows, start=1):
        if 'tcp_x_m' in row:
            tcp_pos_m = np.array([_get_float(row, 'tcp_x_m'), _get_float(row, 'tcp_y_m'), _get_float(row, 'tcp_z_m')])
            tcp_raw = np.array([tcp_pos_m[0]*1000, tcp_pos_m[1]*1000, tcp_pos_m[2]*1000,
                                _get_float(row,'tcp_rx_deg',0), _get_float(row,'tcp_ry_deg',0), _get_float(row,'tcp_rz_deg',0)])
        else:
            tcp_raw = np.array([_get_float(row,'tcp_x_mm'), _get_float(row,'tcp_y_mm'), _get_float(row,'tcp_z_mm'),
                                _get_float(row,'tcp_rx_deg',0), _get_float(row,'tcp_ry_deg',0), _get_float(row,'tcp_rz_deg',0)])
            tcp_pos_m = tcp_raw[:3] / 1000.0

        rb_pos = np.array([_get_float(row,'rb_raw_x_m'), _get_float(row,'rb_raw_y_m'), _get_float(row,'rb_raw_z_m')])

        if all(k in row and row[k] != '' for k in ('rb_qx','rb_qy','rb_qz','rb_qw')):
            rb_rot = np.array([_get_float(row,'rb_qx'), _get_float(row,'rb_qy'),
                               _get_float(row,'rb_qz'), _get_float(row,'rb_qw')])
        else:
            rb_rot = np.array([0.0, 0.0, 0.0, 1.0])
            has_quat = False

        label = row.get('pose_label') or f'csv_row_{idx}'
        if 'tcp_step_mm' in row and row.get('tcp_step_mm') not in ('', None):
            tcp_step = _get_float(row,'tcp_step_mm',0)
            rb_step  = _get_float(row,'rb_step_mm',0)
            step_delta = _get_float(row,'step_delta_mm', rb_step - tcp_step)
            step_ratio = _get_float(row,'step_ratio', rb_step/tcp_step if tcp_step > 1e-9 else np.nan)
        elif prev_tcp is None:
            tcp_step = rb_step = step_delta = 0.0
            step_ratio = np.nan
        else:
            tcp_step = float(np.linalg.norm(tcp_pos_m - prev_tcp) * 1000)
            rb_step  = float(np.linalg.norm(rb_pos - prev_rb) * 1000)
            step_delta = rb_step - tcp_step
            step_ratio = rb_step / tcp_step if tcp_step > 1e-9 else np.nan
        prev_tcp, prev_rb = tcp_pos_m.copy(), rb_pos.copy()

        raw_records.append({
            'elapsed_s': _get_float(row,'elapsed_s',0),
            'pose_label': label,
            'joint_0_deg': _get_float(row,'joint_0_deg',0), 'joint_1_deg': _get_float(row,'joint_1_deg',0),
            'joint_2_deg': _get_float(row,'joint_2_deg',0), 'joint_3_deg': _get_float(row,'joint_3_deg',0),
            'joint_4_deg': _get_float(row,'joint_4_deg',0), 'joint_5_deg': _get_float(row,'joint_5_deg',0),
            'tcp_x_mm': float(tcp_raw[0]), 'tcp_y_mm': float(tcp_raw[1]), 'tcp_z_mm': float(tcp_raw[2]),
            'tcp_rx_deg': float(tcp_raw[3]), 'tcp_ry_deg': float(tcp_raw[4]), 'tcp_rz_deg': float(tcp_raw[5]),
            'rb_raw_x_m': float(rb_pos[0]), 'rb_raw_y_m': float(rb_pos[1]), 'rb_raw_z_m': float(rb_pos[2]),
            'rb_qx': float(rb_rot[0]), 'rb_qy': float(rb_rot[1]),
            'rb_qz': float(rb_rot[2]), 'rb_qw': float(rb_rot[3]),
            'mocap_valid_samples': _get_float(row,'mocap_valid_samples',0),
            'tcp_step_mm': round(tcp_step,4), 'rb_step_mm': round(rb_step,4),
            'step_delta_mm': round(step_delta,4),
            'step_ratio': round(float(step_ratio),6) if not np.isnan(step_ratio) else '',
        })
        points_motive.append(rb_pos)
        quat_motive.append(rb_rot)
        points_tcp_m.append(tcp_pos_m)

    return raw_records, np.array(points_motive), np.array(quat_motive), np.array(points_tcp_m), has_quat
