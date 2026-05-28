# OpenCV `cv2.calibrateHandEye` 기반 캘리브레이션 코드 수정 계획

## 0. 목표

현재 문제는 다음 두 변환을 추정하는 것이다.

\[
{}^{B}T_M
\]

- robot base frame \(B\) 기준에서 본 mocap world frame \(M\)

\[
{}^{RB}T_{TCP}
\]

- mocap rigid body frame \(RB\) 기준에서 본 robot TCP frame

전체 관측식은 다음과 같다.

\[
{}^{B}T_{TCP,i}
=
{}^{B}T_M
{}^{M}T_{RB,i}
{}^{RB}T_{TCP}
\]

여기서 각 pose \(i\)마다 이미 알고 있는 값은 다음 두 개이다.

\[
{}^{B}T_{TCP,i}
\]

- 로봇 컨트롤러/FK에서 읽은 TCP pose

\[
{}^{M}T_{RB,i}
\]

- mocap에서 읽은 rigid body pose

구해야 하는 값은 다음 두 개이다.

\[
{}^{RB}T_{TCP}
\]

\[
{}^{B}T_M
\]

---

## 1. OpenCV `calibrateHandEye`에 맞춘 좌표계 대응

OpenCV `calibrateHandEye`는 일반적으로 다음 구조를 사용한다.

\[
{}^{base}T_{gripper,i}
\,
{}^{gripper}T_{camera}
\,
{}^{camera}T_{target,i}
=
{}^{base}T_{target}
\]

OpenCV 입력/출력 이름으로는 다음과 같다.

```text
입력:
R_gripper2base, t_gripper2base  =  ^base T_gripper
R_target2cam,   t_target2cam    =  ^camera T_target

출력:
R_cam2gripper,  t_cam2gripper   =  ^gripper T_camera
```

우리 문제에 맞추기 위해 다음처럼 대응시킨다.

```text
OpenCV base     = mocap world M
OpenCV gripper  = mocap rigid body RB
OpenCV camera   = robot TCP
OpenCV target   = robot base B
```

따라서 OpenCV에 넣을 값은 다음과 같다.

\[
{}^{base}T_{gripper,i}
=
{}^{M}T_{RB,i}
\]

\[
{}^{camera}T_{target,i}
=
{}^{TCP}T_{B,i}
=
({}^{B}T_{TCP,i})^{-1}
\]

OpenCV가 반환하는 값은:

\[
{}^{gripper}T_{camera}
=
{}^{RB}T_{TCP}
\]

즉, 우리가 원하는 rigid body to TCP 변환이다.

---

## 2. OpenCV 입력으로 변환할 행렬

각 샘플 \(i\)에 대해 기존 코드에서 만드는 행렬은 다음과 같다.

```python
T_motive_rb_i = rb_pose_to_matrix(rb_pos, rb_quat)
T_base_tcp_i  = tcp_raw_to_matrix(tcp_raw)
```

OpenCV에 넣을 행렬은:

```python
T_gripper2base_i = T_motive_rb_i
T_target2cam_i   = invert_transform(T_base_tcp_i)
```

즉:

```python
R_gripper2base.append(T_motive_rb_i[:3, :3])
t_gripper2base.append(T_motive_rb_i[:3, 3])

T_tcp_base_i = invert_transform(T_base_tcp_i)

R_target2cam.append(T_tcp_base_i[:3, :3])
t_target2cam.append(T_tcp_base_i[:3, 3])
```

---

## 3. 새 함수 추가 계획

기존 `solve_handeye_world_tag_tcp()`를 직접 구현한 Park 방식 대신, OpenCV를 호출하는 새 함수를 추가한다.

```python
def solve_handeye_opencv(
    T_motive_rb_list: list[np.ndarray],
    T_base_tcp_list: list[np.ndarray],
    method: int = cv2.CALIB_HAND_EYE_PARK,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve:
        ^B T_TCP_i = ^B T_M · ^M T_RB_i · ^RB T_TCP

    Using OpenCV calibrateHandEye mapping:
        base    = M
        gripper = RB
        camera  = TCP
        target  = B

    OpenCV inputs:
        R_gripper2base = ^M R_RB
        t_gripper2base = ^M t_RB
        R_target2cam   = ^TCP R_B = inv(^B T_TCP).R
        t_target2cam   = ^TCP t_B = inv(^B T_TCP).t

    OpenCV output:
        R_cam2gripper  = ^RB R_TCP
        t_cam2gripper  = ^RB t_TCP
    """
```

---

## 4. 구현 코드 초안

```python
import cv2
import numpy as np


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def solve_handeye_opencv(
    T_motive_rb_list: list[np.ndarray],
    T_base_tcp_list: list[np.ndarray],
    method: int = cv2.CALIB_HAND_EYE_PARK,
) -> tuple[np.ndarray, np.ndarray]:

    if len(T_motive_rb_list) != len(T_base_tcp_list):
        raise ValueError("mocap pose list와 robot TCP pose list 길이가 다릅니다.")

    if len(T_motive_rb_list) < 4:
        raise ValueError("OpenCV hand-eye calibration에는 최소 4개 이상의 pose가 필요합니다.")

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for T_motive_rb, T_base_tcp in zip(T_motive_rb_list, T_base_tcp_list):
        # OpenCV base = mocap world M
        # OpenCV gripper = mocap rigid body RB
        # ^base T_gripper = ^M T_RB
        R_gripper2base.append(T_motive_rb[:3, :3].astype(np.float64))
        t_gripper2base.append(T_motive_rb[:3, 3].reshape(3, 1).astype(np.float64))

        # OpenCV camera = robot TCP
        # OpenCV target = robot base B
        # ^camera T_target = ^TCP T_B = inv(^B T_TCP)
        T_tcp_base = invert_transform(T_base_tcp)

        R_target2cam.append(T_tcp_base[:3, :3].astype(np.float64))
        t_target2cam.append(T_tcp_base[:3, 3].reshape(3, 1).astype(np.float64))

    R_rb_tcp, t_rb_tcp = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=method,
    )

    T_rb_tcp = make_transform(R_rb_tcp, t_rb_tcp)

    # ^B T_M 계산
    # ^B T_TCP = ^B T_M · ^M T_RB · ^RB T_TCP
    # ^B T_M = ^B T_TCP · inv(^M T_RB · ^RB T_TCP)
    T_base_motive_candidates = []

    for T_motive_rb, T_base_tcp in zip(T_motive_rb_list, T_base_tcp_list):
        T_base_motive_i = T_base_tcp @ invert_transform(T_motive_rb @ T_rb_tcp)
        T_base_motive_candidates.append(T_base_motive_i)

    T_base_motive = average_transforms(T_base_motive_candidates)

    return T_base_motive, T_rb_tcp
```

---

## 5. `average_transforms()` 추가 또는 기존 평균 함수 확장

현재 코드에는 `average_rotations()`만 있다.  
translation 평균과 rotation 평균을 같이 처리하는 함수가 필요하다.

```python
def average_transforms(T_list: list[np.ndarray]) -> np.ndarray:
    T_avg = np.eye(4)

    rotations = [T[:3, :3] for T in T_list]
    translations = [T[:3, 3] for T in T_list]

    T_avg[:3, :3] = average_rotations(rotations)
    T_avg[:3, 3] = np.mean(translations, axis=0)

    return T_avg
```

---

## 6. 기존 `calibrate_from_samples()` 수정 계획

현재 hand-eye 분기:

```python
if args.calibration_model == 'handeye':
    ...
    T_align, T_rigidbody_tcp = solve_handeye_world_tag_tcp(...)
```

이 부분을 다음처럼 교체한다.

```python
if args.calibration_model == 'handeye':
    T_motive_rb_list = [
        rb_pose_to_matrix(
            np.array([row['rb_raw_x_m'], row['rb_raw_y_m'], row['rb_raw_z_m']]),
            np.array([row['rb_qx'], row['rb_qy'], row['rb_qz'], row['rb_qw']]),
        )
        for row in raw_records
    ]

    T_base_tcp_list = [
        tcp_raw_to_matrix(
            np.array([
                row['tcp_x_mm'],
                row['tcp_y_mm'],
                row['tcp_z_mm'],
                row['tcp_rx_deg'],
                row['tcp_ry_deg'],
                row['tcp_rz_deg'],
            ]),
            orientation_type=args.tcp_orientation_type,
        )
        for row in raw_records
    ]

    T_align, T_rigidbody_tcp = solve_handeye_opencv(
        T_motive_rb_list,
        T_base_tcp_list,
        method=args.opencv_handeye_method,
    )

    residuals, rot_residuals = handeye_residuals(
        T_align,
        T_rigidbody_tcp,
        T_motive_rb_list,
        T_base_tcp_list,
    )
```

여기서:

```text
T_align          = ^B T_M
T_rigidbody_tcp  = ^RB T_TCP
```

이다.

---

## 7. CLI 인자 추가 계획

OpenCV method를 선택할 수 있게 한다.

```python
def parse_opencv_handeye_method(name: str) -> int:
    methods = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    key = name.lower()
    if key not in methods:
        raise ValueError(f"unknown hand-eye method: {name}")

    return methods[key]
```

argparse에는 문자열로 추가한다.

```python
p.add_argument(
    "--opencv_handeye_method",
    choices=["tsai", "park", "horaud", "andreff", "daniilidis"],
    default="park",
    help="OpenCV calibrateHandEye method",
)
```

`args` 처리 후 실제 OpenCV enum으로 변환한다.

```python
args.opencv_handeye_method = parse_opencv_handeye_method(args.opencv_handeye_method)
```

---

## 8. TCP orientation convention 옵션 추가

현재 코드에서는 TCP orientation을 다음처럼 해석한다.

```python
R = Rz(rz) @ Ry(ry) @ Rx(rx)
```

하지만 로봇 컨트롤러의 `tcp_ref[3:6]`가 실제로 어떤 convention인지 확실하지 않으면 hand-eye 결과가 크게 틀어진다.

따라서 CLI 옵션을 추가한다.

```python
p.add_argument(
    "--tcp_orientation_type",
    choices=["zyx_euler_deg", "xyz_euler_deg", "rotvec_deg", "rotvec_rad"],
    default="zyx_euler_deg",
)
```

그리고 `tcp_raw_to_matrix()`를 다음 형태로 바꾼다.

```python
def tcp_raw_to_matrix(
    tcp_raw: np.ndarray,
    orientation_type: str = "zyx_euler_deg",
) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(tcp_raw[:3], dtype=float) / 1000.0

    r = np.asarray(tcp_raw[3:6], dtype=float)

    if orientation_type == "zyx_euler_deg":
        rx, ry, rz = np.radians(r)
        R = _rot_z(rz) @ _rot_y(ry) @ _rot_x(rx)

    elif orientation_type == "xyz_euler_deg":
        rx, ry, rz = np.radians(r)
        R = _rot_x(rx) @ _rot_y(ry) @ _rot_z(rz)

    elif orientation_type == "rotvec_deg":
        R = matrix_from_rotvec(np.radians(r))

    elif orientation_type == "rotvec_rad":
        R = matrix_from_rotvec(r)

    else:
        raise ValueError(f"unknown orientation_type: {orientation_type}")

    T[:3, :3] = R
    return T
```

---

## 9. `matrix_from_rotvec()` 추가

```python
def matrix_from_rotvec(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    theta = np.linalg.norm(w)

    if theta < 1e-12:
        return np.eye(3)

    k = w / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])

    R = (
        np.eye(3)
        + np.sin(theta) * K
        + (1.0 - np.cos(theta)) * (K @ K)
    )

    return R
```

---

## 10. quaternion 평균 수정

현재는 quaternion을 단순 평균한다.  
부호 flip 문제 때문에 다음 함수로 교체한다.

```python
def average_quaternions_xyzw(quats: list[np.ndarray]) -> np.ndarray:
    if not quats:
        raise ValueError("quaternion list is empty")

    q0 = np.asarray(quats[0], dtype=float)
    q0 = q0 / np.linalg.norm(q0)

    aligned = []

    for q in quats:
        q = np.asarray(q, dtype=float)
        q = q / np.linalg.norm(q)

        if np.dot(q, q0) < 0:
            q = -q

        aligned.append(q)

    q_avg = np.mean(aligned, axis=0)
    q_avg = q_avg / np.linalg.norm(q_avg)

    return q_avg
```

`sample_stable_point()` 안의 기존 코드:

```python
rb_rot = np.mean(rb_rot_samples, axis=0)
rb_rot_norm = np.linalg.norm(rb_rot)
if rb_rot_norm > 0.0:
    rb_rot = rb_rot / rb_rot_norm
```

를 다음으로 교체한다.

```python
rb_rot = average_quaternions_xyzw(rb_rot_samples)
```

---

## 11. residual 계산은 기존 구조 유지

OpenCV로 `T_rb_tcp`를 구한 뒤에도 최종 검증은 기존 식으로 한다.

\[
{}^{B}T_{TCP,pred,i}
=
{}^{B}T_M
{}^{M}T_{RB,i}
{}^{RB}T_{TCP}
\]

```python
def handeye_residuals(
    T_base_motive: np.ndarray,
    T_rb_tcp: np.ndarray,
    T_motive_rb_list: list[np.ndarray],
    T_base_tcp_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:

    pos_residuals = []
    rot_residuals = []

    for T_motive_rb, T_base_tcp in zip(T_motive_rb_list, T_base_tcp_list):
        T_pred = T_base_motive @ T_motive_rb @ T_rb_tcp

        T_err = invert_transform(T_base_tcp) @ T_pred

        pos_residuals.append(np.linalg.norm(T_err[:3, 3]) * 1000.0)

        rot_err = rotvec_from_matrix(T_err[:3, :3])
        rot_residuals.append(np.linalg.norm(rot_err) * 180.0 / np.pi)

    return np.asarray(pos_residuals), np.asarray(rot_residuals)
```

기존처럼 `T_tcp.T @ T_pred` 형태로 회전만 따로 비교하는 것보다, SE(3) error transform을 직접 계산하는 방식이 더 명확하다.

---

## 12. outlier 제거 전략

OpenCV `calibrateHandEye` 자체에는 outlier 제거 기능이 없다.  
따라서 다음 순서로 처리한다.

1. 전체 pose로 1차 OpenCV hand-eye 수행
2. 각 pose별 residual 계산
3. residual이 threshold보다 큰 pose 제외
4. inlier pose만으로 OpenCV hand-eye 재수행
5. 최종 residual 저장

예시:

```python
T_align, T_rigidbody_tcp = solve_handeye_opencv(...)

residuals, rot_residuals = handeye_residuals(...)

inlier_mask = residuals <= args.outlier_threshold_mm

if np.count_nonzero(inlier_mask) >= 6 and np.any(~inlier_mask):
    T_align, T_rigidbody_tcp = solve_handeye_opencv(
        [T for T, ok in zip(T_motive_rb_list, inlier_mask) if ok],
        [T for T, ok in zip(T_base_tcp_list, inlier_mask) if ok],
        method=args.opencv_handeye_method,
    )

    residuals, rot_residuals = handeye_residuals(
        T_align,
        T_rigidbody_tcp,
        T_motive_rb_list,
        T_base_tcp_list,
    )
```

---

## 13. pose plan 수정 권장

OpenCV hand-eye도 결국 \(AX=XB\) 문제를 푸는 것이므로, 다양한 상대 회전이 필요하다.

현재 기본값이:

```python
--tool_roll_sweep_deg 0
```

이면 충분하지 않을 수 있다.

권장:

```bash
--tool_roll_sweep_deg -60 0 60
```

또는 기본값 자체를 수정한다.

```python
p.add_argument(
    "--tool_roll_sweep_deg",
    type=float,
    nargs="+",
    default=[-60.0, 0.0, 60.0],
)
```

추가로 `q5` 회전만으로 부족하면 wrist pitch/yaw가 다른 template을 추가한다.

---

## 14. 저장 포맷 수정

현재 `T_align`이라는 이름은 의미가 모호하다.  
hand-eye 모드에서는 다음 이름으로 저장하는 것이 좋다.

```python
def save_handeye_calibration(
    T_base_motive: np.ndarray,
    T_rb_tcp: np.ndarray,
    path: str,
):
    payload = {
        "T_base_motive": T_base_motive.tolist(),
        "T_rb_tcp": T_rb_tcp.tolist(),

        # backward compatibility
        "T_align": T_base_motive.tolist(),
        "T_rigidbody_tcp": T_rb_tcp.tolist(),

        "convention": {
            "T_base_tcp": "^B T_TCP",
            "T_motive_rb": "^M T_RB",
            "T_base_motive": "^B T_M",
            "T_rb_tcp": "^RB T_TCP",
            "model": "^B T_TCP = ^B T_M @ ^M T_RB @ ^RB T_TCP",
            "opencv_mapping": {
                "base": "mocap world M",
                "gripper": "mocap rigid body RB",
                "camera": "robot TCP",
                "target": "robot base B",
                "R_gripper2base": "^M R_RB",
                "R_target2cam": "^TCP R_B",
                "R_cam2gripper_output": "^RB R_TCP"
            }
        }
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
```

---

## 15. 최종 실행 예시

```bash
python precision_eval_svd.py   --calibration_model handeye   --opencv_handeye_method park   --tcp_orientation_type zyx_euler_deg   --tool_roll_sweep_deg -60 0 60   --outlier_threshold_mm 60
```

TCP orientation convention이 불확실하면 아래 옵션을 바꿔가며 residual을 비교한다.

```bash
--tcp_orientation_type zyx_euler_deg
--tcp_orientation_type xyz_euler_deg
--tcp_orientation_type rotvec_deg
--tcp_orientation_type rotvec_rad
```

---

## 16. 수정 우선순위

1. `import cv2` 추가
2. `solve_handeye_opencv()` 추가
3. `calibrate_from_samples()`의 hand-eye 분기를 OpenCV 호출로 교체
4. OpenCV 입력 mapping 적용  
   - `R_gripper2base = ^M R_RB`
   - `t_gripper2base = ^M t_RB`
   - `R_target2cam = ^TCP R_B`
   - `t_target2cam = ^TCP t_B`
5. OpenCV 출력 `R_cam2gripper`, `t_cam2gripper`를 `T_rb_tcp = ^RB T_TCP`로 저장
6. `T_base_motive = ^B T_M`를 각 pose에서 계산 후 평균
7. quaternion 평균 수정
8. TCP orientation convention 옵션 추가
9. outlier 제거 후 재보정
10. 저장 JSON key를 명확히 변경

---

## 17. 핵심 요약

OpenCV `calibrateHandEye`를 그대로 사용하려면 좌표계를 다음처럼 대응시키면 된다.

```text
OpenCV base     = mocap world
OpenCV gripper  = mocap rigid body
OpenCV camera   = robot TCP
OpenCV target   = robot base
```

따라서 입력은:

\[
R_{gripper2base}, t_{gripper2base}
=
{}^{M}R_{RB}, {}^{M}t_{RB}
\]

\[
R_{target2cam}, t_{target2cam}
=
{}^{TCP}R_B, {}^{TCP}t_B
=
({}^{B}T_{TCP})^{-1}
\]

출력은:

\[
R_{cam2gripper}, t_{cam2gripper}
=
{}^{RB}R_{TCP}, {}^{RB}t_{TCP}
\]

즉:

\[
T_{rb\_tcp} = {}^{RB}T_{TCP}
\]

이다.

그 다음:

\[
{}^{B}T_M
=
{}^{B}T_{TCP,i}
\left(
{}^{M}T_{RB,i}
{}^{RB}T_{TCP}
\right)^{-1}
\]

로 `T_base_motive`를 계산하면 된다.
