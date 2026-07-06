# plugin_pdm — Coordinate Convention

> 본 플러그인의 모든 자세는 아래 규약을 따른다. 외부와 데이터를 주고받기 전 단위·축·순서를 반드시 확인.

## 1. 자세 표현

| 표현 | 형식 | 예시 |
|------|------|------|
| RPY (사용자 입출력) | `np.ndarray (6,)` = `[x, y, z, roll, pitch, yaw]` | `[1.2, 0.5, 0.8, 0, -1.57, 0]` |
| 회전 변환 (내부) | `scipy.spatial.transform.Rotation.from_euler("xyz", rpy)` | intrinsic XYZ |
| 4×4 변환 행렬 (내부) | `T = [[R, t], [0, 1]]` | DDA mesh 변환 등 |

### 단위
- 길이: **미터 (m)**
- 각도: **라디안 (rad)** — 단, RPY 슬라이더(`controller.py`) 입력은 도(°). 슬라이더는 `*3.14/180` 으로 라디안 변환되어 ZMQ 로 발신.
- 시간 변수 없음 (정적 자세 산출)

### 회전 순서
- `scipy.Rotation.from_euler("xyz", rpy)` — 소문자 xyz = **intrinsic** (각 축이 회전하면서 다음 축이 같이 회전).
- URDF 의 `<origin rpy>` 도 같은 intrinsic XYZ 이므로 그대로 호환.

> ⚠️ 다른 라이브러리(eg. `pytransform3d`)는 default 가 extrinsic 인 경우가 있다. 새 라이브러리 도입 시 변환 검증 테스트를 추가할 것.

## 2. World Frame

배관과 로봇팔이 모두 표현되는 공통 좌표계. 보통 시뮬레이션·스캐너 좌표를 그대로 사용한다.

- `controller.cfg` / `viewer3d.cfg` 의 `urdf[].base = [x, y, z, roll, pitch, yaw]` (도, deg) 가 각 로봇의 base 위치/방향.
- 본 플러그인은 base 위치를 직접 다루지 않는다 — 검사 포인트와 자세 후보가 모두 world frame 에서 표현되며, 호출 측이 IK 까지 책임진다.

## 3. DDA TCP 좌표계

검사 포인트가 배관 표면 위에 있고, 배관 축 단위벡터를 `d̂` 라고 할 때:

```
   배관 길이 방향(축)            ↑ Y_DDA  (= d̂ 그대로)
                                 │
                                 │
                  X_DDA          ●─────► (배관 중심 방향)
                ◀───────         │
                                 │
                                 ▼  Z_DDA = X × Y  (오른손 좌표계)
```

| DDA TCP 축 | 의미 | 코드 정의 |
|-----------|------|-----------|
| X | DDA TCP 위치에서 **배관 중심 투영점** 을 향함 | `x_axis = (center − pos) / ‖.‖` |
| Y | **배관 길이 방향**과 평행 (`= pipe_direction`) | `y_axis = pipe_direction / ‖.‖` |
| Z | X × Y, 우수 좌표계 보장 | `z_axis = (x × y) / ‖.‖` |

DDA 자세가 N개 후보로 생성되는 것은 이 좌표계가 배관 축 주위로 회전하기 때문이다 — `θ = 2π·i/N`.

## 4. RT TCP 좌표계 (DDA 기준)

RT 는 DDA 가 정한 평면(DDA 의 X-Y 평면 = 배관 축에 수직 평면) 안에서 결정된다.

```
                  ↑ Y_RT
                  │      ┌──┐
                  │      │RT│
                  │      └─/┘ ← RT-X 가 DDA-X 와 ±α 만큼 벌어짐 (RT가 DDA를 향함)
                  │       /
   ─────●─────────┼─────●─────────  ← 배관 축 (Y_DDA = Y_RT)
   배관 │         │   /
        │         │  /
        │         │ /
        │       ┌─/┐
        │       │DDA│
        │       └──┘
                  ▼ Z_DDA
```

| RT TCP 축 | 코드 정의 |
|-----------|-----------|
| X | `−d` (RT → DDA 방향). `d` 는 DDA-X 축을 DDA-Z 축 주위로 `angle_of_rt` 회전한 단위벡터 |
| Z | DDA-Z 축과 동일 (= 같은 XY 평면 공유) |
| Y | `Z × X`, 행렬식 음수면 Z 부호 반전 후 재계산 |

### 왜 정확히 180° 가 아니라 ±5° 정도 기울일까?

배관과 RT/DDA 가 정확히 일직선이 되면 두 엔드이펙터가 충돌(혹은 직교 정렬되어 자세 자유도가 한 축으로 줄어드는 특이점) 이슈가 있다. 또 X-ray 빔이 검출기 한가운데로 수직 입사하면 검사 영역이 좁아지므로, 약간 기울여 검출 영역을 확장한다.

`angle_of_rt = 5°` (=`np.deg2rad(5)`) 권장. 0°(완전 마주봄)와 ±90°(직각) 같은 극단값은 입력 검증 단계에서 거부하지 말고 호출 측 책임으로 둔다.

## 5. 변환 행렬 사용 규약

URDF 에서 추출한 `tcp_to_link_pose_T` 는 **TCP frame 에서 link mesh frame 으로의 4×4 변환** 이다 (`__extract_tcp_and_end` 참고). 충돌 검사에서:

```
T_world_link = T_world_tcp · T_tcp_link
```

순서로 곱해야 한다 — `link_pose_T = tcp_pose_T @ tcp_to_link_pose_T`. 좌측 곱(우측 이동)이 정의이므로 행렬 순서를 뒤집지 말 것.

## 6. URDF 와의 정합

| URDF 요소 | 의미 |
|-----------|------|
| `<link name="dda_link_end">` | 엔드이펙터(=DDA) 메쉬가 붙는 마지막 링크 |
| `<joint name="dda_joint_tcp">` | `dda_link_end` ↔ TCP 프레임 사이의 고정 변환. `origin` 의 xyz/rpy 가 본 플러그인의 `tcp_to_link` 변환과 직결 |
| `<mesh scale="0.001 0.001 0.001">` | mm → m 보정. 변경 금지 |

이름 규약은 URDF 에 하드코딩되어 있다(`dda_link_end`, `dda_joint_tcp`, `rt_link_end`, `rt_joint_tcp`). URDF 를 새로 받으면 이 이름들을 유지하거나, `EndEffectorPoseOptimizer.load_DDA_from_urdf` / `load_RT_from_urdf` 호출 측에서 인자로 노출하도록 고쳐야 한다.

## 7. 흔한 실수 체크리스트

- [ ] PCD 단위가 mm 인데 `scale=1.0` 으로 로드 → 모든 거리·반경이 1000배 어긋남
- [ ] RPY 를 도(°) 로 그대로 `from_euler("xyz", rpy)` 에 넣음 → 라디안 인자라 실제로는 57배 큰 회전
- [ ] `from_euler("XYZ", ...)` (대문자) — extrinsic 으로 처리되어 결과가 다름
- [ ] 4×4 변환을 좌·우 곱 뒤바꿔 적용
- [ ] 배관 축이 거의 X축에 평행해서 `__calculate_dda_pose_candidate` 의 `basis` 분기가 발동되지 않으면 v1, v2 가 NaN — `dot > 0.9` 임계 검토
