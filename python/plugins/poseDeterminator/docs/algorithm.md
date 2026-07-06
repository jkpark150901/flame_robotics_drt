# plugin_pdm — Algorithm

> 자세 계산의 기하학적 근거와 코드 단계별 설명. 수정 전 반드시 읽을 것.

## A. 전체 파이프라인

```
PCD + g_point ──► [A1] 배관 프로파일 추정 (방향, 중심, 반경)
                  │
                  ▼
            [A2] DDA 위치/방향 후보 N개 (배관 둘레 등간격)
                  │
                  ▼
            [A3] DDA 충돌 필터 (메쉬 ↔ 점군)
                  │
                  ▼  (각 후보마다 0°와 90° 두 자세 모두 검사)
            [A4] DDA 90° 회전 자세
                  │
                  ▼
            [A5] RT1(+α) / RT2(-α) 자세 계산 + 충돌 검사
                  │
                  ▼
                결과 그룹 (0/90 모두 유효 + RT 1개 이상 유효)
```

---

## A1. 배관 프로파일 추정 — `calculate_pipe_profile`

**입력**: `target_point` (배관 표면 위 1점), `_scan_data` (점군+법선)
**출력**: `__pipe_direction (3,)`, `__pipe_center (3,)`, `__pipe_radius (float)` (멤버변수)

### 단계
1. **소영역 박스 ROI**: `target_point ± sampling_size_for_calculating_normal` 의 AABB로 점들 추출 → `selected_points`. 비어 있으면 RuntimeError.
2. **법선 중앙값**: 추출된 점들의 법선들 component-wise median → `normal_m`. (평균 대신 중앙값을 쓰는 것은 스캔 노이즈/엣지 케이스에 강건하기 때문)
3. **반대편 표면점 추정**:
   - `target_point` 에서 `-normal_m` 방향으로 반지름 0.005m·길이 (-0.1~0.3) m 의 가늘고 긴 실린더 ROI
   - 추출된 점들을 `-normal_m` 축으로 투영, 군집화(`__cluster_points_along_line`, 군집 거리 0.005m)
   - **두 번째 군집**(=배관 반대 표면)의 가장 먼 점을 `estimated_opposite_point` 로 선택
   - `estimated_center = (target_point + estimated_opposite_point) / 2`
   - `estimated_radius = ‖opposite − center‖`
4. **본 피팅**: 추정 중심에서 `estimated_radius + offset` (기본 0.003m) 안의 점들을 `__extract_points_in_sphere` 로 추출 → `CylinderFitting.fit_cylinder` (Eberly 의 Powell 최적화)
5. **결과 저장**: `direction`, `center`, `radius` 를 멤버변수에 기록

### 튜닝 포인트
- `sampling_size_for_calculating_normal`(기본 0.01m): 법선이 안정적으로 잡힐 만큼 크되, 곡률이 평균화되어 사라질 만큼 크지 않게.
- 가늘고 긴 실린더 ROI 의 반지름 0.005m / 거리 (-0.1, 0.3)m: 배관 직경/브랜치 거리에 따라 코드 안 상수를 직접 조정해야 한다.
- 군집 거리 0.005m: 점군 밀도(점간 거리) 에 비례. 너무 작으면 한 표면이 여러 군집으로 쪼개짐.

---

## A2. DDA 위치/방향 후보 — `__calculate_dda_pose_candidate`

**입력**: 표면점, `radius = pipe_radius + distance_from_dda_to_surface`, `num_candidates`

### 단계
1. **동적 회전 중심**: `target_point` 를 배관 축에 투영 → 그 평면에서만 회전. (전역 `pipe_center` 를 그대로 쓰면 검사 포인트가 배관 축의 한 끝에 가까울 때 중심이 멀어짐.)
2. **배관 축에 수직인 두 단위벡터** `v1`, `v2`:
   ```python
   basis = [1,0,0] (또는 [0,1,0] if pipe_direction과 거의 평행)
   v1 = pipe_direction × basis,   v2 = pipe_direction × v1   (정규화)
   ```
3. **N등분 위치**: 각도 `θ_i = 2π·i/N` 에 대해
   ```
   position_i = center + radius · (cos θ_i · v1 + sin θ_i · v2)
   ```
4. **방향 행렬**:
   - Y축: `pipe_direction` 단위벡터 (배관 축과 평행)
   - X축: `(center − position)` 정규화 (배관 중심을 봄)
   - Z축: `X × Y` 정규화 (오른손 좌표계)
5. **rpy 변환**: `R.from_matrix([X, Y, Z]).as_euler("xyz")`

> 배관 축이 정확히 X축 또는 Y축에 평행하면 `basis` 선택 분기가 발동된다 (`abs(dot)>0.9`). 곡관·진단 모드에서 임계값을 만질 일이 있을 수 있음.

---

## A3. DDA 충돌 필터 — `__check_collision`

**입력**: `link_model`(o3d.TriangleMesh), `tcp_pose`(6,), `tcp_to_link_pose_T`(4×4)
**출력**: bool

### 단계
1. **메쉬 변환**: `tcp_pose` 를 4×4 행렬로 만들고 `tcp_to_link_pose_T` 로 후처리해 mesh 를 world frame 으로 옮긴 사본 생성.
2. **AABB 마진 ROI**: 변환된 메쉬의 AABB 에 `margin=0.05m` 더해 PCD 에서 점 추출 (성능 최적화). 비어 있으면 `False` 즉시 반환.
3. **메쉬 표면 균일 샘플**: `sample_count=5000` 점.
4. **거리 비교**: `compute_point_cloud_distance(sub_pcd, mesh_pcd)` 로 PCD ↔ 메쉬 표면 점들의 최소 거리. 어느 하나라도 `threshold = 0.001m` (1mm) 이하면 충돌로 판정.

### 주의
- AABB 마진(0.05m) 이 작으면 회전한 메쉬가 박스 밖으로 약간 튀어나갔을 때 **충돌을 놓친다**. 메쉬 크기에 비해 충분히 잡혀 있는지 점검.
- `sample_count` 는 정확도/성능 트레이드오프. 점군 밀도가 낮은 영역에서 메쉬 샘플링이 sparse 하면 거리 비교가 너무 관대해진다.
- threshold(1mm) 변경 시 PR 본문에 사유 기록.

---

## A4. DDA 90° 회전 — `__rotate_dda_pose_around_pipe_axis`

배관 축(world frame) 주위로 위치와 회전 모두 같은 각도 회전.

```
k        = pipe_direction / ‖.‖              (회전축 단위벡터)
proj_len = (dda_pos − pipe_center) · k        (DDA 위치를 축에 투영한 거리)
rot_ctr  = pipe_center + proj_len · k
v        = dda_pos − rot_ctr
v'       = v cos θ + (k × v) sin θ + k (k·v)(1 − cos θ)   ← Rodrigues
new_pos  = rot_ctr + v'

R_axis   = R.from_rotvec(k · θ)
R_dda'   = R_axis · R.from_euler("xyz", dda_rpy)
new_rpy  = R_dda'.as_euler("xyz")
```

> 회전 중심 `rot_ctr` 은 **검사 포인트의 배관-축 투영점** 이다. 모든 N개 후보가 동일 평면 안에서 같은 양 회전한다.

---

## A5. RT 자세 계산 — `__calculate_rt_pose_for_angle`

DDA TCP 좌표계 기준으로 RT 의 위치와 방향을 결정한다.

### 위치
1. DDA 회전행렬에서 X·Y·Z축 추출.
2. **DDA→RT 방향**: DDA-X축을 DDA-Z축 주위로 `angle_deg` 회전 (Rodrigues).
   ```
   d = X·cosα + (Ẑ × X)·sinα + Ẑ (Ẑ·X)(1−cosα)
   ```
3. RT 위치 = DDA 위치 + `d · distance_from_dda_to_rt`.

### 방향
- RT-X축: `−d` 정규화 (RT 가 DDA 를 봄).
- RT-Z축: `Ẑ` (DDA Z축과 같음 = 같은 XY 평면 공유).
- RT-Y축: `Z × X` 정규화.
- 행렬식 < 0 이면 좌수 좌표계 → `Z` 부호 반전 후 `Y = Z × X` 재계산.

### 의미
- `angle_deg` 가 0 이면 RT 가 DDA 의 X축 방향에 위치 (즉 DDA 표면과 같은 쪽). 음수/양수로 부호를 줘야 배관 반대편으로 간다 — 호출 측에서는 보통 ±5° 같은 값으로 두 후보(RT1, RT2)를 만든다.
- 약간 기울어진 마주봄(170° 등)을 만들기 위한 핵심이 이 단계의 **`angle_of_rt`** 이다. 정확히 180° (수학적으로는 `angle_deg = ±180`) 이면 두 RT 가 같은 위치에 와서 의미 없음.

---

## B. 그룹 채택 규칙 — `calculate_DDA_RT_pose_for_taking_xray`

```
for each base DDA candidate (N개):
    skip if DDA(0°) 충돌
    g0 = process(base, +α), process(base, −α)         # RT1·RT2 충돌 검사 후 dict
    rotated = rotate_around_pipe_axis(base, 90°)
    skip if DDA(90°) 충돌
    g90 = process(rotated, +α), process(rotated, −α)
    if "0" in g and "90" in g:
        groups.append(g)
    else (debug 모드면) collision_pose_groups.append(g)
```

채택은 "0° 와 90° 모두 유효" 라는 강한 조건이다. 한쪽만 통과한 그룹은 `debuging_info["collision_pose_groups"]` 에 들어가 노트북에서 빨간색 등으로 시각화 가능.

---

## C. 자주 쓰는 디버그 패턴

- **법선이 이상해 보일 때**: `selected_points` 의 색을 normal magnitude 로 매핑해 가시화. 작은 박스 ROI 안에 너무 적은 점이 잡히면 노이즈로 휘어진다.
- **반경이 음수/너무 큰 값**: `points_in_cylinder` 의 군집 결과를 확인. 가늘고 긴 실린더가 옆 배관을 같이 잡았을 가능성 (반대편 군집이 잘못 잡힘).
- **충돌이 너무 많이/적게**: 메쉬 사본을 `mesh_copy.transform(link_pose_T)` 한 결과를 PyVista 에 그려서 실제로 정확한 위치에 가는지 눈으로 확인.
- **회전 행렬 행렬식 ≠ ±1**: rpy 변환 중간에 직교성이 깨졌다는 신호 → 단위벡터 정규화 누락 검토.
