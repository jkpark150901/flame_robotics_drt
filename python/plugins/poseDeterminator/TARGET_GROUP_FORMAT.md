# Inspection Target Group 포맷

`EndEffectorPoseOptimizer.calculate_DDA_RT_pose_for_taking_xray(...)`가 반환하는
**target group** 리스트의 구조를 정의한다. 이 포맷은 viewer 시각화와 경로 계획에서
공통으로 쓰이므로, 여기 정의된 필드 외의 것을 추가할 때는 반드시 이 문서를 갱신한다.

## 설계 원칙

optimizer(`EndEffectorPoseOptimizer`)는 **검사 기준 위치와 검사 자세 쌍만** 계산해서 반환한다.
positioner 회전 필요 여부, 접근 가능성(first/second) 같은 판단은 optimizer가 하지 않고
**base planner(viewer/path planner) 쪽에서 rt_pose를 보고 직접 판단한다.**
optimizer 쪽에 이런 판단 로직을 다시 추가하지 않는다.

## 반환 형태

```python
target_groups: list[dict]
```

한 group = "검사 자세 한 세트"이며 DDA endeffector pose와 RT endeffector pose 한 쌍을 담는다.

### group 필드 (최소 정보)

| 필드 | 타입 | 설명 |
|------|------|------|
| `name` | `str` | 표시용 이름. 예: `"Inspection pose 1"` |
| `index` | `int` | 0부터의 순번 |
| `target_point` | `list[float]` (길이 3) | 검사 기준 위치 `[x, y, z]` (world 좌표계) |
| `dda_pose` | `list` (4x4) | DDA endeffector target pose (world 좌표계 homogeneous transform) |
| `rt_pose` | `list` (4x4) | RT endeffector target pose (world 좌표계 homogeneous transform) |

> 각도/편차/arc/rt_name/positioner 관련 상세 정보는 여기 넣지 않는다. 필요하면
> `optimizer.debuging_info`(debug 모드)에 담거나 별도 채널로 전달한다.
> RT1/RT2(±틸트) 중 어느 쪽을 썼는지도 노출하지 않는다 — 둘 다 같은 로봇(rb20_1900es)이고,
> 소비자 입장에서는 `rt_pose` 하나만 있으면 충분하다.

### 로봇 이름 매핑

target group은 로봇 이름을 저장하지 않는다. 소비자(viewer)가 pose_name → robot 이름을 매핑한다:

| pose_name | robot_name |
|-----------|------------|
| `DDA` | `dda_rb10_1300e` |
| `RT` | `rb20_1900es` |

viewer에서는 `_ef_pose_robot_name(pose_name)`가 이 매핑을 담당하고,
`_inspection_group_pose_items(group_info)`가 group을 `(robot_name, pose_name, target_T)` 목록으로 펼친다.

## 예시

```python
[
    {
        "name": "Inspection pose 1",
        "index": 0,
        "target_point": [1.2, 0.3, 0.05],
        "dda_pose": [[...4x4...]],
        "rt_pose":  [[...4x4...]],
    },
    ...
]
```

## 소비 지점 (viewer)

이 포맷은 [`python/viewervedo/visualizer.py`](../../viewervedo/visualizer.py)에서만 소비된다.
구조를 바꿀 때 아래를 함께 확인한다. 모두 `_inspection_group_pose_items()`를 거치도록 되어 있다.

- `_show_ef_target_groups` — EF pose mesh/frame/connector 시각화. positioner 회전 필요 여부를
  색으로 표시한다(초록=회전 불필요, 주황=회전 필요), 판정은 아래 reachability 기준과 동일하다.
- `_inspection_group_pose_items` / `_inspection_group_is_reachable_now` / `_inspection_group_rt_position`
  — first/second 분류 및 정렬 (아래 참고)
- `_plan_inspection_group_sequence` — 로봇별 경로 계획 제출
- `_handle_request_plan_inspection_path` — goal pose 시각화 + 계획 시퀀스
- `_handle_request_check_ef_pose_ik` — IK 가능성 체크
- ZApi 응답 직렬화 — target group을 그대로 반환(추가 변환 없음, `dda_pose`/`rt_pose`가 이미 list)

### first/second 분류 (positioner 회전 필요 여부 판단)

이 판단은 optimizer가 아니라 **viewer(base planner)** 가 한다. `_inspection_group_is_reachable_now`:

1. RT의 pipe-facing 로컬 축(설정값, 기본 local -Y)의 반대(back-axis, "상위 링크와 연결되는 방향")를
   RT pose 회전으로 world 변환한다.
2. 그 world 벡터의 x 성분 부호를 본다.
   - `x < 0` → positioner 회전 없이 지금 접근 가능 (first)
   - `x >= 0` → positioner 회전 필요 (second)
3. DDA는 구조상 back-axis의 world x 성분이 항상 0(배관 원주를 도는 후보라 world X와 수직)이라
   판정에 쓰지 않는다. RT만 본다.

정렬 기준: RT target 위치의 x 오름차순, z 내림차순(x 우선).
first 계획 → 룰베이스 포지셔너 가상 회전 → second 계획 순으로 진행한다.
