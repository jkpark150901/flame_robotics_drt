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
| `index` | `int` | 0부터의 순번. optimizer가 매길 때는 검사 지점(target_point) 하나 안에서의 순번이다. |
| `target_point` | `list[float]` (길이 3) | 검사 기준 위치 `[x, y, z]` (world 좌표계) |
| `dda_pose` | `list` (4x4) | DDA endeffector target pose (world 좌표계 homogeneous transform) |
| `rt_pose` | `list` (4x4) | RT endeffector target pose (world 좌표계 homogeneous transform) |

> **여러 검사 지점을 합칠 때 주의**: optimizer는 지점(point)마다 독립적으로 `name`/`index`를
> 1부터 다시 매긴다("Inspection pose 1"이 지점마다 반복됨). 여러 지점의 target group을
> 하나로 합치는 소비자(viewer의 `_handle_request_determine_ef_pose`)는 이름 충돌을 피하기
> 위해 `name`에 지점 번호를 접두어로 붙이고(`"Point 2 - Inspection pose 1"`), `index`를
> 합친 목록 기준으로 다시 매기며, 원래 지점 번호를 `point_index`(0부터)에 별도로 저장한다.
> `point_index`는 optimizer 계약에는 없는, viewer가 추가하는 필드다.

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

## 소비 지점 (viewer / simtool)

target group을 (robot_name, pose_name, target_T)로 펼치는 것, 그리고 회전 필요 여부(first/
second 분류) 판단은 [`python/plugins/robotics/inspection_workflow.py`](../robotics/inspection_workflow.py)의
`inspection_group_pose_items()` / `group_is_reachable()` / `partition_and_sort_target_groups()`에
있다 — viewer와 simtool이 이 순수 함수를 공유해서 쓴다(로직 이중화 방지). 구조를 바꿀 때 아래를
함께 확인한다.

- `python/plugins/robotics/inspection_workflow.py` — first/second 분류/정렬(공유 함수), 로봇 이름 매핑
- `python/simtool/inspection_sequencer.py` — SimTool이 이 함수들로 나눈 group을 순서대로
  `plan_single_target` 요청으로 Robot Core에 제출 (ROBOT_CORE_DECOUPLING_PLAN.md)
- `python/viewervedo/visualizer.py`
  - `_show_ef_target_groups` — EF pose mesh/frame/connector 시각화. positioner 회전 필요 여부를
    색으로 표시한다(초록=회전 불필요, 주황=회전 필요), 판정은 위 공유 함수와 동일하다.
  - `_inspection_group_pose_items` / `_inspection_group_is_reachable_now` — 위 공유 함수에 위임하는 얇은 wrapper
  - `_handle_request_plan_single_target` — 로봇 하나의 source_q -> target_pose 계획을 Robot Core에 제출
  - `_handle_request_check_ef_pose_ik` — IK 가능성 체크
  - ZApi 응답 직렬화 — target group을 그대로 반환(추가 변환 없음, `dda_pose`/`rt_pose`가 이미 list)

### first/second 분류 (positioner 회전 필요 여부 판단)

이 판단은 optimizer가 아니라 **base planner(viewer/simtool이 공유하는 `group_is_reachable`)** 가 한다:

1. RT의 pipe-facing 로컬 축(설정값, 기본 local -Y)의 반대(back-axis, "상위 링크와 연결되는 방향")를
   RT pose 회전으로 world 변환한다.
2. 그 world 벡터의 x 성분 부호를 본다.
   - `x < 0` → positioner 회전 없이 지금 접근 가능 (first)
   - `x >= 0` → positioner 회전 필요 (second)
3. DDA는 구조상 back-axis의 world x 성분이 항상 0(배관 원주를 도는 후보라 world X와 수직)이라
   판정에 쓰지 않는다. RT만 본다.

정렬 기준: RT target 위치의 x 오름차순, z 내림차순(x 우선).
first 계획 → 룰베이스 포지셔너 가상 회전 → second 계획 순으로 진행한다.
