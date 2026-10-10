# Robot Core 책임 분리 구현 기록 (2026-08-07)

[ROBOT_CORE_DECOUPLING_PLAN.md](ROBOT_CORE_DECOUPLING_PLAN.md)에서 설계한 내용 중 Phase 1(group
분할/회전 판단 이관) + Phase 2(source→target 단일화)를 구현한 기록. 설계 근거/대안 검토는 그 문서를,
"지금 뭐가 어떻게 동작하는가"는 이 문서를 본다.

## 1. 무엇이 바뀌었나 (한 줄 요약)

robot_core는 이제 `(robot_name, start_q, target_pose, planner, options) -> q_path` 하나만 안다.
target group 분할/정렬/positioner 회전 판단/target 간 `start_q` 체이닝은 전부 SimTool로 옮겼다.

## 2. 제거한 중복 로직

| 항목 | 이전 | 이후 |
|---|---|---|
| target pose 회전 변환 | `path_planning_service.py::_transform_target_pose`가 `Visualizer._transform_target_pose`와 동일 로직을 재구현 | 그룹 배치 코드 자체를 삭제하며 함께 제거 |
| positioner 회전 필요 판단 | `visualizer.py::_inspection_group_is_reachable_now`와 `inspection_workflow.py::group_is_reachable`가 같은 계산을 각각 구현 | `visualizer.py`가 `group_is_reachable`에 위임 (1곳만 구현) |
| "회전 판단→retreat→재계획" 오케스트레이션 | `path_planning_service.py::plan()`(실제 계획)과 `visualizer.py::_handle_request_check_ef_pose_ik`(IK 미리보기)가 같은 흐름을 각각 손으로 구현 | robot_core 쪽 구현을 통째로 삭제 — IK 미리보기 쪽만 남아 더 이상 중복 아님 |
| 로봇 이름 매핑(`_ef_pose_robot_name`), pose 목록 펼치기(`_inspection_group_pose_items`) | `visualizer.py`에 하드코딩 | `inspection_workflow.py::ef_pose_robot_name` / `inspection_group_pose_items`로 이동, `visualizer.py`는 위임만 |

## 3. robot_core 인터페이스

**삭제**: `OPERATION_PLAN_INSPECTION_PATH`, `InspectionPathPlanningService`(group batch: 분할/정렬/
회전/재시도 정책/병렬 실행 전부 포함하던 클래스).

**신규**: `OPERATION_PLAN_SINGLE_TARGET` (`robot_core/service.py`가 기본 operation).

```python
# robot_core/path_planning_service.py
def plan_single_target(engine, request):
    """request: robot_name, start_q, target_pose, planner, step_size, max_iter,
    fixed_joints/*, planning_timeout, ik_solver, ik_normalize, optimizer, optimize_path.
    반환: {"result": {status, message, ik_failure, elapsed, robot}, "q_path": [...]}
    """
```

`RobotCoreEngine._plan_inspection_path_for_robot()`(기존 단일-target 계획 primitive, 이미 있던
것)를 그대로 재사용한다. `worker.py::execute_robot_core_request`는 `OPERATION_POSE_DETERMINE`(EF
pose 결정, 별개 기능, 안 건드림)과 `OPERATION_PLAN_SINGLE_TARGET` 둘만 분기한다.

## 4. viewervedo(Visualizer) 변경

- 삭제: `_handle_request_plan_inspection_path`, `_inspection_robot_core_snapshot`(그룹 배치용 요청 처리)
- 신규: `_handle_request_plan_single_target` — 로봇 하나의 source_q(생략 시 현재 라이브 pose로 자동
  resolve) → target_pose 요청을 조립해 robot_core로 제출
- 신규: `_robot_core_scene_snapshot()` — target_groups/회전 정보 없이 mesh + 전체 로봇 joint state만
  담는 경량 snapshot (robot_core는 이제 회전을 모르므로 불필요)
- 유지: `_inspection_robot_core_snapshot()` — "Save Planning Snapshot" 기능(벤치마크용, target_groups
  포함)에서만 계속 사용
- `_handle_robot_core_completed` → `_handle_plan_single_target_completed`로 교체: 결과 q를 로봇에
  즉시 반영 + `_send_robot_joint_state_update()`로 joint state push + `reply_plan_single_target`으로 회신
- ZAPI: `zapi_plan_single_target` / `reply_plan_single_target` 신규, `zapi_plan_inspection_path` 삭제

## 5. simtool 변경

**신규 `simtool/inspection_sequencer.py` — `InspectionSequencer`**

- `partition_and_sort_target_groups`/`group_is_reachable`(viewer와 공유하는 순수 함수)로 그룹을
  reachable(회전 불필요) / deferred(회전 필요) phase로 분할
- reachable phase의 target들을 순서대로 `plan_single_target` 요청으로 하나씩 제출, 이전 응답의
  `q_path[-1]`을 다음 target의 `start_q`로 체이닝(로봇별로 독립적으로 추적)
- `request_id`로 매 요청/응답을 상관관계 매칭 (`on_reply()`가 stale/다른 요청의 응답은 무시)
- 첫 target 실패 시 즉시 시퀀스 중단(fail-fast, 기존 `skip_target`/`stop_all` 정책은 미이관)

**`InspectionPathHandler`(`simtool/inspection_path_handler.py`)**: `build_request`/`request_plan`
(옛 그룹 배치 요청 전송) 삭제. `accept_result`(IK 미리보기 결과 처리)/`request_playback`(재생 요청)만
남음.

**`window.py`**: `__on_btn_plan_inspection_path_clicked`가 `InspectionSequencer.start()` 호출로
교체. 메시지 dispatch에 `reply_plan_single_target` 토픽 추가.

## 6. joint state 전달 방식

**새 pub/sub 채널을 만들지 않았다.** 기존에 이미 있던 `_send_robot_joint_state_update()` /
`zapi.update_robot_joint_state()`(요청·이동 이벤트마다 특정 identity로 push하던 메커니즘, playback 등
여러 곳에서 이미 사용 중이었음)를 `plan_single_target` 완료 시점에도 트리거하도록 확장했다. 원래
설계 문서(PLAN.md 2장)의 "채널 A(PUB/SUB)" 아이디어는 이 기존 메커니즘 재사용으로 대체됨 — 인프라
중복을 피하기 위한 판단.

## 7. 벤치마크 스크립트 (`python/benchmark_path_planners.py`)

`InspectionPathPlanningService` 의존을 제거하고 `plan_single_target`을 target마다 직접 호출하도록
재작성. GUI/ZAPI 왕복이 없는 순수 in-process 스크립트라, **SimTool sequencer가 v1에서 defer한
positioner 회전 phase까지 포함해 전체 시나리오를 커버**한다(`spool_fix_r` 정책 차단 감지 포함, 이전
턴에서 만든 `policy_blocked` 리포팅 로직 유지).

## 8. 명시적으로 미이관한 것 (v1 범위 밖)

- **positioner 회전이 필요한 group의 자동 실행** — `InspectionSequencer`는 이런 그룹을
  `deferred_groups`로 보고만 하고 계획하지 않는다. 회전 명령 전송 → 실제 회전 완료 대기(ack) →
  retreat-to-safe-pose 계획 → 회전된 mesh로 재계획, 이 비동기 choreography는 아직 없음.
- **playback 재통합** — `_apply_inspection_planner_output`(경로 시각화/재생 준비 로직)이 지금은
  호출부가 없다(삭제하지 않고 남겨둠 — 회전/playback 재통합 시 다시 필요할 코드). "Start Simulation"
  버튼은 새 sequencer로 계획한 결과에 대해서는 `plan_sequence`가 채워지지 않아 "planned path not
  available"로 실패한다(에러가 명확하게 나는 것이지 조용히 깨지는 게 아님).
- 로봇 간 병렬 실행(`plan_batch`의 robot-parallel 실행) — sequencer는 target을 완전히 순차 처리한다.

## 9. 테스트 현황

- `simtool/tests/`(신규 `test_inspection_sequencer.py` 3건 포함), `plugins/robotics/tests/`,
  `plugins/pathplanner/tests/` — 총 23개 통과 확인.
- `robot_core/tests/`는 이 개발 환경에 `colorlog`가 설치돼 있지 않아 기존부터 실행 불가(이 작업과
  무관한 환경 문제).
- **WSL에서 실제 GUI로 "Plan Inspection Path" 버튼 end-to-end 검증은 아직 안 됨** — 다음 단계로 필요.

## 10. 다음에 할 만한 것

1. WSL + 실제 GUI로 reachable-only 시나리오 먼저 확인 (회전 없는 target group)
2. 확인되면 8장의 회전 choreography를 `InspectionSequencer`에 추가(Phase 3 관점: `move_positioner`
   전송 → `zapi_move_positioner`쪽에 완료 ack 추가 또는 joint-state push로 도착 확인 → retreat 계획 →
   회전된 target_pose로 계획)
3. `InspectionSequencer` 결과를 `plan_sequence` 호환 포맷으로 변환해 playback 버튼 복구
