# Robot Core 책임 분리 설계 (source→target 단일화 + state 채널)

## 0. 구현 현황 (2026-08-07)

Phase 1(group 분할/회전 판단 이관) + Phase 2(source→target 단일화)의 핵심 골격이 구현됨:

- `robot_core`: `OPERATION_PLAN_SINGLE_TARGET` 하나만 지원. group/batch/positioner 지식 없음
  (`robot_core/path_planning_service.py::plan_single_target`).
- `simtool`: `InspectionSequencer`(`simtool/inspection_sequencer.py`)가 `partition_and_sort_target_groups`/
  `group_is_reachable`(둘 다 `plugins/robotics/inspection_workflow.py`, viewer와 공유)로 그룹을 나누고,
  target마다 `plan_single_target`을 순차 제출하며 `start_q`를 체이닝.
- joint state push: 새 pub/sub 채널을 만들지 않고, 기존 `_send_robot_joint_state_update`/
  `update_robot_joint_state`(요청·이동 이벤트마다 identity로 push하던 기존 메커니즘)를
  `plan_single_target` 완료 시에도 그대로 재사용해서 확장함(2장의 "채널 A" 아이디어는 신규 구현 없이 기존
  push 메커니즘 재활용으로 대체).
- **v1 스코프 제한**: positioner 회전이 필요한(second phase) group은 `InspectionSequencer`가 자동
  실행하지 않고 "deferred"로만 보고한다. retreat-to-safe-pose, 회전 명령 전송 + 완료 대기(ack) + 회전된
  mesh로 재계획하는 choreography는 아직 이관되지 않음 — 8장 미해결 목록의 연장. 벤치마크
  스크립트(`benchmark_path_planners.py`)는 GUI/ZAPI 왕복이 없어 이 choreography를 in-process로
  전부 재현하므로, 회전 시나리오 자체는 거기서는 완전히 커버됨.

## 1. 배경 및 목표

현재 `plan_inspection_path` 한 번의 요청 안에서 robot_core(`InspectionPathPlanningService`)가
다음을 전부 처리한다.

- target group 분할(first/second, 접근 가능성 판단)
- 정렬(x 오름차순, z 내림차순)
- positioner 가상 회전(second group용 mesh/pose 변환)
- 로봇별 target 순차 계획 + 이전 target의 `final_q`를 다음 target의 `start_q`로 체이닝
- 실패 정책(stop_robot/skip_target), 병렬 실행(robot 단위)

이 설계는 다음을 목표로 한다.

- **robot_core**: `(robot_name, start_q, target_pose, scene) -> q_path` 단일 계획만 담당.
  group 개념, positioner 회전 판단, 순서 결정을 모른다.
- **simtool**: 어떤 순서로, 어떤 group을, positioner를 언제 돌려서 계획할지 결정하는
  오케스트레이션 주체가 된다.
- **viewervedo**: 3D scene(배관/positioner/로봇)의 유일한 소유자로 남는다. 상태 변경 실행
  (positioner 회전, 로봇 이동)과 그 상태의 "발행"을 담당한다.

핵심 문제는 이전 대화에서 정리한 대로 **URDF/배관의 "현재 상태"를 robot_core에 어떻게
알리느냐**다. 이 설계는 두 개의 서로 다른 채널로 분리해 해결한다.

---

## 2. 두 개의 채널

### 채널 A — State Broadcast (PUB/SUB, 가볍고 빈번함)

viewervedo가 **로봇 joint state / positioner 각도 / spool_fix_r** 같은 작은 상태를
바뀔 때마다(또는 짧은 주기로) 발행한다. 목적은 **모니터링과 simtool의 "지금 뭘 볼 수
있는지" 표시용**이며, 계획 결과의 정확성에 직접 관여하지 않는다.

```
topic: "scene_state"
payload (JSON, 작고 자주 바뀜):
{
    "scene_version": 42,
    "positioner_r_deg": 0.0,
    "spool_fix_r": false,
    "robot_joint_states": {
        "dda_rb10_1300e": {"joint1": 0.12, ...},
        "rb20_1900es": {...}
    }
}
```

- 전송: `common/zpipe.py`의 기존 `publish`/`subscribe` 패턴 그대로 사용(신규 인프라 불필요).
- viewervedo에 새 PUB 소켓 `ZAPI_SCENE_STATE` 추가, 렌더 루프 tick마다 또는 상태가
  실제로 바뀐 프레임에만 발행(mesh 같은 무거운 데이터는 여기 포함하지 않음).
- 구독자는 simtool(UI 표시용)과, 원한다면 robot_core(캐시 무효화 신호용) 둘 다 가능하지만
  **계획 요청 자체에는 이 채널의 데이터를 신뢰해서 쓰지 않는다** (아래 3번 참고 — race 방지).

### 채널 B — Planning Snapshot (REQ/REP, 무겁고 버전 고정)

배관/positioner collision mesh처럼 계획 결과에 직접 영향을 주는 데이터는 지금처럼
**요청 시점에 명시적으로 스냅샷을 찍어 버전(scene_version)을 고정**한다. 다만 지금처럼
매 target 요청마다 mesh를 통째로 실어 나르지 않고, **scene이 실제로 바뀔 때만 새 스냅샷을
만들고 재사용**한다.

```
요청: "get_planning_snapshot" (simtool -> viewervedo, REQ/REP)
응답:
{
    "scene_version": 42,
    "spool_vertices": [...], "spool_triangles": [...],
    "positioner_vertices": [...], "positioner_triangles": [...],
    "second_group_rotation_T": null,   # 이번 채널에서는 옵션 A: 안 씀 (아래 4장 참고)
}
```

- simtool은 시퀀스를 시작하기 전, 그리고 positioner를 회전시킬 때마다 이 요청으로 새
  스냅샷을 받아 `scene_version`과 함께 로컬에 보관한다.
- robot_core로 보내는 모든 개별 `plan(source, target)` 요청에는 **mesh 원본 대신
  `scene_version`만** 실어 보낸다. robot_core는 자신이 마지막으로 받은 scene과
  `scene_version`이 같으면 재사용하고, 다르면 simtool에게 그 버전의 스냅샷을 요청해
  받아온다(또는 최초 1회는 simtool이 스냅샷을 robot_core에도 직접 push).
- **로컬(embedded) 모드**에서는 애초에 같은 프로세스 트리 안이라 캐시 문제가 거의 없고,
  **external(standalone) 모드**에서만 이 캐시가 실제로 이득이 된다(네트워크로 mesh를 반복
  전송하지 않아도 됨).

> 실무적으로는 "완전한 무상태"를 포기하지 않는 절충도 가능하다 — scene_version 캐시를
> 아예 안 하고 지금처럼 매 요청에 mesh를 동봉하되, **group 분할/회전 판단만** simtool로
> 옮기는 것으로 1차 범위를 좁힐 수 있다. 이 문서의 5장 "단계적 적용"에서 이 옵션을
> Phase 1으로 명시한다.

---

## 3. Race 방지 규칙

1. simtool이 positioner 회전을 명령하면(`zapi_set_positioner_axis` 류 기존 커맨드),
   viewervedo는 회전을 **완료한 뒤** REQ/REP로 ack를 반드시 돌려준다(지금 이미 요청/응답
   구조이므로 변경 없음).
2. ack를 받기 전까지 simtool은 그 회전을 전제로 한 계획 요청을 보내지 않는다.
3. `get_planning_snapshot`은 ack 이후에만 호출한다 → 스냅샷의 `scene_version`이 항상
   "요청한 회전이 반영된 상태"임을 보장.
4. robot_core는 자신이 들고 있는 scene의 `scene_version`과 요청에 실린 `scene_version`이
   다르면 **절대 그냥 진행하지 않고** 에러로 반려하거나 최신 스냅샷을 다시 요청한다 —
   조용히 stale 데이터로 계획하는 경우를 원천 차단.
5. 채널 A(state broadcast)의 값은 계획 요청의 입력으로 쓰지 않는다 — 오직 표시/로깅용.

---

## 4. robot_core의 새 인터페이스

### 지금 (group 단위, robot_core가 순서/회전까지 처리)

```python
request = {
    "planner": "rrt_connect",
    "target_groups": [...],              # 전체 group 목록
    "positioner_second_group_r_deg": 180.0,
    ...
}
InspectionPathPlanningService(engine).plan(request)
```

### 목표 (source→target 단일 계획, simtool이 체이닝)

```python
request = {
    "operation": "plan_single_target",   # 새 operation (robot_core/service.py에 상수 추가)
    "scene_version": 42,
    "robot_name": "rb20_1900es",
    "start_q": [...],                    # 이전 target의 결과, 또는 현재 로봇 상태
    "target_pose": [[...4x4...]],
    "planner": "rrt_connect",
    "planner_options": {"step_size": 0.08, "max_iter": 3000},
    "planning_timeout": 30.0,
}
```

응답:

```python
{
    "status": "success" | "failed",
    "q_path": [...],
    "goal_q": [...],
    "elapsed": 1.23,
    "ik_failure": null,
    "verification": {...},
}
```

`plugins/pluginbase/plannerbase.py`의 `plan_target_sequence`/`plan_batch`는 그대로 두되
(simtool이 여러 target을 병렬/순차로 돌리고 싶다면 재사용 가능한 순수 함수라 simtool
쪽에서 import해서 써도 무방 — group 판단 로직만 옮기는 것이지 배치 실행 유틸리티 자체를
버릴 필요는 없다), `InspectionPathPlanningService.plan()`이 하던 **group 분할/positioner
회전 판단/second phase 처리**는 제거하고 `_plan_target()` 한 겹만 남긴다.

### robot_core에 새로 필요해지는 것

- `RobotCoreEngine`이 scene을 **snapshot마다 새로 만들지 않고 `scene_version`으로 캐시**
  하도록 변경 (`robot_core/worker.py`). 지금은 요청마다 `RobotCoreEngine(config, snapshot)`을
  새로 만드는데, 이걸 프로세스 수명 동안 유지되는 캐시로 바꿔야 함 — **stateless →
  stateful 전환이 이 리팩터의 실질적인 비용**이다(이전 답변에서 언급한 지점).

---

## 5. simtool로 옮겨오는 것

- `plugins/pluginbase/plannerbase.py`의 `partition_and_sort_target_groups`
  (`plugins/robotics/inspection_workflow.py`) — 순수 함수라 simtool 프로세스에서 그대로
  import해서 씀. 3D 씬 접근이 필요 없는 로직이라 이동 자체는 어렵지 않음.
- `_inspection_group_is_reachable_now`(현재 visualizer.py) — RT pose의 back-axis
  world-x 부호만 보는 순수 기하 계산. `target_groups`의 pose 데이터만 있으면 되므로
  simtool로 이동 가능.
- 회전 필요 여부 판단 → positioner 회전 명령 전송(`_ZAPI_request_move_positioner` 계열,
  이미 존재) → ack 대기 → `get_planning_snapshot` 요청 → group 내 target들을
  robot_core에 `plan_single_target`으로 순차/병렬 제출 → 결과의 `q_path[-1]`을 다음
  target의 `start_q`로 체이닝(지금 `plan_target_sequence`가 하던 일).

## 6. viewervedo에 남는 것 / 새로 필요한 것

- 3D 씬 소유권 및 회전/이동 실행 — 변경 없음.
- 신규: `get_planning_snapshot` ZAPI 핸들러 (지금의 `_inspection_robot_core_snapshot()`을
  거의 그대로 재사용, `scene_version` 필드만 추가).
- 신규: `scene_version` 카운터 — 배관 로드/이동/positioner 회전/로봇 강제 이동 등 collision
  scene에 영향 주는 모든 이벤트에서 증가.
- 신규(선택): 채널 A PUB 소켓 — 없어도 동작은 하지만 UI 실시간 표시엔 필요.

---

## 7. 단계적 적용 (한 번에 다 바꾸지 않기)

**Phase 1 — group/회전 판단만 이관, snapshot은 지금 방식 유지**
robot_core는 여전히 무상태, 매 요청마다 mesh 동봉. simtool이 `partition_and_sort_target_groups`
+ 회전 판단만 가져가서 "그룹 1개(회전 없는 first)" 또는 "그룹 1개(회전 적용된 second)"
단위로 지금과 같은 `plan_inspection_path` 요청을 그룹별로 여러 번 보낸다. robot_core
인터페이스는 거의 안 바뀜(요청을 여러 번 나눠 보낼 뿐). **리스크가 가장 작은 시작점.**

**Phase 2 — source→target 단일 계획으로 세분화**
robot_core 요청을 `plan_single_target`으로 좁히고, 그룹 내 target 체이닝을 simtool로
이관. `plan_target_sequence`/`plan_batch`를 simtool에서 재사용할지, robot_core를 여러 번
호출하는 얇은 루프를 simtool에 새로 짤지 결정 필요.

**Phase 3 — scene_version 캐시 도입**
robot_core를 stateful로 바꿔 mesh 재전송을 줄인다. external 모드에서만 체감 이득이
크므로, 이 단계는 필요성이 실측(벤치마크 스크립트로 mesh 전송 비용 vs 계획 시간 비교)된
뒤에 진행해도 늦지 않음.

**Phase 4 — state broadcast(채널 A) 추가**
UI 실시간 표시가 필요해지는 시점에 추가. 계획 로직에는 영향 없음.

---

## 8. 미해결 / 추가 검토 필요

- `plan_batch`의 robot 간 병렬 실행을 simtool이 가져가면, simtool은 robot_core에 여러
  robot의 target을 **동시에** 여러 요청으로 보내게 됨 — robot_core(특히 embedded 모드,
  단일 워커 스레드로 순차 처리 중, `robot_core/zapi.py`의 `_worker_loop`)가 병렬 요청을
  받아낼 수 있어야 함. 지금은 요청을 큐에 넣고 하나씩 처리하므로, robot 간 병렬성을
  유지하려면 robot_core도 워커 스레드를 여러 개로 늘리거나 robot_core 프로세스를
  robot별로 분리하는 걸 고려해야 함.
- retreat(안전 자세 후퇴) 경로 계획(`_plan_retreat_path_for_robot`)도 지금은
  `InspectionPathPlanningService.plan()` 안에서 group 시퀀스의 일부로 처리됨 — Phase 2에서
  이것도 simtool이 "target"의 한 종류로 다루도록 일반화할지 별도 operation으로 둘지 결정
  필요.
- `scene_version` staleness 판정 시 robot_core가 반려하면 simtool이 자동 재시도할지,
  사용자에게 알릴지 정책 필요.
