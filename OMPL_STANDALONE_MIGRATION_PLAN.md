# OMPL Standalone 전환 구현 계획

## 1. 목적

현재 Python으로 직접 구현된 `RRTStar`의 joint-space 탐색 로직을 OMPL standalone으로 교체한다.

이번 전환의 주요 목적은 다음과 같다.

- RRT, RRT-Connect, RRT\*, Informed RRT\*, BIT\* 등을 동일한 인터페이스에서 선택
- 기존 Pinocchio/Coal 기반 로봇 모델 및 충돌 검증 코드 재사용
- 기존 `PlannerBase.generate()` 호출부와 다중 로봇/다중 target 실행 구조 유지
- 향후 STOMP, TrajOpt, APF 등의 trajectory optimizer를 같은 평가 pipeline에 연결
- 기존 자체 RRT\*를 즉시 삭제하지 않고 A/B 검증 및 rollback backend로 유지
- 동일한 state metric, collision resolution, timeout 및 평가 지표로 planner를 비교

---

## 2. 현재 코드 기준 분석

### 2.1 유지할 기능

현재 `PlannerBase`에 이미 다음 공통 기능이 구현되어 있으므로 OMPL 전환 후에도 유지한다.

- `generate()`의 workspace/joint-space 분기
- `robotics_backend` 추상화
- Pinocchio/Coal 기반 단일 configuration 충돌 검사
- edge collision 검사
- joint limit 기반 정규화
- fixed joint 적용
- FK 기반 workspace 제한
- planning deadline
- target sequence 계획
- robot 단위 병렬 계획
- 경로 최종 검증
- collision pair 및 디버그 정보 저장
- 외부 소비자가 사용하는 결과 형식

특히 다음 메서드는 OMPL adapter에서도 그대로 사용한다.

```text
PlannerBase.generate()
PlannerBase._prepare_fixed_joint_constraints()
PlannerBase._workspace_position_ok()
PlannerBase.check_robot_collision()
PlannerBase._check_collision()
PlannerBase._edge_collision_info()
PlannerBase.verify_path()
PlannerBase.plan_target_sequence()
PlannerBase.plan_batch()
```

### 2.2 OMPL로 대체할 기능

현재 `RRTStar` 내부에서 직접 수행하는 다음 기능은 OMPL로 넘긴다.

```text
_sample_joint_target()
_nearest_joint_node()
_near_joint_nodes()
_choose_joint_parent()
_rewire_joint_neighbors()
_connect_joint_goal()
_reconstruct_joint_path()
```

즉, 아래 부분만 교체한다.

```text
sampling
nearest-neighbor search
tree expansion
parent selection
rewiring
informed sampling
batch search
goal connection
planner termination
```

### 2.3 초기 전환 범위

1차 전환 범위는 **joint-space planning**으로 제한한다.

현재 `_generate_workspace()` 구현은 기존 planner에 남겨두거나 별도 legacy planner로 유지한다. OMPL adapter에서 workspace planning을 동시에 구현하면 state representation과 pose orientation 처리까지 한 번에 바뀌어 regression 범위가 커진다.

---

## 3. 목표 구조

```text
Application / Visualizer
        |
        v
PlannerBase.generate(start_q, goal_q)
        |
        v
OMPLPlanner._generate_joint_space()
        |
        +-- JointStateCodec
        |     full raw q <-> active normalized OMPL state
        |
        +-- OMPLPlannerFactory
        |     RRT / RRTConnect / RRTstar /
        |     InformedRRTstar / BITstar
        |
        +-- StateValidityAdapter
        |     OMPL state -> full q
        |     -> Pinocchio/Coal collision
        |     -> workspace constraint
        |
        +-- Motion validation
        |     Phase 1: OMPL discrete motion validator
        |     Phase 2: custom backend edge validator
        |
        +-- PathExtractor
        |     OMPL PathGeometric -> full raw q path
        |
        v
PlannerBase.verify_path()
        |
        v
TargetPlanningResult / RobotPlanningResult
```

향후 optimizer 추가 시에는 다음 구조로 확장한다.

```text
GlobalPlannerBackend
    OMPLPlanner
        |
        v
PathOptimizer
    NoOpOptimizer
    APFOptimizer
    STOMPOptimizer
    TrajOptOptimizer
        |
        v
CommonPathEvaluator
```

OMPL planner와 optimizer를 하나의 class에 섞지 않는다.

---

## 4. 권장 디렉터리 구조

```text
plugins/
├── pluginbase/
│   └── plannerbase.py
│
├── planners/
│   ├── legacy/
│   │   └── rrt_star.py
│   │
│   └── ompl/
│       ├── __init__.py
│       ├── ompl_planner.py
│       ├── state_codec.py
│       ├── validity_adapter.py
│       ├── planner_factory.py
│       ├── objective_factory.py
│       ├── path_extractor.py
│       ├── ompl_metrics.py
│       └── ompl_planner.json
│
├── optimizers/
│   ├── optimizer_base.py
│   ├── noop_optimizer.py
│   ├── apf_optimizer.py
│   ├── stomp_optimizer.py
│   └── trajopt_optimizer.py
│
└── benchmark/
    ├── planning_case.py
    ├── planner_benchmark.py
    ├── path_evaluator.py
    └── export_results.py

tests/
├── planners/
│   └── ompl/
│       ├── test_state_codec.py
│       ├── test_validity_adapter.py
│       ├── test_ompl_planner_factory.py
│       ├── test_ompl_path_extraction.py
│       ├── test_ompl_timeout.py
│       └── test_ompl_regression.py
│
└── benchmark/
    └── test_common_path_evaluator.py
```

---

## 5. OMPL 의존성 고정

### 5.1 버전 고정

OMPL은 버전 또는 git commit을 고정한다.

권장 기준:

```text
OMPL tag: 2.0.0
Python: 현재 프로젝트 Python 버전과 동일
Binding: OMPL 공식 Python binding
```

개발 PC마다 시스템 패키지의 서로 다른 OMPL 버전을 사용하지 않는다.

다음 중 하나를 선택한다.

### 방식 A: 저장소 submodule

```text
third_party/ompl
```

장점:

- 정확한 commit 재현 가능
- Python binding과 C++ core 버전 일치
- CI와 개발 PC 구성이 동일

### 방식 B: Docker/build script에서 tag checkout

```text
scripts/install_ompl.sh
docker/requirements-ompl.txt
```

설치 완료 후 다음 정보를 로그로 남긴다.

```text
OMPL version
OMPL git commit
Python binding import path
build type
compiler version
```

### 5.2 라이선스 처리

OMPL의 BSD 라이선스 파일을 배포물의 third-party notice에 포함한다.

```text
THIRD_PARTY_NOTICES/
└── OMPL_LICENSE.txt
```

Pinocchio, Coal/hpp-fcl 등 기존 dependency notice와 함께 관리한다.

---

## 6. OMPL state space 설계

### 6.1 normalized active joint space 사용

현재 `PlannerBase`는 joint limit을 이용하여 q를 정규화한다.

```text
q_norm = (q - lower) / (upper - lower)
```

OMPL에서도 동일한 metric을 유지하기 위해 state space를 active joint별 `[0, 1]` 범위로 구성한다.

```text
OMPL state = normalized active q
OMPL bounds = [0, 1]^N_active
OMPL distance = L2
```

이렇게 하면 현재 코드의 normalized joint distance와 OMPL의 기본 RealVector L2 distance가 동일해진다.

장점:

- revolute와 prismatic joint의 단위 차이 완화
- 기존 `step_size` 해석을 최대한 유지
- Informed RRT\*와 BIT\*의 path-length objective가 동일 metric 사용
- collision resolution을 normalized distance 기준으로 정의 가능
- planner 간 objective 비교가 쉬움

### 6.2 fixed joint는 state dimension에서 제거

fixed joint를 `[value, value]` bound로 넣지 않는다.

대신 `JointStateCodec`이 active joint만 OMPL state에 저장한다.

```python
class JointStateCodec:
    def __init__(
        self,
        full_dof,
        lower,
        upper,
        active_indices,
        fixed_values,
    ):
        ...

    def full_q_to_state_values(self, q_full) -> np.ndarray:
        # full raw q -> active normalized values
        ...

    def state_to_full_q(self, state) -> np.ndarray:
        # active normalized OMPL state -> full raw q
        ...

    def apply_fixed_joints(self, q_full) -> np.ndarray:
        ...
```

필수 검증:

- start와 goal에 fixed joint 설정 적용
- active index와 fixed index가 중복되지 않음
- active dimension이 1 이상
- lower/upper가 유한하고 `upper > lower`
- 변환 round-trip 오차가 tolerance 이하

### 6.3 continuous revolute joint 처리

1차 구현은 현재 코드 동작과 동일하게 bounded RealVector로 처리한다.

무한 joint limit을 자동으로 `[-pi, pi]`로 바꾸는 기존 fallback은 production에서는 경고 또는 명시적 config가 필요하다.

```json
{
  "joint_limit_overrides": {
    "joint_name": [-3.1415926535, 3.1415926535]
  }
}
```

실제 wrap-around가 필요하면 추후 `SO2StateSpace`를 포함하는 compound state space로 확장한다. 첫 전환에서는 metric 변경을 피하기 위해 적용하지 않는다.

---

## 7. 신규 `OMPLPlanner` 인터페이스

```python
class OMPLPlanner(PlannerBase):
    use_joint_space_planning = True

    def __init__(self, config_path: str | None = None):
        super().__init__()
        self.config = load_config(config_path)
        self.algorithm = self.config["algorithm"]
        self.last_planning_status = None
        self.last_returned_path_reaches_goal = False
        self.last_ompl_stats = {}
        self.last_planner_data = None

    def _generate_workspace(
        self,
        current_pose,
        target_pose,
        step_callback=None,
    ):
        raise NotImplementedError(
            "OMPLPlanner phase 1 supports joint-space planning only"
        )

    def _generate_joint_space(
        self,
        start_q,
        goal_q,
        step_callback=None,
    ):
        ...
```

외부 호출 방식은 유지한다.

```python
path = planner.generate(start_q, goal_q)
```

기존 호출부가 planner 구현을 구분하지 않도록 factory를 둔다.

```python
def create_planner(planner_name: str, config_path=None) -> PlannerBase:
    if planner_name == "legacy_rrt_star":
        return RRTStar(config_path)

    if planner_name in {
        "rrt",
        "rrt_connect",
        "rrt_star",
        "informed_rrt_star",
        "bit_star",
    }:
        return OMPLPlanner(config_path)

    raise ValueError(f"unsupported planner: {planner_name}")
```

---

## 8. `_generate_joint_space()` 처리 순서

```text
1. start_q / goal_q shape 검사
2. fixed joint 적용
3. start/goal collision 검사
4. start/goal workspace 제한 검사
5. JointStateCodec 생성
6. OMPL state space 생성
7. SpaceInformation 생성
8. state validity callback 등록
9. motion validation resolution 설정
10. ProblemDefinition 생성
11. start/goal state 설정
12. optimization objective 설정
13. planner 생성 및 parameter 적용
14. 남은 deadline 계산
15. planner.solve()
16. solution status 판정
17. PathGeometric 추출
18. full raw q path로 변환
19. 실제 goal 연결 재검증
20. PlannerBase.verify_path() 최종 검증
21. 통계 및 debug 결과 저장
22. q path 반환
```

---

## 9. state validity adapter

OMPL은 collision geometry를 관리하지 않는다. 기존 backend를 callback에서 호출한다.

```python
class StateValidityAdapter:
    def __init__(self, planner, codec, stats):
        self.planner = planner
        self.codec = codec
        self.stats = stats

    def __call__(self, state) -> bool:
        self.stats["state_validity_calls"] += 1

        q = self.codec.state_to_full_q(state)

        if not self.planner._workspace_position_ok(q):
            self.stats["workspace_rejects"] += 1
            return False

        hit = self.planner.check_robot_collision(q)
        if hit:
            self.stats["state_collision_rejects"] += 1
            return False

        return True
```

주의사항:

- callback 내부에서 exception을 숨기지 않는다.
- timeout을 확인한다.
- 한 planner instance를 여러 thread가 동시에 사용하지 않는다.
- `plan_batch()`에서는 robot job별로 독립적인 OMPL planner instance를 생성한다.
- backend의 Pinocchio `Data` 및 `GeometryData`가 thread-safe한지 확인한다.
- 필요하면 robot/job별 collision context를 복제한다.

---

## 10. edge collision 처리

### 10.1 Phase 1: OMPL discrete motion validator

첫 구현에서는 OMPL의 기본 discrete motion validation을 사용한다.

현재 normalized joint space의 maximum extent를 기준으로 resolution fraction을 계산한다.

```text
resolution_fraction =
    normalized_collision_resolution / state_space_maximum_extent
```

적용 예:

```python
space.setLongestValidSegmentFraction(resolution_fraction)
```

Phase 1의 목적:

- 빠르게 OMPL planner를 기존 시스템에 연결
- planner 알고리즘 비교 시작
- callback과 경로 추출 검증
- 기존 `verify_path()`로 최종 안전성 확인

Phase 1 제한:

- 기존 backend의 `check_edge_collision()`을 직접 사용하지 않을 수 있음
- collision pair와 최초 충돌 alpha를 탐색 중 즉시 얻기 어려움
- 기존 RRT\*와 edge sampling 개수가 완전히 동일하지 않을 수 있음

### 10.2 Phase 2: custom motion validator

다음 조건 중 하나가 발생하면 custom validator를 추가한다.

- default discrete validator와 기존 edge checker 결과가 불일치
- collision check 시간이 전체 planning 시간의 대부분을 차지
- collision pair 및 최초 충돌 정보를 탐색 중 반드시 기록해야 함
- backend에서 CCD 또는 adaptive edge checking을 사용
- planner 간 edge collision call 수를 정확히 비교해야 함

custom validator의 역할:

```text
OMPL state A/B
    -> full raw q A/B
    -> robotics_backend.check_edge_collision()
    -> valid / invalid 반환
    -> valid/invalid motion count 기록
```

Python binding에서 안정적으로 custom `MotionValidator`를 상속하기 어렵거나 callback overhead가 큰 경우 작은 C++/nanobind bridge를 별도 작성한다.

초기 전환을 이 bridge 작업에 의존시키지 않는다.

---

## 11. planner factory

```python
def create_ompl_planner(
    algorithm,
    space_information,
    config,
):
    if algorithm == "rrt":
        planner = og.RRT(space_information)

    elif algorithm == "rrt_connect":
        planner = og.RRTConnect(space_information)

    elif algorithm == "rrt_star":
        planner = og.RRTstar(space_information)

    elif algorithm == "informed_rrt_star":
        planner = og.InformedRRTstar(space_information)

    elif algorithm == "bit_star":
        planner = og.BITstar(space_information)

    else:
        raise ValueError(f"unsupported OMPL algorithm: {algorithm}")

    apply_supported_parameters(planner, config)
    return planner
```

권장 config:

```json
{
  "backend": "ompl",
  "algorithm": "bit_star",

  "state_space": {
    "normalize_joint_space": true,
    "goal_tolerance": 0.001
  },

  "collision": {
    "motion_validator": "discrete",
    "normalized_resolution": 0.02,
    "final_verify": true
  },

  "solve": {
    "timeout_sec": 5.0,
    "allow_approximate_solution": false,
    "stop_on_first_solution": false,
    "convergence_slice_sec": 0.1
  },

  "objective": {
    "type": "path_length"
  },

  "planner": {
    "range": 0.1,
    "goal_bias": 0.05,
    "rewire_factor": 1.1,

    "bit_star": {
      "samples_per_batch": 100,
      "pruning": true
    }
  },

  "postprocess": {
    "simplify": false,
    "interpolate": false
  },

  "debug": {
    "save_planner_data": true,
    "save_convergence": true
  }
}
```

### parameter 매핑 원칙

| 기존 RRTStar 설정 | OMPL 전환 |
|---|---|
| `step_size` | RRT 계열 `range` |
| `goal_bias` | 지원 planner의 goal bias |
| `search_radius` | 직접 매핑하지 않음 |
| `max_iter` | 기본적으로 time limit으로 교체 |
| `early_stop_on_goal` | termination condition으로 구현 |
| `solution_patience` | cost convergence 또는 sliced solve로 구현 |
| `normalize_joint_space` | `[0,1]^N` state space |
| `pinocchio_collision_sample_resolution` | normalized motion resolution |
| `debug_exploration` | PlannerData 및 convergence snapshot |

`search_radius`는 OMPL RRT\*의 asymptotic neighborhood 계산을 깨지 않도록 직접 강제하지 않는다. 필요하면 `rewire_factor` 또는 k-nearest 설정을 algorithm별로 사용한다.

설치된 OMPL 버전에서 제공되는 parameter만 적용하도록 `hasattr` 또는 planner parameter registry를 확인한다. 지원하지 않는 parameter는 silent ignore하지 않고 warning을 남긴다.

---

## 12. optimization objective

1차 구현에서는 normalized joint-space path length를 사용한다.

```python
objective = ob.PathLengthOptimizationObjective(space_information)
problem_definition.setOptimizationObjective(objective)
```

이는 현재 `_joint_distance()` 합과 같은 의미를 갖도록 설계한다.

초기 단계에서는 다음 objective를 한 번에 섞지 않는다.

- clearance
- TCP distance
- acceleration
- jerk
- joint weight
- energy
- manipulability

먼저 planner 자체의 차이를 확인한 뒤 공통 evaluator 또는 trajectory optimizer에서 추가한다.

Informed RRT\*와 BIT\*는 informed sampling과 admissible heuristic의 영향을 받으므로 custom objective를 추가할 때 direct informed sampling 지원 여부를 별도로 검증한다.

---

## 13. start/goal 및 solution status 처리

### 13.1 사전 검사

OMPL 실행 전에 기존 검사를 유지한다.

```text
start collision
goal collision
start workspace
goal workspace
dimension mismatch
invalid joint bounds
invalid fixed joint configuration
```

### 13.2 exact solution만 기본 성공 처리

기본 설정:

```text
allow_approximate_solution = false
```

OMPL이 approximate solution을 반환하더라도 production 경로로 사용하지 않는다.

### 13.3 실제 goal 연결 보장

PathGeometric 마지막 state가 goal tolerance 안에 있어도 최종 raw `goal_q`와 다를 수 있다.

```text
last OMPL q
    |
    +-- goal과 충분히 동일 -> goal_q로 치환
    |
    +-- 다름
          |
          +-- last -> goal edge collision-free
          |       -> goal_q append
          |
          +-- collision
                  -> 실패 또는 approximate 처리
```

최종 경로는 반드시 다음 조건을 만족해야 성공이다.

```text
path[0] == start_q
path[-1] == goal_q
verify_path(path).colliding_edges == 0
verify_path(path).colliding_waypoints == 0
```

### 13.4 timeout 처리

OMPL production backend에서는 다음 기본 정책을 권장한다.

```text
exact solution 있음   -> best exact path 반환
approximate만 있음    -> 기본 실패
solution 없음         -> 빈 path
```

debug 목적으로 approximate path를 별도 metadata에 저장할 수 있다.

```python
self.last_approximate_path
self.last_planning_status = "timeout_approximate"
self.last_returned_path_reaches_goal = False
```

외부 실행부가 approximate path를 성공 경로로 오인하지 않도록 한다.

---

## 14. path 추출 및 후처리

```python
def extract_full_q_path(path_geometric, codec):
    path = []
    for index in range(path_geometric.getStateCount()):
        state = path_geometric.getState(index)
        q = codec.state_to_full_q(state)
        path.append(q)
    return remove_consecutive_duplicates(path)
```

planner 비교 시:

```text
simplify = false
interpolate = false
```

실행용 경로 생성 시:

```text
optional shortcut/simplification
optional uniform resampling
optional trajectory optimization
time parameterization
```

알고리즘 출력과 후처리 결과를 구분해서 저장한다.

```text
raw_path
simplified_path
optimized_path
execution_path
```

---

## 15. debug 및 convergence logging

### 15.1 기존 step callback 차이

현재 자체 RRT\*는 매 iteration마다 `nodes/parents`를 전달할 수 있다.

OMPL은 모든 planner에 공통적인 per-iteration Python callback을 제공하지 않으므로 기존 `step_callback`과 완전히 동일한 동작을 기대하지 않는다.

대체 방식 A:

```text
planner.solve()
planner.getPlannerData()
```

저장 항목:

- vertex 수
- edge 수
- start/goal vertex
- planner 이름
- solution cost
- solve time

대체 방식 B: sliced solve

```text
0.1 s solve
    -> solution cost snapshot
    -> PlannerData snapshot
0.1 s 추가 solve
    -> solution cost snapshot
    -> PlannerData snapshot
...
```

planner를 `clear()`하지 않고 동일 problem에서 계속 실행한다.

시간별 저장 값:

```text
elapsed_s
has_exact_solution
best_cost
vertex_count
edge_count
state_validity_calls
state_collision_rejects
workspace_rejects
motion_checks
invalid_motion_checks
```

### 15.2 기존 로그와 공존

legacy RRT\* 전용 이벤트:

```text
extend
choose_parent
rewire
connect_goal
```

OMPL 공통 로그:

```text
setup
solve_slice
solution_improved
exact_solution
approximate_solution
timeout
final_verify
```

기존 CSV schema를 억지로 유지하지 말고 공통 benchmark schema를 별도로 만든다.

---

## 16. 공통 결과 구조

향후 optimizer 비교를 위해 내부 result를 추가한다.

```python
@dataclass
class PathPlanningResult:
    planner_name: str
    success: bool
    exact: bool
    q_path: list[np.ndarray]

    status: str
    error: str | None

    solve_time: float
    first_solution_time: float | None
    best_cost: float | None

    state_validity_calls: int
    motion_validity_calls: int
    collision_rejects: int

    raw_planner_data: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
```

기존 `generate()` 호환성을 위해 외부에는 당분간 q path만 반환한다.

```python
self.last_result = result
return result.q_path
```

---

## 17. optimizer 연동을 고려한 사전 분리

OMPL 전환과 동시에 optimizer 자체를 구현하지는 않지만 다음 interface는 먼저 만든다.

```python
class PathOptimizerBase(ABC):
    @abstractmethod
    def optimize(
        self,
        seed_path: list[np.ndarray],
        context,
    ) -> list[np.ndarray]:
        ...
```

pipeline:

```python
global_result = global_planner.solve(problem)

optimized_path = optimizer.optimize(
    global_result.q_path,
    context,
)

verification = evaluator.evaluate(optimized_path)
```

이를 통해 다음 조합을 동일 runner에서 비교할 수 있다.

```text
RRTConnect
RRTConnect + APF
RRTConnect + STOMP
RRTConnect + TrajOpt

RRTstar
RRTstar + APF
RRTstar + STOMP
RRTstar + TrajOpt

InformedRRTstar
InformedRRTstar + TrajOpt

BITstar
BITstar + TrajOpt
```

---

## 18. 구현 단계

### Phase 0. 현재 baseline 고정

작업:

- 현재 legacy RRT\* 이름을 `legacy_rrt_star`로 명시
- 대표 planning case 저장
- 현재 성공률, 시간, 경로 비용, collision count 저장
- 현재 config와 random seed 기록
- 현재 결과 path JSON 저장

완료 기준:

- 동일 입력으로 기존 결과를 재실행 가능
- OMPL 전환 실패 시 즉시 rollback 가능

### Phase 1. OMPL import 및 최소 smoke test

작업:

- OMPL version 고정
- 공식 Python binding 설치
- `RealVectorStateSpace` 생성
- 단순 장애물 없는 ND planning smoke test
- RRTConnect, RRTstar, InformedRRTstar, BITstar import 확인

완료 기준:

- 프로젝트 Python 환경에서 OMPL import 성공
- 각 planner가 최소 문제를 solve
- CI 또는 설치 script에서 재현 가능

### Phase 2. `JointStateCodec` 구현

작업:

- full q와 active normalized state 변환
- fixed joint 제거
- joint limit override 처리
- start 기준 fixed value 처리
- round-trip test 작성

완료 기준:

```text
full_q -> OMPL state -> full_q
maximum error < 1e-9
fixed joint 값 불변
active joint bounds [0,1]
```

### Phase 3. state validity 연결

작업:

- OMPL state를 full q로 변환
- `_workspace_position_ok()` 호출
- `check_robot_collision()` 호출
- start/goal 사전 검사
- validity call counter 추가

완료 기준:

- random q 1,000개에 대해 기존 collision 결과와 OMPL callback 결과 일치
- start/goal collision status 일치
- workspace 제한 결과 일치

### Phase 4. discrete motion validation 연결

작업:

- normalized collision resolution 설정
- 간단한 edge collision scene 테스트
- 최종 `verify_path()` 활성화
- collision resolution sweep test 수행

완료 기준:

- 테스트 edge 집합에서 false-negative 0
- OMPL solution을 `verify_path()`가 모두 통과
- resolution 설정이 로그에 남음

### Phase 5. planner factory 구현

작업:

- RRT
- RRTConnect
- RRTstar
- InformedRRTstar
- BITstar
- 공통/개별 parameter 적용
- unsupported parameter warning

완료 기준:

동일 planning problem에서 config의 algorithm 이름만 바꾸어 모든 planner 실행 가능.

### Phase 6. status, timeout 및 path extraction

작업:

- exact/approximate 구분
- deadline 남은 시간 계산
- best exact path 추출
- start/goal endpoint 보장
- duplicate waypoint 제거
- timeout metadata 저장

완료 기준:

- timeout이 외부 job deadline을 초과하지 않음
- approximate path가 성공으로 잘못 반환되지 않음
- 성공 path는 정확한 start와 goal 포함

### Phase 7. debug 및 PlannerData

작업:

- solve summary JSON
- convergence CSV
- PlannerData vertex/edge count
- sliced solve 옵션
- solution cost 시간 변화 저장

완료 기준:

```text
planner
seed
solve time
first solution time
best cost
vertex count
edge count
state validity calls
collision rejects
final verification
```

### Phase 8. regression benchmark

비교 planner:

```text
legacy_rrt_star
rrt
rrt_connect
rrt_star
informed_rrt_star
bit_star
```

비교 조건:

- 같은 start/goal
- 같은 joint bounds
- 같은 fixed joints
- 같은 collision scene
- 같은 normalized state metric
- 같은 edge validation resolution
- 같은 time budget
- 같은 final evaluator
- planner당 최소 30개 random seed
- postprocessing 비활성화

완료 기준:

- OMPL planner 결과가 모두 final verification 통과
- failure 원인이 status로 구분됨
- 기존 application 호출부 수정이 최소화됨
- 선택 planner를 config만으로 변경 가능

### Phase 9. production switch

작업:

- 기본 planner를 `rrt_connect` 또는 benchmark 우승 planner로 변경
- legacy backend feature flag 유지
- 1개 release 동안 legacy 코드 보존
- 현장 입력에 대한 shadow benchmark 수행
- third-party license notice 반영

rollback 설정 예:

```json
{
  "planner_backend": "legacy",
  "planner_name": "legacy_rrt_star"
}
```

---

## 19. 테스트 계획

### 19.1 unit test

State codec:

- raw/normalized round trip
- prismatic joint
- revolute joint
- fixed joint
- invalid joint bounds
- start 값으로 fixed joint 설정
- explicit fixed joint 값

Validity adapter:

- collision-free state
- robot self-collision
- robot-static collision
- workspace reject
- backend exception
- deadline timeout

Path extraction:

- exact start/goal
- duplicate state
- one-state path
- fixed joint 복원
- final goal append
- final goal edge collision

### 19.2 integration test

```text
1. 장애물 없는 직선 연결
2. 중앙 장애물 우회
3. 좁은 통로
4. start collision
5. goal collision
6. unreachable goal
7. prismatic joint 이동이 큰 case
8. fixed joint가 있는 case
9. workspace 제한 위반 case
10. planning timeout
11. multi-target sequence
12. multi-robot parallel execution
```

### 19.3 edge collision parity test

random edge를 생성하고 두 방식의 결과를 비교한다.

```text
기존 backend.check_edge_collision()
OMPL discrete motion validator
```

분류:

```text
true positive
true negative
false positive
false negative
```

false-negative는 허용하지 않는다.

불일치가 있으면:

1. motion resolution 축소
2. normalized/raw 거리 계산 확인
3. custom MotionValidator 도입
4. CCD 적용 검토

### 19.4 thread-safety test

- 같은 robot에 대한 동시 planning
- 서로 다른 robot에 대한 동시 planning
- static collision scene 공유
- per-thread Pinocchio data 사용 여부 확인
- callback exception 및 cancellation 확인
- 동일 planner instance의 동시 사용 금지 확인

---

## 20. benchmark 지표

Global planner 지표:

```text
success rate
exact solution rate
first solution time
total solve time
best path cost
normalized joint path length
TCP path length
minimum clearance
state validity call count
motion validity call count
collision reject count
vertex count
edge count
peak memory
seed별 분산
```

Optimizer 추가 후 지표:

```text
optimization success rate
optimization time
seed path cost
optimized path cost
minimum clearance
smoothness
velocity cost
acceleration cost
jerk cost
joint limit violation
collision after optimization
initial path 의존성
```

경로 waypoint 수가 서로 다르므로 smoothness를 계산하기 전에 동일 기준으로 resampling한다.

---

## 21. 공정한 비교 규칙

1. 모든 planner는 같은 normalized state space를 사용한다.
2. 모든 planner는 같은 collision scene을 사용한다.
3. 모든 결과는 `PlannerBase.verify_path()`로 다시 검사한다.
4. planner 자체 비교 시 simplification과 optimizer를 끈다.
5. optimizer 비교 시 같은 seed path를 입력한다.
6. planning time과 postprocessing time을 분리한다.
7. first solution과 final solution을 분리한다.
8. approximate solution은 exact success와 분리한다.
9. random seed를 저장한다.
10. 실패 결과도 삭제하지 않고 원인을 기록한다.

---

## 22. 예상 위험과 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| Python validity callback overhead | planner 속도 저하 | profile 후 C++ bridge 검토 |
| OMPL discrete validation과 기존 edge checker 차이 | 충돌 경로 반환 가능 | final verify, resolution sweep, custom validator |
| fixed joint를 full dimension에 포함 | informed sampling/metric 문제 | active joint space로 제거 |
| `search_radius` 직접 이식 | RRT\* 수렴 특성 저하 | OMPL의 rewire 정책 사용 |
| approximate path 오인 | goal 미도달 경로 실행 | exact flag와 endpoint 검증 |
| shared Pinocchio data | 병렬 planning race | job별 context 또는 lock |
| per-iteration callback 부재 | 기존 시각화 손실 | PlannerData/sliced solve |
| planner마다 parameter가 다름 | config 오류 | algorithm별 schema 및 warning |
| 후처리 결과를 planner 성능으로 기록 | 잘못된 비교 | raw/optimized path 분리 |
| OMPL 버전 차이 | API 및 성능 재현 실패 | tag/commit 고정 |

---

## 23. 최소 구현 우선순위

### 1차 PR

```text
OMPL dependency
JointStateCodec
RRTConnect
state validity callback
discrete motion validation
path extraction
final verification
```

### 2차 PR

```text
RRT
RRTstar
InformedRRTstar
BITstar
planner factory
common config
```

### 3차 PR

```text
PlannerData
convergence snapshots
benchmark runner
legacy comparison
```

### 4차 PR

```text
custom MotionValidator
collision-call instrumentation
native bridge optimization
```

### 5차 PR

```text
PathOptimizer interface
APF
STOMP
TrajOpt
global planner + optimizer matrix benchmark
```

---

## 24. 완료 정의

OMPL standalone 전환은 다음 조건을 모두 만족하면 완료로 본다.

- 기존 application이 `generate(start_q, goal_q)` 방식으로 계속 호출 가능
- Pinocchio/Coal collision scene을 중복 구축하지 않음
- fixed joint와 normalized metric이 기존 동작과 일치
- RRT, RRTConnect, RRT\*, Informed RRT\*, BIT\*를 config로 교체 가능
- 모든 성공 경로가 final collision verification 통과
- timeout 및 approximate solution이 명확히 구분됨
- planner별 공통 benchmark 결과가 저장됨
- legacy RRT\*로 rollback 가능
- optimizer를 후단에 연결할 interface가 준비됨
- OMPL 및 하위 dependency 라이선스 notice가 배포물에 포함됨

---

## 25. 권장 최종 설정

초기 production 후보:

```text
Global planner: RRTConnect
목적: 첫 feasible path를 빠르게 생성
후처리: 없음 또는 단순 shortcut
```

최적 경로 비교 후보:

```text
RRTstar
InformedRRTstar
BITstar
```

향후 실제 사용 pipeline 후보:

```text
RRTConnect -> TrajOpt
RRTConnect -> STOMP
BITstar -> TrajOpt
Direct interpolation -> APF/TrajOpt
```

먼저 OMPL global planner backend를 안정화한 뒤 optimizer를 추가한다. OMPL 전환과 optimizer 도입을 한 PR에서 동시에 진행하지 않는다.
