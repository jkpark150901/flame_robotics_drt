# plugin_pdm — Roadmap

> 예정된 변경사항과, 변경 시 영향을 받는 코드 위치 / 검증 방법.

## R1. 90° 2회 → 120° 3회

### 배경
현재는 환형 용접부의 검사를 **90° 간격 2회** 촬영(=한 검사 사이클)로 수행한다. 사양 변경에 따라 **120° 간격 3회** 로 바뀔 예정.

### 영향 범위

| 위치 | 현재 가정 | 변경 시 작업 |
|------|-----------|--------------|
| `EndEffectorPoseOptimizer.calculate_DDA_RT_pose_for_taking_xray` | 0° 와 90° 두 자세 모두 유효해야 그룹 채택 | 0°/120°/240° 세 자세 모두 유효해야 채택 (또는 N-of-3 정책 도입) |
| 같은 메서드 | `__rotate_dda_pose_around_pipe_axis(base, 90.0)` 단일 호출 | 회전 각도 리스트로 일반화: `[0, 120, 240]` 또는 `[0, 90]` 둘 다 지원 |
| `pose_groups` dict 키 | `"0"`, `"90"` | `"0"`, `"120"`, `"240"` (문자열 키) |
| 호출 측 노트북 두 개 | `group["0"]`, `group["90"]` 인덱싱 | 키 패턴 변경에 맞춰 루프로 |
| `python/manager/pdm_window.py` | 키 인덱싱 직접 사용은 없으나 결과 시각화에서 가능 | 코드 grep 후 점검 |

### 권장 인터페이스 (제안 — 미확정)

```python
opt.calculate_DDA_RT_pose_for_taking_xray(
    target_point=...,
    num_candidates=...,
    distance_from_dda_to_surface=...,
    distance_from_dda_to_rt=...,
    angle_of_rt=...,
    rotation_angles_deg=[0, 120, 240],   # ← 새 파라미터 (default=[0, 90] 로 호환 유지)
)
```

반환은:
```python
[
  { "0": {...}, "120": {...}, "240": {...} },
  ...
]
```

호환성을 위해 `rotation_angles_deg` 의 default 를 현재값(`[0, 90]`) 으로 두면 기존 호출자는 변경 없이 동작한다. 새 default 가 `[0, 120, 240]` 이 되는 시점은 사양 확정 후.

### 검증
1. 데모 노트북 두 개에 `rotation_angles_deg=[0, 120, 240]` 케이스를 한 셀씩 추가, 시각화에서 세 자세가 등간격으로 배치되는지 눈으로 확인.
2. 충돌 통과 그룹 수가 N=2 → N=3 조건 강화로 인해 줄어들 것 (정상 거동).
3. Debug 모드 `collision_pose_groups` 에 부분 채택 그룹이 들어가는지 확인.

---

## R2. 베이스 인터페이스 확장

### 현재
```python
class PluginPDBase(ABC):
    @abstractmethod
    def find_pose_candidates(self, pcd, g_point) -> List[np.ndarray]:
        ...
```

### 한계
- DDA 단일 자세만 표현 — RT 정보 누락
- `(4,4) 변환행렬 리스트` 와 `EndEffectorPoseOptimizer` 가 반환하는 `[x,y,z,r,p,y]` 6-벡터 사이에 일관성 없음
- 그룹화(0°·90° 쌍, 또는 0°/120°/240° 트리오) 표현 불가

### 제안
인터페이스를 두 단계로 분리:

```python
class PluginPDBase(ABC):
    @abstractmethod
    def find_pose_candidates(self, pcd, g_point, params: dict) -> List[dict]:
        """Return list of dict like {'0': {DDA, RT1?, RT2?}, '90': {...}}"""
```

또는 별도 메서드 `find_xray_pose_groups()` 추가하고 `find_pose_candidates` 는 단일-DDA 탐색용으로 명확화.

### 영향
- `simple.py` 시그니처 갱신
- `python/manager/graphic_window.py:_load_pdm_plugin` 의 inspect 로직은 무관(부모 클래스만 보므로)
- `python/manager/pdm_window.py:on_btn_find_pose` 가 ZMQ 로 보내는 kwargs 에 `params` 추가

---

## R3. flame 프레임워크 직접 통합

현재는 매니저(PyQt6 + ZMQ) 가 사이에 끼어 있다. 향후 `flame` 의 컴포넌트(`components/`)로 본 플러그인을 노출해 직접 메시지 수신 가능하게 할 수 있다. 이 경우:

- `bin/x86_64/flame` 바이너리가 컴포넌트 매니페스트로 본 플러그인을 로드
- ZMQ 토픽/메시지 스키마는 `bin/x86_64/drt/system_echo_replier.json` 패턴(rep 소켓, 5901 포트) 을 참고

flame 프레임워크 자체의 인터페이스가 정해지면 별도 어댑터(예: `flame_adapter.py`)를 본 디렉토리에 추가하는 것이 자연스럽다. 베이스 클래스를 깨뜨리지 않는 추가형 변경으로 가능.

---

## R4. 단위 테스트 추가

현재는 데모 노트북 의존. 회귀 방지를 위해 핵심 헬퍼들을 pytest 로 옮길 가치가 있음:

- `__rotate_dda_pose_around_pipe_axis`: 알려진 입력에 대해 90° 회전 후 위치/방향이 예상치인가
- `__calculate_rt_pose_for_angle`: `angle_deg=0` 일 때 RT-X 가 DDA-X 의 반대 방향, `±α` 일 때 dot 곱 검증
- `__extract_points_in_cylinder`: 알려진 분포에서 정확한 점들이 추출되는가
- `CylinderFitting.fit_cylinder`: 합성 실린더 점군에서 (방향, 중심, 반경) 복원 오차

테스트는 `python/plugins/plugin_pdm/tests/` 신설을 권장 — 메인 워크스페이스 pytest 가 픽업하도록 `tests/__init__.py` 만 두면 된다.

---

## R5. 결정 로그 (Decision Log)

| 일자 | 결정 | 근거 |
|------|------|------|
| 2026-05-08 | readme + AGENTS + docs 4종 신설 | deepinit 으로 플러그인 사양·아키텍처를 한 번에 명문화. 사용자 요청. |

> 본 표는 새 결정이 생길 때 가장 위에 한 줄씩 추가한다.
