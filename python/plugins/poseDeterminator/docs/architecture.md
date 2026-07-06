# plugin_pdm — Architecture

> 본 플러그인이 flame 기반 DRT 시스템에서 차지하는 위치, 매니저와의 결합 방식, 메시지 포맷.

## 1. 시스템 전체 구도

```
                   ┌────────────────────────┐
                   │  python/controller.py  │  ← PyQt6 메인 GUI (URDF, 슬라이더, PCD)
                   └─────────┬──────────────┘
                             │ ZMQ pub/sub (tcp://localhost:9001)
                             │ topic = "call",  payload = {function, kwargs}
                             ▼
                   ┌────────────────────────┐
                   │   python/viewer3d.py   │  ← Open3D 3D 시각화 데몬
                   └────────────────────────┘
                             ▲
                             │ 같은 ZMQ 토픽
                             │
                   ┌────────────────────────┐
                   │ python/manager/        │
                   │  pdm_window.py         │  ← 본 플러그인 호스트
                   │  rcm_window.py         │
                   └─────────┬──────────────┘
                             │ importlib 로 동적 로드
                             ▼
                   ┌────────────────────────┐
                   │ python/plugins/        │
                   │  plugin_pdm/           │  ← (이 플러그인)
                   │  plugin_ppm/           │
                   │  plugin_rcm/           │
                   └────────────────────────┘
```

- ZMQ 컨텍스트와 소켓 추상화는 `python/common/zpipe.py` 의 `AsyncZSocket`/`ZPipe` (싱글톤). controller·viewer·매니저 모두 같은 9001 포트를 공유.
- 플러그인은 ZMQ 에 직접 노출되지 않는다. 매니저가 GUI 이벤트(예: `on_btn_find_pose`)를 받아 **함수 이름 + kwargs** 를 publish 하면, 동일 프로세스 측에서 import 된 플러그인 인스턴스의 메서드를 호출하는 흐름이다.

## 2. 매니저 ↔ 플러그인 결합

`python/manager/pdm_window.py`:

```python
from python.plugins.plugin_pdm.base_pd import PluginPDBase
...
def on_btn_find_pose(self):
    selected_plugin = self.cbx_pdm_plugins.currentText()
    self.__call(function="API_find_pose_candidates",
                kwargs={"plugin": selected_plugin})
```

- 매니저는 콤보박스에 `controller.cfg["pdm_plugins"]` 의 이름을 채운다. 현재 기본값은 `["simple"]`.
- 플러그인 측은 콤보 이름과 같은 모듈명(`plugin_pdm.simple`) 을 가져야 동적 로드된다. 클래스 자체는 `PluginPDBase` 를 상속받는 첫 클래스가 채택된다 (`graphic_window.py:_load_pdm_plugin` 의 inspect 로직).

## 3. 입력 / 출력 데이터

### 3.1 입력

| 항목 | 형태 | 출처 |
|------|------|------|
| 스캔 점군(PCD/PLY) | `(N, 3)` numpy + 법선 | 3D 스캐너 또는 시뮬레이션 STL 변환(`STL2PLY/`) |
| 검사 포인트 `g_point` | `(3,)` world frame | 사용자 클릭(`JupyterVisualizer.visualize_selectable_pointcloud`) 또는 한화오션 지정 좌표 리스트 |
| DDA URDF | `rb10_1300e_DDA.urdf` | `data/robot_models v1.5/robots/` 또는 메인 워크스페이스 `urdf/` |
| RT URDF | `rb10_1300e_RT.urdf` | 동일 |
| 파라미터 | `num_candidates`, `distance_from_dda_to_surface`, `distance_from_dda_to_rt`, `angle_of_rt` | 호출자(노트북/매니저 GUI) |

### 3.2 출력 (`calculate_DDA_RT_pose_for_taking_xray`)

```python
[
  {
    "0":  {"DDA": [x,y,z,r,p,y], "RT1": [...], "RT2": [...]},
    "90": {"DDA": [...],         "RT1": [...], "RT2": [...]},
  },
  ...
]
```

- 좌표/각도는 모두 SI: 길이 m, 각도 라디안 (XYZ intrinsic Euler).
- `RT1` 은 `+angle_of_rt`, `RT2` 는 `-angle_of_rt`. 둘 중 하나만 충돌 통과해도 그룹에는 그쪽만 들어간다 — 호출자가 어느 한쪽을 선택할 수 있게.
- JSON 직렬화 버전(`pose_groups_json`)도 함께 반환되어 ZMQ 페이로드로 그대로 사용 가능.

## 4. 외부 의존성과 단독 실행성

본 플러그인은 두 가지 모드로 사용된다:

1. **매니저 통합 모드** — `python/controller.py` 가 띄운 GUI 에서 콤보박스로 선택. 이 경우 위 다이어그램 전체가 동작.
2. **노트북 단독 모드** — Jupyter에서 `EndEffectorPoseOptimizer` 를 직접 import 하여 사용. ZMQ 매니저 없이 알고리즘 검증·디버깅·시연용.

매니저 통합이 깨지는 일반적인 원인:

- `controller.cfg["pdm_plugins"]` 에 등록되지 않은 모듈명을 콤보에서 선택할 수 없음
- `find_pose_candidates` 시그니처가 베이스(`(pcd, g_point) → List[ndarray]`)와 어긋남
- `python/manager/graphic_window.py` 의 `_load_pdm_plugin` 이 발견하는 첫 번째 `PluginPDBase` 서브클래스가 의도한 클래스가 아닌 경우(같은 모듈에 둘 이상의 서브클래스를 두지 말 것)

## 5. 데이터 단위와 좌표계 결정

매니저-플러그인 인터페이스의 **모든 길이는 m, 각도는 라디안** 으로 통일한다. 스캐너에서 받은 PCD가 mm 라면:

```python
opt.load_scan_data(file_path, scale=0.001)
```

URDF 의 mesh `<scale>` 은 그대로 둔다(이미 m 기준). `link_dda.stl` / `link_rt.stl` 은 mm 단위라 URDF 의 `scale="0.001 0.001 0.001"` 로 보정되어 있다.

## 6. 향후 변경 충격 분석

- **베이스 시그니처를 `(DDA, RT1, RT2)` 그룹 반환으로 확장**: `python/manager/graphic_window.py:API_find_pose` 와 `simple.py` 동시 변경. 그룹 채택 규칙(0°·90° 둘 다 유효) 도 호출 측으로 노출 필요.
- **120° × 3회 모드 추가**: `__rotate_dda_pose_around_pipe_axis(..., 90.0)` 호출 부분을 다중 각도 루프로 일반화. `pose_groups` 의 키가 `"0"/"90"` → `"0"/"120"/"240"` 으로 늘어나면 매니저/노트북 측 키 인덱싱 모두 검토. ([roadmap.md](roadmap.md))
