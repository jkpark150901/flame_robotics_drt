# plugin_pdm — Pose Determination Module

배관 용접부 비파괴검사(NDT)를 위해 **2개의 로봇팔(RT/DDA)** 의 **엔드이펙터 자세 후보**를 계산하는 플러그인.

상위 시스템(`flame` 프레임워크 기반 DRT) 에서 검사포인트(배관 표면의 한 점) 와 3D 스캔 점군(PCD/PLY) 을 받아, 충돌 없이 X-ray 촬영이 가능한 DDA-RT 자세 쌍의 집합을 산출한다.

---

## 1. 검사 시스템 개요

| 항목 | 설명 |
|------|------|
| 대상 | 직배관(straight pipe) 의 환형(circumferential) 용접부 |
| 검사 방법 | 디지털 방사선 투과검사(DR / Digital Radiography) |
| 로봇팔 | Rainbow Robotics RB10-1300E × 2대 (URDF: `rb10_1300e_RT.urdf`, `rb10_1300e_DDA.urdf`) |
| 엔드이펙터 A | **RT (Radiographic Tube)** — 방사선 조사기 |
| 엔드이펙터 B | **DDA (Digital Detector Array)** — 디지털 필름(검출기) |
| 배치 | 배관을 사이에 두고 RT ↔ DDA 가 **거의 마주보는** 형태(완전한 180° 가 아니라 약간 기울어진 ~170°) |
| 촬영 횟수 | 현재: 90° 간격 2회 / 추후: **120° 간격 3회** 로 변경 예정 |
| 좌표 규약 | DDA TCP / RT TCP 의 **X축이 배관 중심을 향함**, **Y축이 배관 축과 평행** |

상위 프레임워크는 ROS2가 아닌 **flame** (사내 동료가 개발한 ZMQ 기반 메시지 프레임워크) 이며, 플러그인은 매니저(`python/manager/pdm_window.py`) 의 `API_find_pose_candidates` 호출을 통해 구동된다.

---

## 2. 플러그인 위치와 역할

```
python/plugins/
├── plugin_pdm/         ← 이 플러그인 (Pose Determination Module)
├── plugin_ppm/         ← Path Planning Module (경로/궤적)
└── plugin_rcm/         ← Robot Control Module (로봇 제어)
```

`plugin_pdm` 은 **"검사 자세를 어디로 보낼지"** 를 결정하는 단계만 담당한다. 결정된 자세를 거기까지 **어떤 경로로 이동시킬지(plugin_ppm)**, **실제 어떻게 명령할지(plugin_rcm)** 는 별도 플러그인의 책임이다.

검사 흐름 안에서의 위치:

```
[3D 스캐너 PCD]            [사용자 클릭 / 자동 검출]
        │                          │
        ▼                          ▼
   ┌─────────────────────────────────────────┐
   │   plugin_pdm (이 플러그인)              │
   │   ─────────────────                     │
   │   1) 배관 프로파일 추정 (방향/중심/반경)│
   │   2) DDA 자세 후보 N개 생성             │
   │   3) RT1/RT2 자세 계산 (DDA 기준 ±α°)   │
   │   4) 모든 후보에 대해 PCD-mesh 충돌 검사│
   │   5) 통과한 (DDA, RT1, RT2) 그룹 반환   │
   └─────────────────────────────────────────┘
                    │
                    ▼
          plugin_ppm  →  plugin_rcm  →  실제 로봇
```

---

## 3. 디렉토리 구조

```
plugin_pdm/
├── __init__.py
├── readme.md                        ← 본 문서 (개요 + 사용법)
├── AGENTS.md                        ← AI 에이전트용 작업 가이드
├── docs/                            ← 세부 문서
│   ├── architecture.md              ← 아키텍처/데이터 흐름
│   ├── algorithm.md                 ← 알고리즘 상세 (수식·기하)
│   ├── coordinate-convention.md     ← TCP/배관 좌표 규약
│   └── roadmap.md                   ← 변경 예정 사항(120° 3회 등)
│
├── base_pd.py                       ← 추상 베이스 클래스 (PluginPDBase)
├── simple.py                        ← 더미 구현체(스켈레톤, 동작 확인용)
├── EndEffectorPoseOptimizer.py      ← 핵심 계산기 (실 알고리즘)
├── CylinderFitting.py               ← 실린더 피팅(Eberly 방법)
├── JupyterVisualizer.py             ← 노트북에서 결과를 PyVista로 가시화
├── requirements.txt                 ← 추가 파이썬 의존성
│
├── data/
│   └── robot_models v1.5/           ← 플러그인 단독 실험용 URDF/메쉬 사본
│       ├── robots/                  ← rb10_1300e{,_DDA,_RT}.urdf
│       └── meshes/                  ← link0~6, link_dda, link_rt
│
├── STL2PLY/                         ← STL → PLY/PCD 변환 유틸 (오프라인 데이터 준비)
│   ├── STL2PCD.py                   ← Open3D 균일/포아송 샘플링 기반 (권장)
│   ├── STL2PLY.py                   ← trimesh subdivide/decimation 기반
│   └── STL 파일에서 점군 데이터 생성.ipynb
│
├── demo(사용자 입력 포인트).ipynb            ← 사용자가 직접 점을 클릭해 검사하는 시나리오
├── demo(한화오션 지정 포인트).ipynb          ← 한화오션 지정 포인트 시나리오
├── 한화오션 지정 포인트 실험 결과/           ← 실험 결과 이미지 (성공/실패 케이스)
└── temp/                            ← 일시적 디버그 스크립트(`debug_*.py`, `test_*.py`)
```

> 데모 노트북과 실험 결과 폴더의 한글 파일명은 **그대로 유지**할 것 (현장 보고서/이력 추적 목적). PyCharm/VSCode/Git 환경에서 한글 파일명이 깨지지 않도록 주의한다.

---

## 4. 핵심 인터페이스

### 4.1 베이스 클래스 — `base_pd.py`

```python
class PluginPDBase(ABC):
    @abstractmethod
    def find_pose_candidates(self, pcd: np.ndarray, g_point: np.ndarray) -> List[np.ndarray]:
        """
        pcd     : (N, 3) 형태의 점군
        g_point : (3,) 검사 포인트(배관 표면 위의 한 점, world frame)
        return  : (4, 4) 변환행렬의 리스트 — DDA TCP 자세 후보들
        """
```

> 현재 베이스 시그니처는 **단일 엔드이펙터(DDA) 자세 후보** 만 반환하도록 정의되어 있다. RT 까지 포함한 그룹화된 결과는 `EndEffectorPoseOptimizer` 가 `(DDA, RT1, RT2)` dict 로 별도 제공한다. 베이스 클래스가 `(DDA, RT1, RT2)` 그룹을 반환하도록 확장될 가능성이 있으므로 `simple.py`/하위 클래스 작성 시 향후 시그니처 변경 가능성을 염두에 둘 것. (TODO — `docs/roadmap.md` 참조)

### 4.2 핵심 계산기 — `EndEffectorPoseOptimizer.py`

베이스 클래스와는 별개로, **알고리즘 본체는 이 클래스에 모여 있다**. 노트북·플러그인 양쪽에서 직접 사용한다.

```python
opt = EndEffectorPoseOptimizer(debug_mode=False)

# 1) 입력 데이터 로드
opt.load_scan_data("scan.ply", scale=0.001)               # 단위가 mm 라면 m로 변환
opt.load_DDA_from_urdf("rb10_1300e_DDA.urdf")
opt.load_RT_from_urdf ("rb10_1300e_RT.urdf")

# 2) 검사 포인트 주변에서 배관 프로파일(방향/중심/반경) 추정
opt.calculate_pipe_profile(target_point=(x, y, z))

# 3) 검사용 DDA-RT 자세 그룹 산출
pose_groups_json, pose_groups = opt.calculate_DDA_RT_pose_for_taking_xray(
    target_point=(x, y, z),
    num_candidates=8,                       # 배관 둘레를 N등분
    distance_from_dda_to_surface=0.10,      # DDA TCP ↔ 배관 표면 거리 [m]
    distance_from_dda_to_rt=0.60,           # DDA TCP ↔ RT  TCP 거리 [m]
    angle_of_rt=5.0,                        # 정확히 마주보지 않고 ±5° 기울임 (~170°)
)
```

반환되는 `pose_groups` 의 구조:

```python
[
  {
    "0":  {"DDA": [x,y,z,r,p,y], "RT1": [...], "RT2": [...]},   # 0°  자세 (현재 위치)
    "90": {"DDA": [...],         "RT1": [...], "RT2": [...]},   # 90° 회전 자세 (두 번째 촬영)
  },
  ...
]
```

`"0"` 과 `"90"` **둘 다** 충돌 없이 유효할 때만 그룹이 채택된다(단일 검사포인트는 90° 간격 2회 촬영을 모두 만족해야 1개의 검사 사이클이 완성). 120° × 3회 규칙 변경에 대한 대응은 `docs/roadmap.md` 참조.

### 4.3 부가 단일-엔드이펙터 모드 — `calculate_DDA_pose_for_detecting_welding_point`

용접부 자체를 **탐색**(검출) 하는 단계에서는 RT 없이 DDA 단독 자세 후보만 필요하다. 이 경우 별도 메서드를 사용한다 — JSON·필터링 결과·전체 후보(시각화용) 의 3-튜플을 반환.

---

## 5. 좌표/기하 규약 요약

`docs/coordinate-convention.md` 에 자세하게 기록. 핵심만:

| 축 | DDA TCP | RT TCP |
|----|---------|--------|
| X | **배관 중심을 바라봄** (DDA 위치 → 배관 축 위 투영점) | **DDA TCP 위치를 바라봄** (RT 위치 → DDA 위치) |
| Y | **배관 길이 방향과 평행** | (DDA Z축과 X축으로 결정되는 직교 보완) |
| Z | X × Y (오른손 좌표계) | DDA Z축과 동일 평면 공유 |

- 모든 자세는 `[x, y, z, roll, pitch, yaw]` (rpy 는 **라디안**, intrinsic XYZ 순서 — `scipy Rotation.from_euler("xyz", ...)`) 로 표현한다.
- RT 위치는 DDA 의 X축을 **DDA Z축 주위로 ±`angle_of_rt`** 만큼 회전한 방향으로 `distance_from_dda_to_rt` 만큼 떨어진 점이다. RT–배관축–DDA 가 정확한 일직선이 아니라 약간 기울어 있는 이유.

---

## 6. 알고리즘 단계 요약

`docs/algorithm.md` 에 수식 포함 상세 설명. 단계별 요약:

1. **배관 프로파일 추정** — `calculate_pipe_profile`
   1. 검사 포인트 주변의 작은 박스 ROI에서 점군 추출
   2. 그 점들의 법선 중앙값을 1차 법선 추정치로 사용
   3. 법선 반대 방향으로 가늘고 긴 실린더 ROI 를 던져 배관 반대편 점들을 채취
   4. 직선 투영 후 군집화 → 가장 먼 군집의 가장 먼 점을 "반대 표면점" 으로 잡고 → 1차 중심·반경 추정
   5. 1차 중심을 기준으로 구 ROI 안의 점들에 대해 **Eberly 실린더 피팅**(`CylinderFitting.fit_cylinder`) → 최종 (방향, 중심, 반경) 확정

2. **DDA 자세 후보 생성** — `__calculate_dda_pose_candidate`
   - 검사 포인트를 배관 축에 투영해 동적 회전 중심을 잡고, 배관 축에 수직인 평면에서 N등분된 위치들을 생성. 각 위치에서 X(중심을 봄) / Y(축과 평행) / Z(우수 좌표계) 회전 행렬을 직접 구성 후 rpy 변환.

3. **DDA 충돌 필터링**
   - 각 후보 자세에 DDA 메쉬를 변환(transform) 한 뒤, AABB 마진 안의 스캔 점들과 메쉬 표면 샘플 점들 간 최소 거리 ≤ 1mm 을 충돌로 판정.

4. **두 번째 촬영 자세(90°) 회전**
   - `__rotate_dda_pose_around_pipe_axis` — 로드리게스 회전으로 위치/방향을 배관 축 주위로 회전시킨다.

5. **RT 자세 계산** — `__calculate_rt_pose_for_angle`
   - DDA 의 X축을 DDA Z축 기준으로 `±angle_of_rt` 회전시킨 방향으로 `distance_from_dda_to_rt` 만큼 떨어진 위치에 RT 를 배치. RT 의 X 축은 DDA 를 향하도록(= 회전된 DDA→RT 방향의 반대) 설정.
   - 회전 행렬 행렬식이 음수면 좌수 → 우수 좌표계 보정.

6. **그룹 채택 조건**
   - DDA(0°), DDA(90°) 둘 다 비충돌, **그리고** 각 DDA 자세에서 RT1·RT2 중 최소 1개가 비충돌.

---

## 7. 디버그 모드

`EndEffectorPoseOptimizer(debug_mode=True)` 로 생성하면, 호출마다 `self.debuging_info` (dict) 에 중간 산출물을 채운다:

| 키 | 의미 |
|----|------|
| `sampling_box` | 법선 추정용 박스의 (min, max) |
| `selected_points` | 박스 안 점군(o3d.PointCloud) |
| `normal_m` | 법선 중앙값 |
| `points_in_cylinder` | 가늘고 긴 실린더 ROI 안 점들 |
| `estimated_center / estimated_radius` | 1차 중심/반경 |
| `dda_base_candidates` | 모든 DDA 위치 후보 (충돌 전) |
| `valid_base_dda_poses` | DDA 충돌 통과 후보 |
| `collision_pose_groups` | 부분만 유효해 채택되지 않은 그룹들 |

가시화는 `JupyterVisualizer.py` 의 헬퍼 (`add_cylinder`, `add_mesh`, `calculate_arrows_from_pose_candidates` 등) 와 함께 노트북에서 사용한다. 데모 노트북 두 개가 그대로 사용 예시다.

---

## 8. 환경 / 의존성

`requirements.txt` 에 명시된 패키지:

- `open3d` — 점군/메쉬, 충돌 후보의 AABB·샘플링 처리
- `numpy`, `scikit-learn`, `pandas`
- `urdf_parser_py` — DDA/RT URDF 파싱 (베이스 워크스페이스의 `urdf_parser/` 와는 다른 외부 패키지)
- `pyvista[all,trame]`, `ipympl`, `ipywidgets` — Jupyter 노트북 가시화

> 메인 워크스페이스의 `requirements.txt` (저장소 루트) 위에 본 플러그인용을 **추가 설치**하는 구조다. 단독 실행 시:
>
> ```powershell
> pip install -r requirements.txt
> pip install -r python/plugins/plugin_pdm/requirements.txt
> ```

URDF 메쉬 경로는 `file://meshes/...` 상대 표기를 사용한다 — `EndEffectorPoseOptimizer.__extract_tcp_and_end` 가 URDF 파일 경로 기준으로 `../` 로 변환해 해석한다. 따라서 **URDF 파일은 `data/robot_models v1.5/robots/`** 처럼 메쉬 디렉토리(`../meshes/...`) 를 한 단계 위에서 찾을 수 있는 위치에 두어야 한다.

---

## 9. 사용 예시 (노트북)

가장 빠른 진입점:

| 노트북 | 시나리오 |
|--------|----------|
| `demo(사용자 입력 포인트).ipynb` | PyVista 인터랙티브 픽킹으로 검사 포인트를 즉석 선택 |
| `demo(한화오션 지정 포인트).ipynb` | 사전 정의된 좌표 리스트에 대한 일괄 산출 + 결과 캡처 |

플러그인을 매니저 GUI에서 호출하는 예시는 `python/manager/pdm_window.py` 의 `on_btn_find_pose` 참조 — 콤보박스에서 선택된 플러그인 이름을 `kwargs={"plugin": ...}` 로 ZMQ publish 한다.

---

## 10. 상세 문서

| 파일 | 내용 |
|------|------|
| [docs/architecture.md](docs/architecture.md) | flame 프레임워크와의 결합 방식, 메시지/입출력 포맷 |
| [docs/algorithm.md](docs/algorithm.md) | 자세 계산 수식 / 기하 / 충돌 검사 디테일 |
| [docs/coordinate-convention.md](docs/coordinate-convention.md) | TCP·배관 좌표 규약, RPY 변환 |
| [docs/roadmap.md](docs/roadmap.md) | 90°→120° 변경 등 예정된 변경사항과 영향 범위 |
| [AGENTS.md](AGENTS.md) | AI 에이전트가 이 디렉토리에서 작업할 때의 규칙 |
