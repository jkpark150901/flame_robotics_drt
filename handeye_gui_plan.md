# Hand-Eye Calibration GUI — 구현 계획

simtool(`python/simtool/`) 구조를 참조한 PyQt6 단일 창 애플리케이션.

---

## 파일 구조

```
python/
└── verifytool/
    ├── __init__.py
    ├── window.py               # AppWindow(QMainWindow) — UI 로드, 탭 조율
    ├── verifytool.ui           # Qt Designer .ui (3-tab 레이아웃)
    ├── workers/
    │   ├── __init__.py
    │   ├── natnet_worker.py    # QThread: NatNet 수신 → pyqtSignal
    │   └── robot_worker.py     # QThread + asyncio: 로봇 TCP/관절 → pyqtSignal
    └── calib_runner.py         # QThread: 경로 실행 + 샘플 수집 루프
verifytool.py                      # 진입점 (simtool.py 동일 패턴)
verifytool.cfg                     # JSON 설정 (기본값 모음)
```

의존하는 기존 코드:
- `calibration/solver.py` — 수학 함수 전체 (import만)
- `tools/NatNet/NatNetClient` — NatNet 수신
- `rbpodo` — 로봇 통신

---

## 설정 파일 (`handeye.cfg`)

```json
{
  "window_title": "Hand-Eye Calibration Tool",
  "gui": "handeye.ui",
  "robot_ip": "10.0.2.7",
  "natnet_server_ip": "192.168.0.241",
  "natnet_client_ip": "auto",
  "natnet_rigid_body_id": 1,
  "speed": 400,
  "accel": 200,
  "speed_bar": 0.3,
  "settle_time": 0.5,
  "sample_count": 5,
  "min_mocap_samples": 3,
  "opencv_handeye_method": "park"
}
```

---

## UI 레이아웃 — 3탭 구성

### 공통 (탭 외부 하단)

```
┌─────────────────────────────────────────────────────────────────┐
│ [Tab 1: Connection]  [Tab 2: Calibration]  [Tab 3: Verification]│
│─────────────────────────────────────────────────────────────────│
│                      (탭 내용)                                   │
│─────────────────────────────────────────────────────────────────│
│ 로그 패널 (QPlainTextEdit, read-only, 하단 고정 120px)           │
└─────────────────────────────────────────────────────────────────┘
```

---

### Tab 1 · Connection

```
┌─ Robot ──────────────────────┐  ┌─ NatNet ─────────────────────┐
│ IP:   [10.0.2.7        ]     │  │ Server IP: [192.168.0.241   ] │
│ Speed:[400] Accel:[200]      │  │ Client IP: [auto            ] │
│ SpeedBar: [0.30]             │  │ RigidBody ID: [1]            │
│ [Connect Robot] [Disconnect] │  │ [▶ Connect] [■ Disconnect]   │
│ Status: ● Disconnected       │  │ Status: ● Disconnected       │
│                              │  │ FPS: -- Hz                   │
└──────────────────────────────┘  └──────────────────────────────┘

┌─ Live Monitor ───────────────────────────────────────────────────┐
│ Robot TCP  pos: [  xxx.x,   yyy.y,   zzz.z ] mm                 │
│            rot: [  rx.xx,   ry.yy,   rz.zz ] deg               │
│ Joints:    [ J1,   J2,   J3,   J4,   J5,   J6 ] deg            │
│ NatNet RB  pos: [  x.xxx,  y.yyy,  z.zzz ] m   tracking: ✓/✗   │
│            quat: [ qx, qy, qz, qw ]                             │
└──────────────────────────────────────────────────────────────────┘
```

- 상태 표시: `● Disconnected` / `● Connected` (색상 QLabel stylesheet)
- FPS: 1초 QTimer로 NatNetWorker 카운터 읽어 갱신
- Live Monitor: 100ms QTimer 폴링

---

### Tab 2 · Calibration

```
┌─ Path Plan ────────────────────────────┐
│ CSV: [경로/파일명.csv      ] [Browse]  │
│ Poses: 32개  Speed: [400] Accel: [200] │
│ Settle: [0.5]s  Samples/pose: [5]      │
│ Method: [Park ▼]  Outlier: [60.0] mm  │
│ [▶ Run Calibration]  [■ Stop]          │
│ Progress: ████████░░░░  18 / 32        │
└────────────────────────────────────────┘

┌─ Trajectory Plot (pyqtgraph GLViewWidget) ──────────────────────┐
│                                                                  │
│   · TCP 위치 (파란 점)   ○ RB → T_base_motive 변환 위치 (빨간 점)│
│   → 실시간 scatter, 각 자세 완료 시 점 추가                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌─ Calibration Result ─────────────────────────────────────────────┐
│ Model: handeye (Park)   Inlier: 30/32                           │
│ T_base_motive  trans: [  0.345,  1.023, -0.012 ] m             │
│                rot:   [ 0.001, 0.002, 1.571 ] rad (rotvec)      │
│ T_rb_tcp       trans: [ -0.012,  0.003,  0.150 ] m             │
│                rot:   [ 0.000, 0.000, 0.012 ] rad               │
│ RMSE pos: inlier 2.4 mm / all 4.1 mm                           │
│ RMSE rot: inlier 0.3°  / all 0.6°                               │
│ [💾 Save JSON]  [📂 Load JSON]                                  │
└──────────────────────────────────────────────────────────────────┘
```

- CSV 파일 → `calibration/solver.py:load_calibration_plan_csv()` 파싱
- Run Calibration → `CalibrationRunner(QThread)` 실행
  - 각 pose: `move_j` → settle → sample (TCP + RB)
  - 완료 pose마다 pyqtgraph에 점 추가 (Signal 경유)
- 완료 후 `calibration.solver.solve_handeye_opencv()` 호출
- Save JSON: `calibration/solver.py:save_calibration()` 형식 (기존 호환)
- Load JSON: 결과 표시 + Verification 탭으로 전달

---

### Tab 3 · Verification

```
┌─ Calibration ─────────────────────────────┐
│ 현재 로드된 캘리브레이션: [파일명.json]    │
│ T_base_motive: [0.345, 1.023, -0.012] m   │
│ [📂 Load Other JSON]                       │
└────────────────────────────────────────────┘

┌─ Verification Path ──────────────────────┐
│ CSV: [경로/파일명.csv    ] [Browse]       │
│ [▶ Run Verification]  [■ Stop]           │
│ Progress: ████░░░░░  8 / 20              │
└──────────────────────────────────────────┘

┌─ Error Plot (pyqtgraph PlotWidget) ──────────────────────────────┐
│  Position error (mm)                                             │
│  │                                                               │
│  │  ●   ●                 ← 각 pose에서의 residual              │
│  │        ●  ●   ●                                              │
│  └───────────────── pose index                                  │
│  RMSE: 3.2 mm  |  Max: 7.1 mm  |  Mean: 2.9 mm                 │
└──────────────────────────────────────────────────────────────────┘

┌─ Pose Table ─────────────────────────────────────────────────────┐
│ idx │ pose_label          │ error_mm │ rb_aligned_x │ ...       │
│  1  │ max_reach_j1_90_... │   2.4    │   0.345      │ ...       │
│  2  │ ...                 │   3.1    │   ...        │ ...       │
└──────────────────────────────────────────────────────────────────┘
```

- Run Verification → `CalibrationRunner`(동일 QThread, 수집만) + 캘리브레이션 적용 없이 residual 계산
- 각 pose 완료 시 error plot + table 실시간 갱신
- 완료 후 CSV 저장 버튼 (기존 `apply_calibration_and_save` 호환)

---

## 스레딩 모델

```
Main Thread (Qt UI)
  │
  ├─ NatNetWorker(QThread)            pyqtSignal:
  │    NatNetClient 실행 (daemon)      rb_updated(id, pos, rot)
  │    FPS 카운터 (1초마다)            fps_updated(float)
  │
  ├─ RobotWorker(QThread)             pyqtSignal:
  │    asyncio event loop              tcp_updated(list[6])
  │    CobotData.request_data() 루프  joints_updated(list[6])
  │    Queue로 이동 명령 수신          move_done()
  │
  └─ CalibrationRunner(QThread)       pyqtSignal:
       sample_poses 순서대로           pose_done(idx, record_dict)
       move_j → settle → sample        all_done(records_list)
       asyncio.run() 내부 루프         error(str)
```

- UI → CalibrationRunner 중단: `threading.Event`
- RobotWorker는 항상 살아있고, CalibrationRunner는 탭 2/3 실행 시만 활성화
- NatNet과 Robot 연결은 Tab 1에서 독립적으로 관리

---

## 기존 코드 재사용

| 재사용 대상 | 위치 |
|-------------|------|
| 수학/IO 함수 전체 | `calibration/solver.py` |
| `NatNetState`, `_on_rigid_body` | `natnet_worker.py` 내부로 이동 |
| `_resolve_client_ip`, `_validate_ip_pair` | `natnet_worker.py` 내부로 이동 |
| `calibration_joint_poses`, `load_calibration_plan_csv` | `motive_robot_calibration.py` import |
| `solve_handeye_opencv`, `handeye_residuals` | `calibration/solver.py` import |
| `save_calibration`, `apply_calibration_and_save` | `calibration/solver.py` import |

---

## 구현 순서 (단계별)

1. **골격 파일 생성**: `handeye.py`, `handeye.cfg`, `python/handeye/__init__.py`
2. **Worker 구현**: `natnet_worker.py`, `robot_worker.py` (Signal/Slot 인터페이스 확정)
3. **UI 파일 작성**: `handeye.ui` (Qt Designer XML — 3탭, 위젯 이름 확정)
4. **window.py**: AppWindow 구현 (탭 1 연결 → 탭 2 캘리브레이션 → 탭 3 검증)
5. **CalibrationRunner**: 경로 실행 + 샘플 수집 QThread
6. **통합 테스트**: 연결 → CSV 로드 → 캘리브레이션 → 결과 저장 → 검증

---

## 미결 사항

- **위치**: `python/handeye/` vs 루트의 `handeye/` — 어느 쪽?
- **pyqtgraph 3D vs 2D**: Tab 2 trajectory plot을 3D (`GLViewWidget`) 또는 2D 3-panel (X-Y, Y-Z, X-Z) 중 어느 쪽이 더 유용한지?
- **font 경로**: simtool.cfg의 `font_path` 참조 여부 (`NanumSquare` 폰트 공유)?
- **CalibrationRunner 재사용**: Tab 2(calibration)과 Tab 3(verification)이 동일 QThread를 재사용하도록 할지, 별도로 분리할지?
