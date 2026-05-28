# Hand-Eye Calibration GUI 설계 계획

## 목표

Rainbow Robotics 로봇과 OptiTrack NatNet 모션캡처를 연동하여,  
두 좌표계 간 변환 행렬을 GUI에서 조작·수집·계산·저장하는 도구.

구하는 변환:
```
T_base_motive  : 로봇 베이스 ↔ 모션캡처 월드
T_rb_tcp       : 모션캡처 Rigid Body ↔ 로봇 EF (TCP)
```
캘리브레이션 모델:
```
T_base_tcp_i = T_base_motive @ T_motive_rb_i @ T_rb_tcp
```

---

## 파일 구조 (안)

```
handeye_calibration_gui/
├── main.py                  # 진입점, QApplication 시작
├── app_state.py             # 전체 공유 상태 (연결, 샘플 리스트 등)
├── workers/
│   ├── natnet_worker.py     # NatNet 수신 스레드 → Qt Signal
│   ├── robot_worker.py      # rbpodo asyncio 워커 스레드
│   └── trajectory_runner.py # 궤적 실행 워커 (수집 루프)
├── calibration/
│   └── solver.py            # motive_robot_calibration.py 에서 분리한 수학 함수들
├── ui/
│   ├── main_window.py       # QMainWindow, 탭 레이아웃
│   ├── tab_connection.py    # 연결 설정 탭
│   ├── tab_trajectory.py    # 궤적 계획 탭
│   ├── tab_collection.py    # 데이터 수집 탭
│   ├── tab_calibration.py   # 캘리브레이션 연산 탭
│   └── widgets/
│       ├── pose_monitor.py  # 실시간 TCP / RB 위치 표시 위젯
│       └── sample_table.py  # 수집된 샘플 테이블 위젯
└── assets/                  # 아이콘 등 리소스
```

> `calibration/solver.py` 는 `motive_robot_calibration.py` 의 수학 함수를  
> 클래스/함수 단위로 임포트해 재사용. 중복 구현하지 않음.

---

## GUI 레이아웃 — 탭 구성

### Tab 1 · 연결 설정

| 항목 | 내용 |
|------|------|
| NatNet 서버 IP | 텍스트 입력 + 포트 (기본 1510/1511) |
| NatNet 클라이언트 IP | 텍스트 입력 or "auto" |
| NatNet Rigid Body ID | 정수 스핀박스 |
| NatNet 버전 강제 | 체크박스 + major.minor 입력 (force_version) |
| 로봇 IP | 텍스트 입력 (기본 10.0.2.7) |
| 속도 / 가속도 / 속도 비율 | 슬라이더 or 스핀박스 |
| 연결 / 해제 버튼 | NatNet, 로봇 각각 별도 |
| 상태 표시 | NatNet 수신율(Hz), 로봇 연결 상태, rb tracking 유효 여부 |

**실시간 모니터 (연결 후 항상 표시)**

```
NatNet RB pos : [ x, y, z ] m      tracking: ✓
Robot TCP pos : [ x, y, z ] mm     mode: Real
Robot Joints  : [ J1 .. J6 ] deg
```

---

### Tab 2 · 궤적 계획

궤적을 어떻게 생성할지 선택:

#### 모드 A — 자동 생성 (base sweep)

`motive_robot_calibration.py` 의 `calibration_joint_poses()` 와 동일한 파라미터를 GUI로 노출.

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| base_start_deg | 0 | J0 시작 각도 |
| base_stop_deg | 360 | J0 종료 각도 |
| base_step_deg | 45 | J0 간격 |
| tool_roll_sweep_deg | -60, 0, 60 | J5 추가 회전 목록 |
| max_reach_j1_90 | [0,90,0,0,0,0] | 템플릿 1 |
| max_reach_j1_45 | [0,45,0,0,0,0] | 템플릿 2 |
| half_reach_down | [0,45,-90,0,45,0] | 템플릿 3 |
| half_reach_up | [0,45,90,0,-45,0] | 템플릿 4 |

#### 모드 B — CSV 로드

기존 `calibration_plan_*.csv` 파일 (pose_label, J1..J6 형식) 직접 업로드.

#### 모드 C — 수동 입력

GUI 테이블에 관절 각도를 직접 입력하거나, 현재 로봇 관절 각도를 "현재 자세 추가" 버튼으로 등록.

---

**공통 미리보기**

- 생성된 자세 목록을 테이블로 표시 (pose_label, J1..J6)
- "플랜 CSV 저장 / 불러오기" 버튼
- 예상 소요 시간 표시 (pose 수 × (settle_time + sample_period × sample_count))

---

### Tab 3 · 데이터 수집

| 항목 | 내용 |
|------|------|
| settle_time | 각 자세 도달 후 대기 시간 (s) |
| sample_count | 자세당 평균 샘플 수 |
| min_mocap_samples | 자세 유효 인정 최소 mocap 프레임 수 |
| stale mocap 검출 | on/off + tcp_step, mocap_step 임계값 |
| 시작 자세 | start_joints 6개 관절 입력 |
| 스텝 확인 모드 | 체크박스 (각 자세마다 "다음" 버튼 클릭 필요) |

**수집 실행 UI**

```
[▶ 수집 시작]  [⏸ 일시정지]  [■ 중단]

진행:  ██████████░░░░░  18 / 32 poses

현재 자세: half_reach_up_j0_90_j5_0
  TCP  : [123.4, -56.7, 890.1] mm
  RB   : [0.432, -0.218, 1.023] m  (tracking ✓)
  샘플 : 5/5 완료

수집된 샘플: 18개  |  마지막 step Δ: +1.2 mm
```

- 각 자세 완료 후 실시간 샘플 테이블 갱신
- stale mocap / mocap timeout 샘플은 경고 색상으로 표시
- 중단해도 지금까지 수집된 샘플 유지 → 캘리브레이션 탭에서 사용 가능

---

### Tab 4 · 캘리브레이션 연산

#### 입력

- 현재 수집된 샘플 사용 (Tab 3) or 기존 CSV 파일 로드
- TCP orientation type 선택: `zyx_euler_deg` / `xyz_euler_deg` / `rotvec_deg` / `rotvec_rad`
- outlier_threshold_mm (0 = 비활성화)

#### 캘리브레이션 모델 선택

| 모드 | 설명 | 언제 사용 |
|------|------|-----------|
| **handeye** | `solve_handeye_world_tag_tcp` → `solve_handeye_absolute_ls` | T_base_motive **와** T_rb_tcp **동시** 추정. tool roll sweep 샘플 필요 |
| **point (SVD)** | `compute_T_align_svd` or `compute_T_align_with_rb_offset` | T_base_motive만 추정 (RB 원점 = TCP 가정 또는 offset 보정) |

> 모델별로 필요한 최소 샘플 수, 권장 tool_roll 여부를 UI에 안내 표시.

#### 결과 표시

```
캘리브레이션 완료
모델         : handeye
샘플         : 32개 (inlier 30 / outlier 2)

T_base_motive:
  translation: [  0.345,  1.023, -0.012 ] m
  rotation   : rotvec [ 0.001, 0.002, 1.571 ] rad

T_rb_tcp:
  translation: [ -0.012,  0.003,  0.150 ] m
  rotation   : rotvec [ 0.000, 0.000, 0.012 ] rad

RMSE (position): inlier 2.4 mm  /  all 4.1 mm
RMSE (rotation): inlier 0.3 °   /  all 0.6 °
```

- 결과 JSON 저장 (`calibration_svd.json` 호환 포맷)
- 결과 CSV 저장 (aligned 오차 포함)
- outlier 목록 표시 (pose_label, residual_mm)

---

## 스레딩 구조

```
Main Thread (Qt UI)
  │
  ├─ NatNetWorker (QThread)
  │    NatNetClient 실행, rigid_body_listener → Qt Signal(rb_id, pos, rot)
  │
  ├─ RobotWorker (QThread + asyncio event loop)
  │    rbpodo asyncio.Cobot 실행
  │    TCP 데이터, 관절 상태 → Qt Signal
  │
  └─ TrajectoryRunner (QThread, Tab 3 수집 실행 시만 활성)
       sample_poses 순서대로 move_j → settle → sample
       진행상황 → Qt Signal(progress, sample_record)
       완료 → Qt Signal(all_records)
```

- UI → Worker 명령: `QMetaObject.invokeMethod` or `Queue`
- Worker → UI 상태 갱신: Qt Signal/Slot (thread-safe)
- `NatNetState` 클래스는 기존 코드 그대로 재사용 (lock 포함)

---

## 기존 코드 재사용 계획

| 기존 함수/클래스 | 재사용 방법 |
|------------------|-------------|
| `NatNetState` | `natnet_worker.py` 에서 그대로 사용 |
| `_on_rigid_body` | `NatNetWorker` 내부로 이동, Qt Signal로 emit |
| `calibration_joint_poses()` | `trajectory.py` 에서 import |
| `load/save_calibration_plan_csv()` | `trajectory.py` 에서 import |
| `solve_handeye_*()` | `calibration/solver.py` 에서 import |
| `compute_T_align_svd()` | `calibration/solver.py` 에서 import |
| `save_calibration()` | `calibration/solver.py` 에서 import |
| `apply_calibration_and_save()` | `calibration/solver.py` 에서 import |
| `tcp_raw_to_matrix()` 외 수학 유틸 | `calibration/solver.py` 에서 import |
| `_resolve_client_ip()` / `_validate_ip_pair()` | `natnet_worker.py` 에서 import |

---

## 기술 스택

| 용도 | 라이브러리 |
|------|-----------|
| GUI | **PyQt6** (또는 PySide6) |
| 수치 계산 | numpy, scipy (기존 동일) |
| 로봇 | rbpodo (asyncio) |
| 모션캡처 | tools/NatNet/NatNetClient (기존 동일) |

---

## 미결 사항 (확인 필요)

1. **GUI 프레임워크**: PyQt6 / PySide6 / tkinter 중 선호하는 것?
2. **3D 시각화**: 수집된 샘플 포인트클라우드나 좌표계 시각화 필요 여부  
   (필요하면 pyqtgraph, open3d, vedo 등 검토)
3. **T_rb_tcp 표시 단위**: GUI에서 mm / m 중 어느 단위로?
4. **수동 자세 입력 (모드 C)**: 필요 여부
5. **수집 일시정지**: 중간에 멈췄다가 재개할 때 샘플 인덱스 유지 방식
6. **결과 검증 탭**: 캘리브레이션 후 별도 검증 궤적을 돌려 residual 확인하는 탭 추가 여부  
   (현재 `verify_calibration_trajectory.py` 와 유사한 기능)
