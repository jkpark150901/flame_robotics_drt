# 결과 데이터 명세서 (결과 CSV · 원본 궤적 CSV)

- 대상 보고서: 「RB 협동로봇 외부 스크립트 기반 궤적 추종 성능 평가 보고서」
- 수록 범위: 보고서 표의 원본 수치 (`results/`, CSV 7개), 실험에 쓴 궤적 (`trajectories/`, CSV 6개). 원시 측정 데이터는 미수록
- 로봇: rb20_1900es (RT), 실험일 2026-10-08, 비교 대상 24회 (jb2 blend 0 / 50 / 100, servo × 궤적 A·B·C × 시뮬레이션·실기)
- 형식: 구분자 쉼표. `results/` 는 UTF-8 (BOM 포함, 엑셀에서 바로 열림), `trajectories/` 는 UTF-8

**`results/`**

| 파일 | 내용 | 보고서 위치 |
|---|---|---|
| `runs_index.csv` | 실행 목록과 명령 파라미터 | 3.1절 |
| `run_metrics.csv` | 실행별 지표와 판정 | 3.3 ~ 3.4절 |
| `method_summary.csv` | 방식별 요약 | 3.5절 |
| `link_displacement_rb20.csv` | 궤적·방식·모드별 링크 변위 | 3.7절 |
| `sim_real_pairs.csv` | 시뮬레이션 대 실기 | 3.6절 |
| `trajectory_chord_error.csv` | 궤적별 지령 간격 현 오차 | 1.3절 |
| `joint_sensitivity_rb20.csv` | 관절별 1° 당 표면 변위 | 3.7절 |

**`trajectories/`**

| 파일 | 내용 | 보고서 위치 |
|---|---|---|
| `waypoints_sample.csv`, `waypoints_robot1.csv`, `waypoints_robot2.csv` | 궤적 A / B / C 의 입력 웨이포인트 (경로 계획기 출력) | 1.2절, 3.1절 |
| `reference_A_50Hz.csv`, `reference_B_50Hz.csv`, `reference_C_50Hz.csv` | 입력 웨이포인트를 내삽한 원본 궤적 (평가 기준) | 1.2 ~ 1.3절 |

## 0. 공통 정의

- 궤적: A `waypoints_sample` (14.67 s), B `waypoints_robot1` (8.16 s), C `waypoints_robot2` (5.78 s)
- 방식: jb2 = `move_jb2_add(J, 50, 120, 0, blend)` (blend 0 / 50 / 100), servo = `move_servo_j(J, 0.2, 0.2, 1, 1)` 0.2 s 주기, 마지막 점 30회 반복
- 궤적 추종 오차 `d_i`: 측정 관절각과 DTW 로 짝지은 원본 경로 위 점의 6관절 L2 거리 [deg]
- 표면 편차 (최악 점 변위) `D_i`: 측정 자세와 짝지은 원본 자세를 rb20_1900es_RT URDF 로 배치했을 때 같은 메시 점끼리 거리의 최댓값 [mm], 추종 구간의 매칭쌍 전체로 계산
- 통계: `_mean` 평균, `_p95` 95 백분위수, `_max` 최댓값 (모두 추종 구간 전체에 대한 값)
- 판정 (`verdict`): 속도 리플을 제외한 8개 항목 (추종 오차 p95 ≤ 1°, 최대 ≤ 2°, 커버리지 ≤ 2°, 진행도 ≥ 0.99, 샘플 간 이동 ≤ 5°, 시작/종료 각 오차 ≤ 0.5°, 흔들림 ≤ 0.3°, 정체 ≤ 2 %)

## 1. 파일별 명세

### 1.1 `runs_index.csv` — 실행 목록

| 열 | 내용 |
|---|---|
| `run` | 실행 폴더 이름 |
| `role` | `비교 대상` 또는 `보조: <용도>` |
| `method_group` | `jb2 blend 0` / `jb2 blend 50` / `jb2 blend 100` / `servo` |
| `trajectory`, `trajectory_file` | A / B / C, 입력 파일 |
| `mode` | 시뮬레이션 / 실기 |
| `command` | `move_jb2_add` / `move_servo_j` |
| `speed`, `accel`, `blend` | jb2 파라미터 [deg/s, deg/s², %] |
| `t1`, `t2`, `gain`, `alpha`, `final_hold` | servo 파라미터 [s, s, -, -, 회] |
| `subsample_step` | 서브샘플링 비율 |

### 1.2 `run_metrics.csv` — 실행별 지표 (비교 대상 24회)

| 열 | 단위 | 내용 |
|---|---|---|
| `run`, `method_group`, `trajectory`, `mode` | - | 실행 식별 |
| `t_track` | s | 추종 시간 (참고) |
| `path_dev_mean`, `path_dev_p95`, `path_dev_max` | deg | 궤적 추종 오차 |
| `cover_max`, `progress_max`, `max_step` | deg, 0 ~ 1, deg | 커버리지 최대, 진행도, 샘플 간 최대 이동 |
| `start_err`, `end_err` | deg | 시작 / 종료 각 오차 |
| `smooth_ripple`, `smooth_wobble`, `smooth_stall` | 비율, deg, 비율 | 속도 리플(참고), 횡방향 흔들림, 정체 비율 |
| `t_latency` | s | 지령 → 모션 시작 |
| `surf_disp_mean`, `surf_disp_p95`, `surf_disp_max` | mm | 최악 점 변위 `D_i` 의 추종 구간 평균, 95 백분위수, 최댓값 (rb20) |
| `surf_chamfer_max`, `surf_hausdorff_max` | mm | Chamfer, Hausdorff 의 추종 구간 최댓값 |
| `surf_worst_link` | - | 추종 구간 중 최대 변위가 나온 링크 |
| `verdict`, `failed_items` | - | **보고서 판정** (리플 제외 8개 항목), 불합격 항목 |

### 1.3 `method_summary.csv` — 방식별 요약 (보고서 3.5절 종합 표)

- 행: `jb2 blend 0`, `jb2 blend 50`, `servo`, `jb2 blend 100` (각 6회)
- 열: `n_runs`, 그리고 `path_dev_p95`, `path_dev_max`, `smooth_ripple`, `smooth_wobble`, `surf_disp_mean`, `surf_disp_p95`, `surf_disp_max` 각각의 `_median`, `_min`, `_max`

### 1.4 `sim_real_pairs.csv` — 시뮬레이션 대 실기 (보고서 3.6절)

| 열 | 단위 | 내용 |
|---|---|---|
| `method_group`, `trajectory` | - | 조건 |
| `sim_run`, `real_run` | - | 짝지은 실행 |
| `path_dist_p95_deg`, `path_dist_max_deg` | deg | 실기 측정 경로의 각 점에서 시뮬레이션 측정 경로까지 거리 (시간 무관) |
| `sim_path_dev_p95`, `real_path_dev_p95` | deg | 각각의 궤적 추종 오차 p95 |
| `sim_surf_disp_p95`, `real_surf_disp_p95` | mm | 각각의 표면 변위 p95 |

### 1.5 `trajectory_chord_error.csv` — 지령 간격에 따른 현 오차 (보고서 1.3절)

| 열 | 단위 | 내용 |
|---|---|---|
| `trajectory`, `file` | - | 궤적 |
| `duration_s` | s | 원본 길이 |
| `n_ref_points_50Hz`, `n_command_points` | 개 | 원본 점 수, 지령점 수 |
| `path_length_deg` | deg | 경로 길이 (6관절 L2 호길이) |
| `chord_err_p95_deg`, `chord_err_max_deg` | deg | 원본 각 점에서 지령 꺾은선까지 거리의 95 백분위수, 최댓값 |

### 1.6 `joint_sensitivity_rb20.csv` — 관절별 1° 감도 (보고서 3.7절)

| 열 | 단위 | 내용 |
|---|---|---|
| `joint` | - | J1 ~ J6 |
| `surface_disp_max_mm_per_1deg` | mm | 해당 관절만 1° 돌렸을 때 최악 점 변위 (궤적 A 시작 자세 기준) |

### 1.7 `link_displacement_rb20.csv` — 궤적별 링크 변위 (보고서 3.7절)

- 행 = 실행 1개 × 링크 1개 (24 × 7 = 168행)
- 링크마다 "그 순간 해당 링크의 최악 점 변위" 시계열을 매칭쌍 전체로 구해 요약

| 열 | 단위 | 내용 |
|---|---|---|
| `trajectory`, `method_group`, `mode`, `run` | - | 실행 식별 |
| `link` | - | `link1` ~ `link6`, `link_end` (URDF 의 `rt_link1` ~ `rt_link_end`, 베이스 제외) |
| `disp_max_mm` | mm | 추종 구간 중 해당 링크 변위의 최댓값 |
| `disp_p95_mm` | mm | 95 백분위수 (추종 시간의 95 % 는 이 값 이하) |
| `disp_mean_mm` | mm | 평균 |

---

## 2. 궤적 파일 명세 (`trajectories/`)

### 2.1 입력 웨이포인트 `waypoints_*.csv`

| 파일 | 궤적 | 열 | 단위 | 비고 |
|---|---|---|---|---|
| `waypoints_sample.csv` | A | `waypoint, rt_joint_linear_track, rt_joint_carriage, rt_base, rt_shoulder, rt_elbow, rt_wrist1, rt_wrist2, rt_wrist3` | rad (팔 6관절) | 앞 2열(외부축: 리니어 트랙, 캐리지)은 사용하지 않음 |
| `waypoints_robot1.csv` | B | `Order, J1_rad .. J6_rad, Speed, Acceleration` | rad | `Speed`, `Acceleration` 열은 사용하지 않음 |
| `waypoints_robot2.csv` | C | `Order, J1_rad .. J6_rad, Speed, Acceleration` | rad | 위와 같음 |

- 행 = 웨이포인트 1개, 시간 정보 없음 (순서만 의미)

### 2.2 원본 궤적 `reference_*_50Hz.csv`

- 생성: 연속 중복 웨이포인트 제거 → 구간 시간 부여 (최대 이동 관절 기준 20 deg/s, 하한 0.2 s) → 3차 스플라인 (시작·끝 속도 0) → 50 Hz 샘플링
- 같은 궤적을 쓴 모든 실행에서 동일 (실행 폴더의 `traj_ref.csv` 와 같은 내용)
- 로봇 지령점은 이 파일의 10번째 점마다 (0.2 s 간격, 마지막 점 포함)

| 열 | 단위 | 내용 |
|---|---|---|
| `t` | s | 원본 시간 (0 시작, 0.02 s 간격) |
| `j0` ~ `j5` | **rad** | 관절각 J1(base) ~ J6(wrist3) |

| 파일 | 길이 | 점 수 |
|---|---|---|
| `reference_A_50Hz.csv` | 14.67 s | 735 |
| `reference_B_50Hz.csv` | 8.16 s | 410 |
| `reference_C_50Hz.csv` | 5.78 s | 291 |

## 3. 주의사항

- 시뮬레이션 측정각은 제어기 지령 추종값 `jnt_ref`, 실기는 관절 엔코더 `jnt_ang` → 링크 처짐, 백래시, 공구 끝 위치 미포함
- 표면 편차는 URDF 와 실제 로봇이 같다는 가정에 기반 (일치 여부 미확인), 공구·케이블 미포함
- 각도 단위: 궤적 파일은 rad, 결과 CSV 는 deg
- `runs_index.csv` 의 보조 실행 2개는 보고서 설명 그림용이며 다른 표에는 포함되지 않음
