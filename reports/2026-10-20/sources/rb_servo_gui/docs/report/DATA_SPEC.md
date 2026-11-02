# 실험 데이터 명세서

- 대상 보고서: 「RB 협동로봇 외부 스크립트 기반 궤적 추종 성능 평가 보고서」 (`report.md` / `report.pdf`)
- 수록 범위: **보고서에 사용된 데이터만** 수록
- 로봇: Rainbow Robotics rb20_1900es (RT), 제어기 IP 10.0.2.7, 외부 스크립트 API (TCP 5000/5001)
- 실험일: 2026-10-08
- 생성 도구: `rb_servo_gui/docs/report/make_report_dataset.py` (원본 실행 폴더에서 복사·계산, 원본은 수정하지 않음)

---

## 1. 구성

```
report_dataset_20261008/
├── DATA_SPEC.md                 이 문서
├── runs/                        비교 대상 실행 24개 (실행마다 폴더 1개)
│   └── <실행 이름>/
│       ├── config.json          실행 설정
│       ├── traj_ref.csv         원본 궤적 (50 Hz 내삽)
│       ├── traj_sub10.csv       로봇 지령점 (1/10 서브샘플링)
│       ├── follow_*.csv         측정 기록 (원시 데이터)
│       ├── compare_*_pairs.csv  DTW 매칭쌍과 쌍별 오차
│       ├── compare_*.json       실행별 지표와 판정
│       ├── velocity_*.csv       추종 속도
│       ├── smooth_*.csv         매끄러움 시계열
│       └── surface_rb20_*.csv   링크 표면 편차 시계열 (rb20 URDF, 이 묶음에서 새로 계산)
├── aux_runs/                    설명 그림용 보조 실행 2개 (비교에는 미사용)
├── inputs/                      입력 웨이포인트 3종
├── urdf/                        rb20_1900es_RT.urdf + 충돌 메시
└── tables/                      보고서 표의 원본 수치 (CSV)
```

- 모든 CSV: UTF-8. 이 묶음에서 새로 만든 CSV(`tables/*.csv`, `surface_rb20_*.csv`)는 엑셀 호환을 위해 BOM 포함(UTF-8-SIG)
- 그림(PNG)은 수록하지 않음 (보고서에 포함)

---

## 2. 실행 목록

### 2.1 비교 대상 (24회)

- 조건: 궤적 3종을 시뮬레이션과 실기로 모두 수행한 조건만
- 실행 이름 규칙: `<날짜>_<시각>[_REAL]_<방식>_step<서브샘플링 비율>` (`REAL` 이 있으면 실기, 없으면 시뮬레이션)

| 방식 | 궤적 | 시뮬레이션 | 실기 |
|---|---|---|---|
| jb2 blend 0 | A | `20261008_165637_jb2_step10` | `20261008_164607_REAL_jb2_step10` |
| jb2 blend 0 | B | `20261008_165307_jb2_step10` | `20261008_164116_REAL_jb2_step10` |
| jb2 blend 0 | C | `20261008_165514_jb2_step10` | `20261008_165012_REAL_jb2_step10` |
| jb2 blend 50 | A | `20261008_165736_jb2_step10` | `20261008_164714_REAL_jb2_step10` |
| jb2 blend 50 | B | `20261008_165339_jb2_step10` | `20261008_164156_REAL_jb2_step10` |
| jb2 blend 50 | C | `20261008_165540_jb2_step10` | `20261008_165048_REAL_jb2_step10` |
| jb2 blend 100 | A | `20261008_165824_jb2_step10` | `20261008_164805_REAL_jb2_step10` |
| jb2 blend 100 | B | `20261008_165406_jb2_step10` | `20261008_164230_REAL_jb2_step10` |
| jb2 blend 100 | C | `20261008_165602_jb2_step10` | `20261008_165117_REAL_jb2_step10` |
| servo | A | `20261008_163900_servo_step10` | `20261008_164846_REAL_servo_step10` |
| servo | B | `20261008_163425_servo_step10` | `20261008_164302_REAL_servo_step10` |
| servo | C | `20261008_163634_servo_step10` | `20261008_165140_REAL_servo_step10` |

### 2.2 보조 실행 (2회, 비교에는 미사용)

| 실행 | 조건 | 용도 |
|---|---|---|
| `20261008_132802_servo_step10` | servo, 궤적 C, 시뮬레이션, gain 1, 마지막 점 반복 없음 | 종료 오버슈트 사례 (보고서 3.2절 그림) |
| `20261008_162955_servo_step10` | servo, 궤적 C, 시뮬레이션, gain 0.2, 마지막 점 30회 반복 | 지표 설명 그림의 원본 (보고서 1.5절·1.7절의 DTW 매칭, 추종 오차, 리플, 흔들림 그림) |

### 2.3 공통 조건

| 항목 | 값 |
|---|---|
| 내삽 | 시간 부여 `v_max = 20 deg/s` (구간 하한 0.2 s), 3차 스플라인(시작·끝 속도 0), 50 Hz |
| 서브샘플링 | 10 (지령 간격 0.2 s, 마지막 점 포함) |
| 측정 | `reqdata` 50 Hz |
| jb2 | `move_jb2_add(J, 50, 120, 0, blend)`, `move_jb2_run(0)` — speed 50 deg/s, accel 120 deg/s², blend 0 / 50 / 100 % |
| servo | `move_servo_j(J, 0.2, 0.2, 1, 1)` 0.2 s 주기 — t1 = t2 = 0.2 s, gain 1, alpha 1, 마지막 점 30회 반복 |
| 시작 자세 이동 | `move_j(첫 점, 50, 120)` |

---

## 3. 입력 데이터 (`inputs/`)

| 파일 | 궤적 | 형식 | 단위 | 비고 |
|---|---|---|---|---|
| `waypoints_sample.csv` | A | `waypoint, rt_joint_linear_track, rt_joint_carriage, rt_base, rt_shoulder, rt_elbow, rt_wrist1, rt_wrist2, rt_wrist3` | rad (팔 6관절) | 경로 계획기 출력. 앞 2열(외부축)은 사용하지 않음 |
| `waypoints_robot1.csv` | B | `Order, J1_rad .. J6_rad, Speed, Acceleration` | rad | `Speed`, `Acceleration` 열은 사용하지 않음 |
| `waypoints_robot2.csv` | C | `Order, J1_rad .. J6_rad, Speed, Acceleration` | rad | 위와 같음 |

---

## 4. 실행 폴더 파일 명세 (`runs/<실행>/`, `aux_runs/<실행>/`)

### 4.1 공통 규칙

- 관절 순서: 0 ~ 5 = J1(base) ~ J6(wrist3)
- 각도 단위: **`traj_ref.csv`, `traj_sub10.csv` 는 rad, 나머지는 deg**
- 첫 줄이 `#` 로 시작하는 CSV: 첫 줄은 JSON 메타데이터, 둘째 줄부터 일반 CSV
- 시간 기준
  - `t` (follow): 기록 시작 시각 기준 [s]
  - `t_track`: 모션 시작 시각 기준 [s] (모션 시작 = 시작 자세에서 0.2° 넘게 벗어나기 직전 샘플)

### 4.2 `config.json` — 실행 설정

| 키 | 내용 |
|---|---|
| `requested` | 사용자가 입력한 설정 |
| `effective` | 실제 적용된 설정 (주요 키 아래) |

| `effective` 키 | 단위 | 내용 |
|---|---|---|
| `ip` | - | 제어기 IP |
| `robot_mode` | - | `simulation` / `real` |
| `traj_path` | - | 입력 웨이포인트 경로 |
| `vmax`, `interp_rate` | deg/s, Hz | 내삽 시간 부여 속도, 내삽 주기 |
| `mode`, `step` | - | `jb2` / `servo`, 서브샘플링 비율 |
| `speed`, `accel`, `blend` | deg/s, deg/s², % | jb2 및 시작 자세 이동 파라미터 |
| `t1`, `t2`, `gain`, `alpha` | s, s, -, - | servo 파라미터 (`t1`, `t2` 가 0 이면 지령 간격 사용) |
| `final_t2`, `final_hold` | s, 회 | servo 마지막 지령의 t2 (0 = t2 와 동일), 마지막 점 반복 횟수 |
| `reach_tol`, `reach_hold` | deg, s | 모션 종료 판정 허용 오차, 유지 시간 |
| `record_hz` | Hz | 측정 주기 |
| `th_*` | - | 판정 기준값 (프로그램 기본값, `th_ripple` 포함) |

### 4.3 `traj_ref.csv` — 원본 궤적

| 열 | 단위 | 내용 |
|---|---|---|
| `t` | s | 원본 시간 (0 시작, 0.02 s 간격) |
| `j0` ~ `j5` | **rad** | 3차 스플라인 내삽 관절각 |

### 4.4 `traj_sub10.csv` — 로봇 지령점

| 열 | 단위 | 내용 |
|---|---|---|
| `t` | s | 원본 시간 (0.2 s 간격, 마지막 점은 원본 끝 시각) |
| `j0` ~ `j5` | **rad** | 지령 관절각 (로봇에는 deg 로 변환해 전송) |

### 4.5 `follow_*.csv` — 측정 기록 (원시 데이터)

- 첫 줄 메타데이터 (JSON)

| 키 | 내용 |
|---|---|
| `timestamp`, `mode`, `robot_mode` | 실행 시각, 방식, 모드 |
| `step`, `n_points`, `dt` | 서브샘플링 비율, 지령점 수, 지령 간격 [s] |
| `traj`, `record_hz` | 입력 파일 이름, 측정 주기 |
| `measured_source` | 측정각 출처 (시뮬레이션 `jnt_ref`, 실기 `jnt_ang`) |
| `methods` | 명령 파라미터 |
| `marks` | 이벤트 시각 [s, `t` 기준]: `servo`/`jb2` = 추종 지령 시각, `servo_sent` = 마지막 servo 지령 전송, `*_done` = 완료 확인, `jb2_build` = 리스트 구성 시작 |

- 열

| 열 | 단위 | 내용 |
|---|---|---|
| `t` | s | 기록 시작 기준 시각 |
| `phase` | - | `<방식>_move_to_start` (시작 자세 이동), `<방식>` (추종), `<방식>_end` (종료 후) |
| `robot_state` | - | `reqdata` robot_state (1 = idle, 3 = 실행 중) |
| `meas0` ~ `meas5` | deg | 측정 관절각 (시뮬레이션: `jnt_ref`, 실기: `jnt_ang` 엔코더) |

### 4.6 `compare_*_pairs.csv` — DTW 매칭쌍

- 첫 줄 메타데이터: 명령 파라미터, `T_ref` (원본 길이 [s]), `T_track` (추종 시간 [s]), `reached`
- 행 = 추종 구간의 측정 샘플 1개

| 열 | 단위 | 내용 |
|---|---|---|
| `t_track` | s | 모션 시작 기준 측정 시각 |
| `t_ref_matched` | s | 짝지은 원본 위 점의 원본 시각 |
| `ref_idx` | - | DTW 가 고른 원본 정점 번호 (`traj_ref.csv` 행 번호, 0 시작) |
| `progress` | 0 ~ 1 | 짝지은 점의 원본 경로 진행도 (호길이 비율) |
| `meas_q0` ~ `meas_q5` | deg | 측정 관절각 `q_i` |
| `ref_q0` ~ `ref_q5` | deg | 짝지은 원본 경로 위 점 `p_i` (정점 양옆 선분에 투영한 연속 위치) |
| `err_q0` ~ `err_q5` | deg | 쌍 오차 `e_i = q_i − p_i` |
| `pair_dist` | deg | 궤적 추종 오차 `d_i = ‖e_i‖₂` |

### 4.7 `compare_*.json` — 실행별 지표와 판정

| 키 | 내용 |
|---|---|
| `source`, `mode`, `params`, `step` | 측정 파일, 방식, 명령 파라미터, 서브샘플링 비율 |
| `metrics` | 지표 (아래) |
| `verdict` | 항목별 판정 (**속도 리플 항목 포함**, 보고서는 이 항목을 제외) |
| `all_pass` | 전 항목 통과 여부 (**리플 포함 기준**, 보고서 판정과 다를 수 있음 → `tables/run_metrics.csv` 의 `verdict` 사용) |

| `metrics` 키 | 단위 | 내용 |
|---|---|---|
| `t_track`, `t_ref`, `track_ratio` | s, s, - | 추종 시간, 원본 길이, 비율 (참고값) |
| `t_latency`, `t_total`, `t_build` | s | 지령 → 모션 시작, 지령 → 완료 확인, jb2 리스트 구성(고정 대기 0.5 s 포함) |
| `path_dev_mean`, `_rms`, `_p95`, `_max` | deg | 궤적 추종 오차 `d_i` 통계 |
| `pair_rmse_after`, `time_rmse_before` | deg | 관절별 RMSE (DTW 쌍 기준 / 같은 정규화 시각 기준) |
| `cover_mean`, `_p95`, `_max` | deg | 커버리지 (원본 각 점 → 측정 경로 거리) |
| `progress_max` | 0 ~ 1 | 경로 진행도 |
| `max_step` | deg | 연속 측정 샘플 간 최대 이동 |
| `start_err`, `end_err` | deg | 시작 / 종료 각 오차 (관절별 최대) |
| `smooth_ripple` | 비율 | 속도 리플 `R` (참고 지표) |
| `smooth_wobble` | deg | 횡방향 흔들림 `W` |
| `smooth_stall` | 비율 | 정체 비율 `S` |
| `rough_ratio`, `rough_score_max` | 비율, - | 국소 거칠기 점수 기준 초과 비율, 최대 점수 |
| `v_max_joint`, `v_max_norm`, `v_mean_norm` | deg/s | 관절 최대 속도, 6관절 속력 최대 / 평균 |
| `lag_max`, `lag_min`, `lag_mean`, `lag_abs_max` | 비율 | 정규화 시간 기준 타이밍 지연 (+ 늦음 / − 빠름, 참고) |
| `lag_max_s`, `late_ratio`, `early_ratio` | s, 비율, 비율 | 지연 최대의 초 환산, 늦음 / 빠름 구간 비율 |
| `reached` | bool | 최종 목표 도달 여부 |

- 링크 표면 지표(`surf_*`)는 실험 당시 계산하지 않아 JSON 에 없음 → `surface_rb20_*.csv`, `tables/run_metrics.csv` 사용

### 4.8 `velocity_*.csv` — 추종 속도

| 열 | 단위 | 내용 |
|---|---|---|
| `t_track` | s | 모션 시작 기준 시각 |
| `v0` ~ `v5` | deg/s | 관절 속도 (부호 있음, ±0.1 s 중심 차분) |
| `speed_norm` | deg/s | 6관절 속력 `‖v‖₂` |

### 4.9 `smooth_*.csv` — 매끄러움 시계열

| 열 | 단위 | 내용 |
|---|---|---|
| `t_track` | s | 모션 시작 기준 시각 |
| `progress` | 0 ~ 1 | 경로 진행도 |
| `speed_along_path` | deg/s | 경로 속력 `v^s` (진행 호길이의 시간 변화율) |
| `trend` | deg/s | 경로 속력의 이동평균 추세 |
| `local_ripple` | 비율 | 국소 리플 `R_i` |
| `local_wobble_deg` | deg | 국소 흔들림 `W_i` |
| `score` | - | 국소 거칠기 점수 (1 = 기준선) |

### 4.10 `surface_rb20_*.csv` — 링크 표면 편차 시계열 (이 묶음에서 계산)

- 계산 조건
  - URDF: `urdf/rb20_1900es_RT.urdf` (충돌 메시), 링크당 표면 점 600개
  - 대상 링크: `rt_link1` ~ `rt_link_end` (팔과 함께 움직이는 링크, 레일·캐리지 제외), 외부축은 두 자세 모두 0
  - **매칭쌍 전체** 사용 (`compare_*_pairs.csv` 의 모든 행)
  - 측정 자세 `meas_q*` 와 원본 자세 `ref_q*` 를 각각 순기구학으로 배치해 비교

| 열 | 단위 | 내용 |
|---|---|---|
| `t_track` | s | 모션 시작 기준 시각 (`compare_*_pairs.csv` 와 같은 행) |
| `disp_max_mm` | mm | 그 순간 최악 점 변위 `D_i` (같은 메시 점끼리 거리의 최댓값) |
| `disp_p95_mm`, `disp_mean_mm` | mm | 그 순간 메시 점 변위의 95 백분위수, 평균 |
| `chamfer_mm` | mm | Chamfer 거리 (최근접 거리 평균) |
| `hausdorff_mm` | mm | Hausdorff 거리 (최근접 거리 최댓값) |

---

## 5. 요약 표 명세 (`tables/`)

### 5.1 `runs_index.csv` — 실행 목록

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

### 5.2 `run_metrics.csv` — 실행별 지표 (비교 대상 24회)

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

### 5.3 `method_summary.csv` — 방식별 요약 (보고서 3.5절 종합 표)

- 행: `jb2 blend 0`, `jb2 blend 50`, `servo`, `jb2 blend 100` (각 6회)
- 열: `n_runs`, 그리고 `path_dev_p95`, `path_dev_max`, `smooth_ripple`, `smooth_wobble`, `surf_disp_mean`, `surf_disp_p95`, `surf_disp_max` 각각의 `_median`, `_min`, `_max`

### 5.4 `sim_real_pairs.csv` — 시뮬레이션 대 실기 (보고서 3.6절)

| 열 | 단위 | 내용 |
|---|---|---|
| `method_group`, `trajectory` | - | 조건 |
| `sim_run`, `real_run` | - | 짝지은 실행 |
| `path_dist_p95_deg`, `path_dist_max_deg` | deg | 실기 측정 경로의 각 점에서 시뮬레이션 측정 경로까지 거리 (시간 무관) |
| `sim_path_dev_p95`, `real_path_dev_p95` | deg | 각각의 궤적 추종 오차 p95 |
| `sim_surf_disp_p95`, `real_surf_disp_p95` | mm | 각각의 표면 변위 p95 |

### 5.5 `trajectory_chord_error.csv` — 지령 간격에 따른 현 오차 (보고서 1.3절)

| 열 | 단위 | 내용 |
|---|---|---|
| `trajectory`, `file` | - | 궤적 |
| `duration_s` | s | 원본 길이 |
| `n_ref_points_50Hz`, `n_command_points` | 개 | 원본 점 수, 지령점 수 |
| `path_length_deg` | deg | 경로 길이 (6관절 L2 호길이) |
| `chord_err_p95_deg`, `chord_err_max_deg` | deg | 원본 각 점에서 지령 꺾은선까지 거리의 95 백분위수, 최댓값 |

### 5.6 `joint_sensitivity_rb20.csv` — 관절별 1° 감도 (보고서 3.7절)

| 열 | 단위 | 내용 |
|---|---|---|
| `joint` | - | J1 ~ J6 |
| `surface_disp_max_mm_per_1deg` | mm | 해당 관절만 1° 돌렸을 때 최악 점 변위 (궤적 A 시작 자세 기준) |

### 5.7 `link_displacement_rb20.csv` — 궤적별 링크 변위 (보고서 3.7절)

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

## 6. 주의사항

- 시뮬레이션의 측정각은 `jnt_ref` (제어기 지령 추종값)이며, 시뮬레이션에서 `jnt_ang` 는 0 으로 고정되어 기록하지 않음
- 실기 측정각 `jnt_ang` 는 관절 엔코더 값 → 링크 처짐, 감속기 백래시, 공구 끝 위치 미포함
- JSON 의 `verdict`, `all_pass` 는 속도 리플을 포함한 프로그램 판정 → 보고서 판정은 `tables/run_metrics.csv` 의 `verdict`
- 표면 편차는 URDF 와 실제 로봇이 같다는 가정에 기반 (일치 여부 미확인), URDF 에 없는 공구·케이블 미포함
- `urdf/` 에는 계산에 쓰는 충돌 메시만 수록 (URDF 가 참조하는 시각화 메시 `.dae` 는 미수록 → 시각화 용도로는 원본 저장소의 `urdf/` 사용)
- 보조 실행 `162955` 는 servo gain 0.2 조건으로, 비교 대상에서 제외한 조건임 (지표 설명 그림에만 사용)

---

## 7. 재생성

```bash
cd rb_servo_gui
# 이 데이터 묶음
PYTHONPATH=src python docs/report/make_report_dataset.py --out <출력 폴더>
# 보고서 그림과 표 수치
PYTHONPATH=src python docs/report/make_report_figures.py
```
