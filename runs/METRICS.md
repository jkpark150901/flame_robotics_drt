# 평가 지표 정의

## summary.csv 컬럼

| 컬럼 | 지표 | 단위 | 설명 |
|---|---|---|---|
| `run` | 결과 폴더 | - | runs/ 아래 폴더 이름 {알고리즘}_{최적화}_{시나리오} |
| `algorithm` | 경로 계획 알고리즘 | - | OMPL RRTConnect / RRTstar / InformedRRTstar |
| `optimizer` | 경로 최적화 | - | none = 플래너 결과 그대로, stomp / gpmp2 / trajopt = APF 장애물 비용 기반 최적화 |
| `scenario` | 시나리오 | - | p1 = sample/planning_snapshot_p1_2.pkl |
| `n_requests` | 요청 수 | 건 | 평가한 목표 수 (실패 포함) |
| `n_planned` | 계획 성공 수 | 건 | 경로가 만들어진 목표 수 |
| `n_safe` | 안전 성공 수 | 건 | 계획 성공 & C_eff >= 0 & 필수 비관통 & E_d <= eps_d & E_theta <= eps_theta |
| `planned_rate_pct` | 계획 성공률 | % | n_planned / n_requests |
| `SR_safe_pct` | 안전 성공률 SR_safe | % | n_safe / n_requests |
| `C_eff_mean_safe_mm` | 유효 여유거리 평균 (안전 성공) | mm | C_eff = 경로 전체 로봇-장애물 최소 거리 - 안전 마진 |
| `C_eff_mean_planned_mm` | 유효 여유거리 평균 (계획 성공) | mm |  |
| `C_eff_min_planned_mm` | 유효 여유거리 최소 (계획 성공) | mm | 음수 = 안전 마진 침범 (더 작으면 관통) |
| `C_nom_min_planned_mm` | 명목 여유거리 최소 (계획 성공) | mm | 마진 빼기 전 최소 거리, 음수 = 관통 |
| `E_d_mean_mm` | 검사 거리 오차 평균 | mm | \|최종 센서-배관 거리 - d*\| (DDA) |
| `E_theta_mean_deg` | 검사 방향 오차 평균 | deg | 최종 센서 축과 배관 표면 법선 사이 각도 (DDA, 기준 5°) |
| `approach_dir_error_mean_deg` | 접근 방향 오차 평균 | deg | 마지막 0.1 m EE 이동 방향과 표면 법선 사이 각도 |
| `L_ee_mean_m` | 궤적 길이 평균 (EE) | m | 계획 성공 경로의 TCP 경로 길이 평균 |
| `L_ee_total_m` | 궤적 길이 합 (EE) | m | 계획 성공 경로의 TCP 경로 길이 합 |
| `T_motion_mean_s` | 모션 수행 시간 평균 | s | 사다리꼴 속도 프로파일 (회전 50 °/s, 30 °/s², 구간별 최대 변위 관절 기준 / 직동 0.5 m/s, 0.3 m/s²) - 플레이백 시간과 같음 |
| `T_motion_total_s` | 모션 수행 시간 합 | s | 계획 성공 경로의 모션 시간 합 (성공 수가 다르면 직접 비교 주의) |
| `L_q_normalized_mean` | 정규화 관절 경로 길이 평균 | - | 관절 범위로 나눈 관절 공간 경로 길이 |
| `T_plan_mean_s` | 계획 시간 평균 | s | 목표 1건 계획 소요 시간 (IK + 경로 계획 + 최적화 + 검증), 실패 포함 |
| `T_plan_p50_s` | 계획 시간 중앙값 | s |  |
| `T_plan_p95_s` | 계획 시간 95% 분위 | s |  |
| `timeout_rate_pct` | 타임아웃 비율 | % | timeout(상태검사 상한 또는 시간 제한)으로 끝난 목표 비율 |
| `waypoints_mean` | 웨이포인트 수 평균 | 개 | 계획 성공 경로의 웨이포인트 수 (경로 계획/최적화 결과 q_path 점 수) |
| `waypoints_min` | 웨이포인트 수 최소 | 개 |  |
| `waypoints_max` | 웨이포인트 수 최대 | 개 |  |
| `waypoints_total` | 웨이포인트 수 합 | 개 |  |
| `eval_samples_mean` | 평가 샘플 수 평균 | 개 | 여유거리/충돌 평가 점 수 - 웨이포인트 사이를 관절 공간 0.02 (rad/m 노름) 간격으로 보간, 경로당 최대 250 (넘으면 균등 추출) |
| `n_paths_sample_capped` | 샘플 상한에 걸린 경로 수 | 개 | 평가 샘플이 250 상한에 걸려 실제 간격이 0.02보다 넓어진 경로 수 |
| `SR_safe_ci95_low_pct` | 안전 성공률 95% 신뢰구간 하한 | % | Wilson 구간 |
| `SR_safe_ci95_high_pct` | 안전 성공률 95% 신뢰구간 상한 | % | Wilson 구간 |

## 평가 지표 상세 (각 폴더 eval/targets.csv, eval/summary.json)

설정: 안전 마진 m = 0.005 m, d_physical_min = 0.001 m, approach_zone = 0.1 m, d* = {'dda': 0.01, 'rt': None}, eps_d = 0.01 m, eps_theta = 5.0 deg, ccd_step = 0.02, MC 시행 = 0회, 센서 로봇 = dda
속도 프로파일: 사다리꼴, 회전 축 50 deg/s, 30 deg/s^2 (구간별 최대 변위 각 기준), 직동 축 0.5 m/s, 0.3 m/s^2

### 목표별 지표 (targets.csv)

| 컬럼 | 지표 | 단위 | 설명 |
|---|---|---|---|
| `safe` | 안전 성공 | bool | 계획됨 & C_eff >= 0 & 필수 비관통 OK & E_d <= eps_d & E_theta <= eps_theta (MC 사용 시 P_collision <= 최대 충돌률) |
| `planned` | 계획 성공 | bool | 경로가 만들어졌는지 |
| `C_nom` | 명목 여유거리 C_nom | m | 연속 경로 전체의 로봇 링크-장애물 최소 거리 (접근 구간의 센서-배관 쌍, 가상 벽 제외) |
| `C_eff` | 유효 여유거리 C_eff | m | C_nom - 안전 마진 m (>= 0 이어야 안전) |
| `mandatory_min_distance` | 필수 최소 간격 | m | 센서-배관 포함 모든 로봇-정적 장애물 쌍의 경로 전체 최소 거리 |
| `mandatory_ok` | 필수 비관통 | bool | 필수 최소 간격 > d_physical_min 이고 자기충돌 없음 |
| `collision_samples` | 충돌 샘플 수 | 개 | 경로 샘플 중 충돌(자기충돌 포함) 수 |
| `E_d` | 검사 거리 오차 E_d | m | |최종 센서-배관 거리 - d*| |
| `E_theta_deg` | 검사 방향 오차 E_theta | deg | 최종 자세의 센서 축과 -n(센서 광선이 맞는 배관 표면 법선) 사이 각도 |
| `aim_error_deg` | 조준 오차 | deg | 센서 축과 (검사점 - TCP) 방향 사이 각도 |
| `E_pos` | TCP 위치 오차 | m | 최종 TCP 위치 - 목표 자세 위치 |
| `E_rot_deg` | TCP 자세 오차 | deg | 최종 TCP 자세와 목표 자세의 회전 차이 |
| `approach_dir_error_deg` | 접근 방향 오차 | deg | 마지막 approach_span 구간 EE 이동 방향과 -n 사이 각도 |
| `side_excess_max` | 반대편 침범 | m | 팔이 배관 축 너머 로봇 반대편으로 넘어간 최대 거리 (<= 0 이면 없음) |
| `L_ee` | 궤적 길이 (EE) | m | TCP 경로 길이 |
| `S_rev_deg` | 궤적 길이 (최대 변위 각) | deg | 구간별 최대 회전 관절 변위의 합 |
| `S_lin` | 궤적 길이 (직동 축) | m | 구간별 최대 직동 관절(리니어 트랙/캐리지) 변위의 합 |
| `T_motion` | 모션 수행 시간 | s | 구간마다 가장 느린 관절(회전 속도/직동 속도 한계 기준)이 구간 시간을 정하고 전 관절 동기 이동, 경로 전체 사다리꼴 1회 (플레이백 시간과 같음) |
| `T_rev` | 회전 축 수행 시간 | s | S_rev 를 사다리꼴 속도 프로파일(속도/가속도 한계)로 이동하는 시간 |
| `T_lin` | 직동 축 수행 시간 | s | S_lin 을 사다리꼴 속도 프로파일(직동 속도/가속도 한계)로 이동하는 시간 |
| `L_q` | 관절 경로 길이 | m+rad | 관절 공간 경로 길이 |
| `L_q_normalized` | 정규화 관절 경로 길이 | - | 관절 범위로 나눈 관절 공간 경로 길이 |
| `t_plan` | 계획 시간 T_plan | s | 목표 1건 계획 소요 시간 |
| `P_collision` | 충돌 확률 | 0~1 | Monte Carlo(추종/정합 오차) 시행 중 관통 비율 |
| `P_violation` | 위반 확률 | 0~1 | MC 시행 중 충돌, C_nom<0 또는 검사 조건 위반 비율 |
| `DTW_RMSE` | 궤적 편차 DTW-RMSE | m | MC 실행 EE 궤적과 명목 궤적의 DTW-RMSE |

### 전체 지표 (summary.json / summary.md)

| 키 | 지표 | 단위 | 설명 |
|---|---|---|---|
| `SR_safe_pct` | 안전 성공률 SR_safe | % | 안전 성공 수 / 요청 수 (95% 신뢰구간: SR_safe_ci95_pct, Wilson) |
| `planned_rate_pct` | 계획 성공률 | % | 계획 성공 수 / 요청 수 |
| `C_eff_mean_safe_mm` | 유효 여유거리 평균 (안전 성공) | mm |  |
| `C_eff_min_planned_mm` | 유효 여유거리 최소 (계획 성공) | mm |  |
| `C_nom_min_planned_mm` | 명목 여유거리 최소 (계획 성공) | mm |  |
| `E_d_mean_mm` | 검사 거리 오차 평균 | mm |  |
| `E_theta_mean_deg` | 검사 방향 오차 평균 | deg |  |
| `approach_dir_error_mean_deg` | 접근 방향 오차 평균 | deg |  |
| `L_ee_mean_m / L_ee_total_m` | 궤적 길이 평균 / 합 | m | 계획 성공 경로 |
| `T_motion_mean_s / T_motion_total_s` | 모션 수행 시간 평균 / 합 | s | 계획 성공 경로 |
| `L_q_normalized_mean` | 정규화 관절 경로 길이 평균 | - |  |
| `T_plan_mean_s / _p50_s / _p95_s` | 계획 시간 평균 / 중앙값 / 95% 분위 | s |  |
| `timeout_rate_pct` | 타임아웃 비율 | % |  |
| `P_collision_pct / P_violation_pct` | 충돌률 / 위반률 평균 | % | MC 사용 시 |
| `DTW_RMSE_mean_mm` | 궤적 편차 평균 | mm | MC 사용 시 |
