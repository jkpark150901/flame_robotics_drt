| Planner | Safe Success Rate ↑ (%) | Effective Clearance ↑ (mm) | Collision Rate ↓ (%) | 검사 거리 오차 ↓ (mm) | 검사 방향 오차 ↓ (°) | Planning P95 ↓ (s) |
|---|---:|---:|---:|---:|---:|---:|
| RRTstar | 38.9 (25–55) | 82.7 | — | 1.9 | 39.9 | 320.25 |

| Planner | n 요청 | n 계획됨 | n safe | C_nom,min (mm) | C_eff,min (mm) | P_violation (%) | 접근방향 오차 (°) | L_q 정규화 | 궤적 길이 평균 (m) | 모션 시간 평균/합 (s) | DTW-RMSE (mm) | T_plan 평균/P50 (s) | timeout (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RRTstar | 36 | 24 | 14 | 2.7 | -2.3 | — | 76.4 | 0.75 | 2.73 | 5.5/130.9 | — | 126.22/55.31 | 30.6 |


설정: margin=0.005 m, d_physical_min=0.001 m, approach_zone=0.1 m, d*={'dda': 0.01, 'rt': None}, eps_d=0.01 m, eps_theta=5.0°, ccd_step=0.02, MC 0회 (추종 σ lin=0.002 m / rot=0.002 rad, 정합 σ t=0.003 m / r=0.003 rad), max collision rate=0.05, timeout=300.0 s, 속도 프로파일: 사다리꼴 회전 축 50 deg/s / 30 deg/s^2 (최대 변위 각 기준), 직동 축 0.5 m/s / 0.3 m/s^2
