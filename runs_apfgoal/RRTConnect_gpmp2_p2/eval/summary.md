| Planner | Safe Success Rate ↑ (%) | Effective Clearance ↑ (mm) | Collision Rate ↓ (%) | 검사 거리 오차 ↓ (mm) | 검사 방향 오차 ↓ (°) | Planning P95 ↓ (s) |
|---|---:|---:|---:|---:|---:|---:|
| RRTConnect+gpmp2 | 50.0 (29–71) | 128.8 | — | 3.3 | 37.7 | 689.53 |

| Planner | n 요청 | n 계획됨 | n safe | C_nom,min (mm) | C_eff,min (mm) | P_violation (%) | 접근방향 오차 (°) | L_q 정규화 | 궤적 길이 평균 (m) | 모션 시간 평균/합 (s) | DTW-RMSE (mm) | T_plan 평균/P50 (s) | timeout (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RRTConnect+gpmp2 | 18 | 16 | 9 | 19.8 | 14.8 | — | 60.6 | 0.61 | 1.12 | 3.8/60.9 | — | 152.91/56.22 | 11.1 |


설정: margin=0.005 m, d_physical_min=0.001 m, approach_zone=0.1 m, d*={'dda': 0.01, 'rt': None}, eps_d=0.01 m, eps_theta=5.0°, ccd_step=0.02, MC 0회 (추종 σ lin=0.002 m / rot=0.002 rad, 정합 σ t=0.003 m / r=0.003 rad), max collision rate=0.05, timeout=300.0 s, 속도 프로파일: 사다리꼴 회전 축 50 deg/s / 30 deg/s^2 (최대 변위 각 기준), 직동 축 0.5 m/s / 0.3 m/s^2
