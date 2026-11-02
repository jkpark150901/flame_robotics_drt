# 경로 계획 결과 (플레이백 + 평가)

폴더 이름: `{알고리즘}_{최적화}_{시나리오}` - 최적화 `none` = 플래너 결과 그대로, `stomp` / `gpmp2` / `trajopt` = APF 장애물
비용 기반 경로 최적화. 시나리오 `p1` = `sample/planning_snapshot_p1_2.pkl` (PIPE NO.1, 검사 자세 6그룹 = 목표 12건).

각 폴더는 그대로 SimTool **Load Result...** 로 불러와 재생할 수 있다 (`scene_snapshot.pkl` 로 장면 복원, 목표별
`NN_<robot>_<pose>/joint_states.csv`, 실패 목표의 IK 목표 `failures.json`). 평가 결과는 `eval/`
(`summary.md`, `targets.csv`, `summary.json`, 지표 정의 `METRICS.md`).

`summary.csv`: 폴더별 평가 지표 한 줄씩 (`python python/runs_summary.py` 로 다시 생성 - 폴더를 추가하면 다시 실행).
`METRICS.md`: 평가 지표 정의 (summary.csv 컬럼 + 목표별/전체 지표 상세, runs_summary.py 가 함께 생성).

공통 조건: OMPL, 목표당 300 s, step 0.1, 상태검사 상한 16000회, collision-free IK (재시도 + 목표 = 시작 자세와 가까운 해),
C-space 제약, 리니어 트랙 원점 +0.5 m 기준. 평가: 몬테카를로 제외, 속도 프로파일 회전 50 °/s, 30 °/s² / 직동 0.5 m/s, 0.3 m/s².

| 폴더 | 계획 성공 / 안전 성공 (12건 중) | 원본 |
|---|---|---|
| `RRTConnect_none_p1` | 12 / 6 | `debug/eval_p1_20261030_0036_nearestIK/RRTConnect` |
| `RRTstar_none_p1` | 8 / 5 | `debug/eval_p1_20261030_0036_nearestIK/RRTstar` |
| `InformedRRTstar_none_p1` | 9 / 6 | `debug/eval_p1_20261030_0036_nearestIK/InformedRRTstar` |
| `RRTstar_stomp_p1` | 8 / 5 | `debug/eval_p1_20261030_0106_apf_opt/RRTstar+stomp` |
| `RRTstar_trajopt_p1` | 8 / 4 | `debug/eval_p1_20261030_0106_apf_opt/RRTstar+trajopt` |
| `InformedRRTstar_stomp_p1` | 8 / 6 | `debug/eval_p1_20261030_0106_apf_opt/InformedRRTstar+stomp` |

| `RRTConnect_stomp_p1` | 12 / 6 | `debug/eval_p1_20261030_0106_apf_opt/RRTConnect+stomp` |
| `RRTstar_gpmp2_p1` | 9 / 6 | `debug/eval_p1_20261030_0106_apf_opt/RRTstar+gpmp2` |
| `InformedRRTstar_gpmp2_p1` | 9 / 6 | `debug/eval_p1_20261030_0106_apf_opt/InformedRRTstar+gpmp2` |
| `InformedRRTstar_trajopt_p1` | 9 / 6 | `debug/eval_p1_20261030_0106_apf_opt/InformedRRTstar+trajopt` |

| `RRTConnect_gpmp2_p1` | 12 / 6 | `debug/eval_p1_20261030_0106_apf_opt/RRTConnect+gpmp2` |
| `RRTConnect_trajopt_p1` | 12 / 6 | `debug/eval_p1_20261030_0106_apf_opt/RRTConnect+trajopt` |

시나리오 `p2` = `sample/planning_snapshot_p2_3.pkl` (목표 18건), `p3` = `sample/planning_snapshot_p3_6.pkl` (목표 36건),
RRT 계열 최적화 없음 - 원본 `debug/eval_{p2,p3}_20261030_0907_rrt/<알고리즘>`. p1 같은 조건 재실행 결과(시드 차이 확인용)는
`debug/eval_p1_20261030_0907_rrt/` (RRTConnect 11/5, RRTstar 8/5, InformedRRTstar 7/4).

| 폴더 | 계획 성공 / 안전 성공 |
|---|---|
| `RRTConnect_none_p2` | 18/18 / 8 |
| `RRTstar_none_p2` | 16/18 / 8 |
| `InformedRRTstar_none_p2` | 11/18 / 5 |
| `RRTConnect_none_p3` | 35/36 / 18 |
| `RRTstar_none_p3` | 24/36 / 14 |
| `InformedRRTstar_none_p3` | 19/36 / 13 |

주의: `RRTConnect_stomp_p1` 의 P1-pose1 / P2-pose3 DDA 경로는 STOMP 최적화가 포지셔너에 0.2~0.5 mm 파고든 경로
(최적화 전 경로는 비관통) - 최적화 결과 채택 기준 강화 전 결과.
