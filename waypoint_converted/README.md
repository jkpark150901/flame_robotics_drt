# waypoint_converted

`debug/debug` 의 플래너 출력(`joint_states.csv`)을 `experimental/waypoint_test/robot1_*.csv` 와 **같은 형식**으로 변환한 궤적 모음입니다.

- 형식: `Order,J1..J6,Speed,Acceleration`, 단위 **deg** (열 이름에 단위 표시가 없으면 deg 로 읽습니다), 6관절 팔만 (외부축 2개는 제거)
- `Speed`/`Acceleration` 은 robot1 파일과 같은 자리 표시값(80/100)입니다. 이 값은 의미가 없고, rb_servo_gui 는 읽지 않습니다.
- 이 폴더를 rb_servo_gui 의 궤적 파일로 그대로 지정하면 됩니다 (`traj_kind = waypoint`).
- 파일 이름: `dda_<번호>_<웨이포인트 수>wp_<팔 이동량>deg.csv`, 파일별 자세한 정보는 **`index.csv`**

## 반드시 알아둘 것

**변환한 476개는 모두 외부축(레일 linear_track, 캐리지)을 움직이는 경로입니다.** (외부축 이동이 0.01 미만인 경로는 0개, 레일 최대 6.5, 캐리지 최대 1.9)
원본은 팔과 외부축을 함께 계획하고 충돌 검증한 경로인데, 변환본은 팔 6관절만 남겼으므로 **외부축 없이 팔만 실행하면 검증된 경로와 같지 않습니다.**
`index.csv` 의 `ext_linear_track_travel_m`, `ext_carriage_travel` 열(외부축 이동량)이 작은 경로일수록 영향이 작습니다.
외부축 없이 쓰는 용도라면 `experimental/waypoint_test` 의 robot1/robot2 궤적(처음부터 6관절 전용)이 더 맞습니다.

## 포함/제외 기준

스캔한 `joint_states.csv` 1155개 중 **476개** 변환.

| 제외 사유 | 개수 |
|---|---|
| RT 로봇 (현재 장비에서 쓰지 않음) | 526 |
| 너무 짧음 (웨이포인트 < 3 또는 팔 이동 < 1°) | 57 |
| 실패 기록 있음 (summary.csv 의 failed 또는 failure_info.json) | 37 |
| 같은 팔 경로의 중복 | 23 |
| 성공 여부를 알 수 없음 (summary.csv 없음) | 19 |
| 충돌 기록 있음 (collisions.json) | 11 |
| 구버전 폴더 `rrt_guarantee_test` (README: 미세 충돌 1건 확인됨, v2 가 대체) | 6 |

- 포함: summary.csv 상태가 success 이고 `failure_info.json`/`collisions.json` 이 없는 DDA 경로.
- 중복은 팔 경로(소수 2자리)가 같으면 한 번만 남기고, 우선순위는 `rrt_guarantee_test_v2` > `planner_compare` > `algo_compare` > `single_check` > `archive`.

## 검증 수준 (`index.csv` 의 `verification`)

| 검증 수준 | 개수 | 의미 |
|---|---|---|
| 독립 재검증(0.02 rad) 무충돌 확인 | 6 | `rrt_guarantee_test_v2` — 무충돌이 실제로 확인된 유일한 세트 |
| 플래너 성공, collisions.json 없음 | 2 | `single_check` |
| 플래너 성공 (독립 재검증 없음) | 468 | 플래너가 성공으로 기록했을 뿐 충돌 검증은 확인하지 못함 |

즉 **무충돌이 확인된 것은 6개뿐**이고 나머지는 플래너 성공 기록만 있습니다. 실기에서는 별도 확인이 필요합니다.
출처 폴더: `archive` 437, `planner_compare` 17, `algo_compare` 14, `rrt_guarantee_test_v2` 6, `single_check` 2.

## 라디안으로 쓰려면

`--unit rad` 로 다시 만들면 열 이름이 `J1_rad..J6_rad` 인 rad 파일이 됩니다 (rb_servo_gui 로더가 단위 표시로 구분).

## 다시 만들기

```bash
python3 rb_servo_gui/tools/convert_debug_trajectories.py --debug-dir debug/debug --out-dir waypoint_converted [--robot dda] [--unit rad]
```
