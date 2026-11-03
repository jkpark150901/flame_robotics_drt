# RT / DDA joint-only 오프라인 전문 및 URDF 가시화

## 최신 모션 매핑 및 가변 주기 Servo 생성

사용자 최종 확인에 따라 **R1=RT, begin `90,45,-130,0,-90,-90` (5점)**,
**R2=DDA, begin `89.81,10.06,-133.84,65.50,-91.36,-0.01` (6점)** 입니다.
이전 출력 폴더의 R1/R2 파일명은 이 확인 이전 결과이므로 최신 생성에는
`motions.json`과 `generate_servo.py`를 사용하세요. 아래 초기 가시화 기록의 매핑보다 이 내용이 우선합니다.

프로젝트 루트에서 실행 (Python 표준 라이브러리만 필요):

```bash
# 20ms (50Hz), R1과 R2 모두 생성
python rb_servo_test/joint_only_motion/generate_servo.py --period-ms 20

# 10ms (100Hz), R1만 지정한 폴더에 생성
python rb_servo_test/joint_only_motion/generate_servo.py --period-ms 10 --robot r1_RT --output-dir /tmp/r1_servo_10ms

# 속도·가속도 및 서보 인자 지정
python rb_servo_test/joint_only_motion/generate_servo.py --period-ms 20 --speed 0.4 --accel 0.1 --t2 0.04 --gain 0.1 --alpha 0.5
```

기본 출력 폴더는 `servo_20ms/`처럼 주기에 따라 구분됩니다. 주기는 2~100ms를 지원하며,
`t1`은 지정한 주기와 같습니다. 기본 `t2=max(0.04초, 주기)`, gain=0.1, alpha=0.5입니다.
속도와 가속도 입력은 rad/s, rad/s²이며, 관절각 및 생성 전문은 도 단위입니다.
원래 linear 블록의 끝점 관절각을 포함하고 각 웨이포인트에서 정지합니다.
TCP 직선 경로를 유지하는 변환은 아닙니다. 주기를 바꾸면 보간을 다시 계산하며 단순히 명령 행을 건너뛰지 않습니다.

로봇별 파일:

- `*_setup.txt`: begin 자세로 가는 Move J 1개. 처음 한 번 수행하고 완료를 기다리는 용도입니다.
- `*_servo_j.txt`: 대기 구간까지 동일 목표를 반복한 한 반복의 Servo J 전문. begin 접근 Move J는 포함하지 않습니다.
- `*_schedule.csv`: 모든 행이 동일 주기의 command이며 phase가 motion/settle/dwell을 구분합니다. begin 도달 이후를 시간 0으로 봅니다.
- `*_trajectory.csv`: 전송 시각, 목표 도달 기준 시각(전송 시각+한 주기), 웨이포인트 번호, phase, J1~J6 관절각[deg].
- `*_waypoints.csv`: 원본 관절 목표 및 대기 시간.
- `summary.json`: 로봇 매핑, 주기, 명령 수, 예상 한 반복 시간, 지령 속도·가속도 검증 결과.

TXT는 전송 실행 프로그램이 아닙니다. TXT를 한꺼번에 전송하면 타이밍이 적용되지 않습니다.
추후 전송 측에서 setup 완료 후 CSV 시각에 맞춰 모든 command 행을 같은 주기로 보냅니다.
대기도 이미 지령점에 포함되므로 별도의 sleep/WAIT를 추가하지 않습니다.
20ms의 1초 대기는 동일 목표 명령 50개입니다. 대기가 주기로 나누어떨어지지 않으면
주기의 정수배로 올림합니다(30ms에서 1초는 34개=1.02초).
반복은 `cycle_duration_s`마다 한 body를 반복하며 setup은 반복하지 않습니다.
각 목표점에서 끝점 명령을 2회 추가하므로 원래 대기 외에 주기×2초가 더해집니다.
한 반복 예상 시간은 생성 프로파일 기준이며 실제 로봇 추종 검증 결과는 아닙니다.

20ms 생성 결과: R1 RT 1,229개/24.58초, R2 DDA 1,403개/28.06초 (setup 제외, 대기 포함).

## 초기 가시화 기록

### 대기를 포함한 Move J 출력 (최신 매핑 사용)

```bash
python rb_servo_test/joint_only_motion/generate_move_j.py --period-ms 20
```

`move_j_20ms/`에 `*_move_j.txt`, `*_move_j_schedule.csv`, `*_move_j_trajectory.csv`,
`*_setup.txt`, `summary.json`을 생성합니다. R1 RT는 5개, R2 DDA는 6개의 Move J 명령입니다.
각 명령은 `wait=1`로 완료를 기다리고 원문의 대기 시간을 적용하는 레시피입니다.
TXT의 WAIT/LOOP 주석은 별도의 실행 지침이며 제어기 명령이 아닙니다.

`--period-ms`는 Move J의 **예상 궤적 CSV 샘플 간격**입니다. Move J를 이 주기로 전송하는 뜻이 아닙니다.
CSV는 begin에서 한 반복을 시작하며 최초 begin 접근은 제외합니다.
대기 구간에는 같은 관절각을 유지하는 샘플을 포함합니다(20ms에서 1초=50개).
사용자 정의 대기 시간이 샘플 간격과 나누어떨어지지 않으면 마지막 간격만 짧게 두어 원래 대기를 정확히 유지합니다.
Move J는 별도의 Servo 종점 안정화 2샘플을 추가하지 않으므로 예상 시간은 R1 24.38초, R2 27.82초입니다.
이 시간과 중간 관절각은 사다리꼴 보간으로 계산한 참고값이며 실제 제어기의 시간·중간 궤적과 다를 수 있습니다.

`output/preview.html`을 브라우저로 열면 로봇 선택, 재생/일시정지, 시간 슬라이더와 3D 회전이 가능합니다. 필요한 JavaScript와 메시를 HTML에 포함하여 인터넷 없이 열 수 있습니다. `output/waypoints.png`는 유지한 모든 joint 목표 자세입니다.

- R1: `urdf/rb20_1900es_RT.urdf`, joint 목표 4개.
- R2: `urdf/rb10_1300e_DDA.urdf`, joint 목표 3개.
- 관절각은 원문의 도 단위이며 속도 `0.40 rad/s`, 가속도 `0.10 rad/s²`를 적용합니다. Rainbow 외부 스크립트의 도 단위로 각각 `22.9183118052 deg/s`, `5.72957795131 deg/s²`를 출력합니다.
- `move linear { ... }` 블록만 삭제합니다. 블록 밖의 `wait (1.0 sec)`는 삭제하지 않습니다. 따라서 RT의 세 번째 joint 목표 뒤에는 총 3초, DDA의 두 번째 목표 뒤에도 총 3초를 대기합니다.
- begin 자세는 두 로봇 모두 마지막 joint 목표와 같습니다. begin으로 이동한 뒤 나열한 loop body를 무한 반복하는 구성이며, 가시화는 begin에서 출발하는 한 반복을 재생합니다. 현재 로봇 위치에서 begin까지의 접근 이동은 가시화하지 않습니다.

각 로봇의 `*_move_j.txt`와 `*_servo_j.txt`는 명령과 실행 지침 주석을 담은 **오프라인 레시피**입니다. 전문을 전송하거나 소켓을 열지 않습니다. 주석의 `WAIT` / `LOOP`는 제어기에서 실행되는 문법이 아닙니다. TXT를 그대로 일괄 전송하면 대기·반복·서보 주기가 적용되지 않습니다.

`*_servo_j_schedule.csv`는 begin에 도달한 시점을 0초로 하는 한 반복의 전송 스케줄입니다. action이 `wait`인 행의 command 값은 대기 초입니다. 나머지 행의 command는 외부 스크립트 전문입니다. 각 관절이 설정 상한을 지키도록 동기화한 사다리꼴/삼각형 프로파일을 20ms 주기로 생성하며 모든 웨이포인트에서 정지합니다. 각 정지점에는 마지막 목표를 2회 추가하고 40ms를 확보합니다. Servo 파라미터는 기존 UI 기본값 `t1=0.02, t2=0.04, gain=0.1, alpha=0.5`입니다.

`*_move_j_schedule.csv`의 시각은 같은 보간 프로파일을 사용한 참고 예상치입니다. Move J 실행 시에는 시각 대신 각 명령의 실제 완료를 기다린 뒤 원문의 대기를 적용해야 합니다. Move J 제어기의 실제 시간/가감속 곡선과 Servo 보간 시간은 일치한다고 보장하지 않습니다. `*_waypoints.csv`는 begin과 남은 joint 목표, 대기 시간을 기록합니다.

가시화는 URDF의 visual 메시와 joint origin/axis를 사용한 FK입니다. 각 로봇의 장착 베이스를 좌표 원점으로 표시하며 외부 레일/슬라이드는 0으로 고정합니다. 툴 끝 경로는 URDF의 `*_link_end` 기준입니다. 실제 제어기의 별도 TCP/보정값은 반영하지 않았고 충돌 검사는 수행하지 않습니다. 화면의 메시만 격자 단순화하며 FK 관절값은 단순화하지 않습니다.

재생성에 필요한 Python 패키지: numpy, scipy, matplotlib, trimesh, pycollada, plotly, tkinter.

```bash
python rb_servo_test/joint_only_motion/generate.py
```

리니어 구간의 **끝점 관절각**도 Move J 목표로 포함하려면:

```bash
python rb_servo_test/joint_only_motion/generate.py --include-linear
```

이 결과는 `with_linear_endpoints/`에 별도로 생성됩니다. RT는 6점(P4·P5가 원래 리니어), DDA는 5점(P3·P4가 원래 리니어)입니다. 원문의 linear 블록 뒤쪽 6개 관절각을 그대로 사용하며 IK를 다시 풀지 않습니다. 각 원래 위치의 대기를 유지합니다. 속도·가속도는 모든 구간에 동일한 관절 한계 `0.40 rad/s`, `0.10 rad/s²`로 적용합니다. TCP 직선 경로와 원래 linear 모션의 Cartesian 속도는 유지하지 않습니다.

모델을 서로 바꿔 R1 모션을 DDA, R2 모션을 RT에서 확인하려면 `--swap-models`를 추가합니다. 관절각, 속도·가속도, 대기는 그대로 두고 URDF만 교환합니다. `--include-linear --swap-models`의 결과는 `with_linear_endpoints_swapped/`에 생성됩니다.

생성 과정에서 유한차분으로 모든 관절의 지령 속도·가속도 상한을 검사합니다. 측정된 상한과 한 반복의 예상 시간은 `output/summary.json`에 있습니다. 실측 로봇 추종 검증은 아닙니다.
