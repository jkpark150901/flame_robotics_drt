# RRT Connect 초기 경로를 이용한 SAC 비용 실험

이 문서는 **저장된 초기 경로가 있는 residual 방식**을 설명한다. 초기 경로 없이 RRT처럼 탐색하며 비용을 학습하는 현재 방식은 [RRT_RANGE_COST_EXPERIMENT.md](RRT_RANGE_COST_EXPERIMENT.md)를 사용한다.

`rrt_connect_cost_experiment.json`은 저장된 `RRTConnect_none_p1/p2/p3` 경로를 기준 경로로 사용한다. 시작 자세와 저장된 IK 목표 자세는 기준 경로에서 가져온다. SAC의 액션은 기준 경로 추종 명령에 더해지는 잔차이며, 성공 시 목표 자세를 경로 끝에 붙인다. 경로의 중간은 정책이 순차 생성하므로 CHOMP처럼 고정 개수 웨이포인트를 한꺼번에 수정하는 방식은 아니다.

보상은 다음 항의 합이다. `d_i`는 각 관절 변위를 관절 범위로 나눈 벡터, `U`는 센서–배관 쌍을 제외한 링크–장애물 SDF 반발 비용이다. 장애물 SDF는 근사 거리이며 충돌 종료 판정에는 PyBullet의 실제 경로 충돌 검사를 쓴다.

| 설정 키 | 보상 항 | 기본값 |
|---|---|---:|
| `reward.goal` | 목표 도착 시 보상 | +1000 |
| `reward.collision` | 충돌 시 종료 패널티 | -7500 |
| `reward.apf` | `-apf × min(U, apf_clip)`; 현재 최대 50/스텝 | 10 |
| `reward.margin` | 안전거리 부족분 패널티 | 200 |
| `reward.length` | `-length × ||d_i||` 경로 길이 | 1 |
| `reward.curvature` | `-curvature × ||d_i-d_(i-1)||²` 경로 굴곡 | 2 |
| `reward.time` | 스텝당 시간 패널티 | 0.1 |
| `reward.residual` | 기준 경로에서 벗어나는 액션 크기 패널티 | 0.05 |

`reward.apf_baseline`은 시작·도착 자세의 불가피한 근접 비용을 주변에서 차감한다. `reward.side=0`, `reward.smooth=0`으로 두어 위 비용 항을 분리했다. 장애물 증강은 기본 실험에서 꺼 두어 코스트 가중치 비교의 장면을 일정하게 한다. 증강 실험은 별도 복제 설정에서 `env.random_obstacles.enabled=true`로 진행한다.

현재 저장된 RRT 경로에서 트랙/캐리지가 고정된 실험만 할 때 `env.fixed_joint_indices`에 0부터 시작하는 관절 인덱스를 지정할 수 있다. 기준 경로 자체가 해당 관절을 움직이면 환경은 오류를 내어 설정 불일치를 알린다. 기본값 `[]`는 고정을 강제하지 않는다.

가중치를 바꿔 별도 실험 디렉터리를 만든 뒤 훈련한다:

```bash
PYTHONPATH=python /home/dev/miniforge/envs/flame/bin/python python/rl/train_reward_experiment.py \
  --name rrt_chomp_cost_a --config python/rl/rrt_connect_cost_experiment.json \
  --keep-config-reward --reward length=1 --reward curvature=2 \
  --reward apf=10 --apf-max-penalty 50 --steps 60000 --device cpu
```

`--reward`의 `length`, `curvature`, `apf`, `margin`, `time`, `collision`, `goal`, `residual`을 바꿔 대조군을 만든다. APF 한계는 `--apf-max-penalty`로 설정한다. 우선 `--prepare-only`로 설정 파일과 명령만 만들어 검토할 수 있다. 각 실험의 `debug/rl/<name>/train.json`과 TensorBoard `custom/reward_*`에 적용값과 항별 보상이 기록된다. 학습을 새로 시작하므로 기존 체크포인트와 섞이지 않는다.

```bash
PYTHONPATH=python /home/dev/miniforge/envs/flame/bin/python python/rl/evaluate_reward_experiment.py \
  --run debug/rl/rrt_chomp_cost_a --methods sac shield zero --device cpu
```

평가는 같은 홀드아웃 대상과 시드에서 SAC, 충돌 shield, 기준 경로만 추종하는 `zero`를 비교한다. 도착률, 경로 안전률, 최소 여유거리를 우선 비교한다. CHOMP와 직접 비교할 때는 같은 RRT 초기 경로·장면·목표 자세를 사용해야 한다. 단순 보상 총합은 가중치가 달라지면 서로 비교할 수 없다.
