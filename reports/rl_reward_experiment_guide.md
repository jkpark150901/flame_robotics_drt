# 리워드 조정 학습·평가 실행 안내

프로젝트 루트 `/home/dev/flame_robotics_drt`에서 실행한다. 실제 로봇 구동 없이 저장된 씬에서 시뮬레이션 학습·평가한다.

## 1. 파일

- [학습 스크립트](../python/rl/train_reward_experiment.py)
- [평가 스크립트](../python/rl/evaluate_reward_experiment.py)
- [리워드·actor/critic loss 설명](rl_reward_and_losses_20261102.md)

두 스크립트는 기존 train_sac.py / eval_sac.py를 실행한다. 같은 Python 환경에 설치된 패키지를 사용한다. 출력 디렉터리가 이미 있으면 중단하므로 실험 이름을 새로 지정한다. 학습은 체크포인트 재개 없이 매번 처음부터 실행한다.

## 2. 리워드 조정 후 학습

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/dev/miniforge/envs/flame/bin/python -u python/rl/train_reward_experiment.py \
  --name goal2000_apf10_collision7500 \
  --reward goal=2000 \
  --reward collision=7500 \
  --reward apf=10 \
  --reward apf_d0=0.3 \
  --reward side=0 \
  --apf-max-penalty 50 \
  --steps 60000
```

명령은 학습이 끝날 때까지 실행된다. 오래 실행할 때에는 tmux 등 지속되는 터미널에서 실행한다. GPU 번호는 다른 작업의 사용량을 확인한 후 선택한다. CPU 학습은 `--device cpu`를 추가한다.

| 옵션 | 기본값 / 의미 |
|---|---|
| `--name` | 필수. `debug/rl/이름`에 결과 저장 |
| `--config` | 기존 APF10/30cm 증강 실험 train.json. 저장된 cfg 래퍼 형식도 허용 |
| `--reward KEY=VALUE` | 여러 번 지정 가능. 비음수 계수 입력 |
| `--apf-max-penalty` | 50. 가중치 곱까지 포함한 APF 페널티 크기 상한 |
| `--steps` | 60000. 최대 학습 step |
| `--min-steps` | 25000. plateau 판정 시작 |
| `--window-steps` | 5000. 집계 구간 |
| `--tolerance` | 0.1. 인접 구간 reward/actor/critic 상대 변화 한도 |
| `--patience` | 3. 안정화 조건 연속 충족 횟수 |
| `--no-early-stop` | 조기 종료 없이 최대 step까지 실행 |
| `--prepare-only` | 설정·실행 명령만 저장하고 학습하지 않음 |

학습 wrapper는 기본 설정에 `goal=1000, collision=7500, side=0, apf=10`을 적용한 뒤 사용자가 지정한 `--reward`로 덮어쓴다. 다른 리워드 항목은 입력 config를 유지한다. `apf_clip`은 직접 지정하지 않고 `apf_max_penalty / apf`로 계산한다. 예: 가중치 10, 최대 페널티 50이면 clip=5다. apf=0이면 APF 항을 끈다.

조정 가능한 항목: `goal`, `collision`, `apf`, `apf_d0`, `progress`, `margin`, `safety_margin`, `side`, `time`, `smooth`, `residual`. `apf_baseline` 같은 중첩 설정은 별도 JSON을 수정해 `--config`로 전달한다.

`progress`는 현재 목표 관절 자세까지 거리 감소에 대한 보상이다. 남은 기준 경로 길이를 사용하는 보상이 아니므로 우회를 불리하게 만들 수 있다.

기본 학습/평가 분할은 각 스냅샷의 학습 24개 / 평가 7개다. 장애물 증강·기준 경로 무작위 선택 등은 base config를 따른다. 자세한 적용값은 각 실행의 train.json을 확인한다.

## 3. 평가

```bash
PYTHONNOUSERSITE=1 /home/dev/miniforge/envs/flame/bin/python -u \
  python/rl/evaluate_reward_experiment.py \
  --run debug/rl/goal2000_apf10_collision7500
```

기본 평가 구성:

- 평가 대상은 학습 run의 target_split에서 `eval` 분할 사용.
- 기준 경로는 기존 벤치마크와 같은 고정 RRTConnect 결과 사용. 이 wrapper는 프로젝트의 p1/p2/p3 씬용이다.
- 시작 노이즈·중간 시작·기준 경로 무작위 선택 비활성.
- 기본 씬 seed 810000, 장애물 증강 seed 910000/920000/930000.
- SAC, SAC+Shield, zero(학습 보정 없이 기준 경로 추종)를 각각 실행.
- 정책은 deterministic, 추가 stochastic rollout은 0회.
- 기본 추론 장치는 CPU. `--device cuda`로 변경 가능.
- 증강 평가에서는 장애물 추가 시도 확률을 1로 설정한다. 실제 배치 개수는 결과의 n_boxes로 확인한다.

```bash
# 단일 방법·단일 증강 시드로 빠른 확인
/home/dev/miniforge/envs/flame/bin/python python/rl/evaluate_reward_experiment.py \
  --run debug/rl/goal2000_apf10_collision7500 \
  --methods sac --aug-seeds 910000 --out debug/rl/goal2000_apf10_collision7500/eval_quick

# 특정 체크포인트 평가
/home/dev/miniforge/envs/flame/bin/python python/rl/evaluate_reward_experiment.py \
  --run debug/rl/goal2000_apf10_collision7500 \
  --model debug/rl/goal2000_apf10_collision7500/checkpoints/sac_49920_steps.zip \
  --out debug/rl/goal2000_apf10_collision7500/eval_checkpoint
```

체크포인트 이름은 실제 파일 목록을 확인한다. 병렬 환경 수에 따라 저장 step이 5,000의 정확한 배수가 아닐 수 있다. `--prepare-only`는 모델 없이도 평가 config/명령을 검토할 수 있다. 사용자 config의 경로는 프로젝트 루트 기준 또는 절대 경로를 사용한다.

## 4. 결과 확인

학습 디렉터리:

- `train.json`, `config.json`: 실제 리워드·학습 설정
- `final_model.zip`: 종료 시 모델
- `checkpoints/`: 중간 모델
- `training_result.json`: 실제 종료 step, 시간, 종료 이유
- `progress.json`: 주기적 진행 상황. 최종 step은 training_result.json 우선
- `convergence.jsonl`: 구간별 reward / actor loss / critic loss / 성공률
- `tb/`: TensorBoard 로그 및 항목별 리워드

평가 디렉터리(기본 `RUN/evaluation`):

- `summary.md`: 도달률, 경로 안전 통과율, C_eff 평균/최솟값
- `clean_*.json`, `aug_*.json`: 목표별 경로·실패 사유·거리·시간·증강 장애물 기록
- `*.log`: 평가 로그
- `config_*.json`, `commands.json`: 평가 재현 설정과 명령

경로 안전은 도달 성공 + C_eff >= 0 + mandatory_ok다. side 리워드를 0으로 해도 평가의 기존 반대편 제약은 남는다. 여유거리 평균은 도달 경로만 대상으로 하므로 실패율과 함께 본다.

## 5. 비교 원칙

한 번에 한 항목만 바꾸고 나머지 설정과 평가 시드를 동일하게 유지한다. 예를 들어 goal=1000/2000/3000을 별도 이름으로 학습해 비교한다. 기본 학습 seed는 기존 train_sac.py의 0으로 고정된다. 이는 초기 비교용이며 여러 학습 seed에서의 통계적 우열을 입증하지 않는다.

리워드 정의가 다르면 총 리워드와 loss 절댓값의 직접 비교는 부적절하다. 우선 도달률·안전 통과율을 보고, 그다음 여유거리와 이동 시간을 비교한다. `step_limit` 종료는 수렴을 의미하지 않는다. plateau도 휴리스틱이며 성공률이 낮은 상태로 안정화될 수 있다.
