# 초기 경로 없는 탐색 기반 강화학습

이 실험은 저장된 RRT 경로를 입력으로 사용하지 않는다. 데이터셋에는 현재 URDF에서 검증한 **시작 관절값과 목표 IK 관절값**만 저장한다. 각 에피소드에서 양방향 탐색 트리를 빈 경로에서 시작하고, SAC 액션(관절 수만큼의 `[-1,1]` 값)이 다음 샘플의 관절별 탐색 범위를 정한다. 트리 확장과 연결은 RRT Connect 방식이며 각 간선은 PyBullet로 충돌 검사한다. 따라서 정책은 최종 경로의 웨이포인트를 직접 수정하는 대신, 탐색할 범위를 학습한다.

`rrt_range_cost.json`에서 코스트 비율을 변경한다. 경로 길이는 관절 가동 범위로 정규화한 관절 공간 호 길이, 장애물 비용은 영향거리 `influence` 안에서 거리 부족분을 간선 길이에 따라 적분한 값, 부드러움은 최종 경로를 같은 호 길이 간격으로 다시 샘플링한 이차 차분 비용이다. 장애물 거리는 PyBullet 충돌 장면의 링크–장애물 거리다. 연결된 경로는 모든 간선을 다시 검사한 뒤에만 성공 처리한다.

| 설정 | 기본값 | 역할 |
|---|---:|---|
| `obstacle_weight` | 5 | 가까운 장애물 비용 |
| `length_weight` | 1 | 탐색 간선 및 최종 경로 길이 비용 |
| `smooth_weight` | 0.01 | 완성된 경로의 굴곡 비용 |
| `success_reward` | 100 | 충돌 없는 목표 연결 보상 |
| `collision_penalty` | 2 | 충돌로 거절된 확장 비용 |
| `time_penalty` | 0.05 | 매 탐색 스텝 비용 |
| `influence` | 0.3 m | 장애물 영향거리 |
| `range_min`, `range_max` | 0.02, 0.4 | 액션으로 정할 관절 범위 비율 |

현재 학습 보상은 탐색 중 수락된 간선의 길이·장애물 비용과, 성공 시 완성된 경로의 길이·장애물·부드러움 비용을 합산한다. 따라서 완성 경로 비용은 성공 시 추가로 반영된다. 가중치가 다른 실험의 보상 총합을 직접 비교하지 말고 평가의 도달률, 경로 길이, 부드러움, 장애물 비용을 비교한다.

기존 `debug/rrt_range_20261102/prepared/dataset.json`은 현재 DDA·RT URDF와 해시가 달라 사용할 수 없다. 새 데이터셋을 준비한다. 스냅샷은 예시 경로를 실제 파일로 바꿔도 된다.

```bash
PYTHONPATH=python /home/dev/miniforge/envs/flame/bin/python python/rl/prepare_rrt_range_data.py \
  --snapshot debug/rrt_range_20261102/prepared/snapshot.pkl \
  --out debug/rrt_range_current/prepared
```

준비 스크립트는 검사점 단위로 학습/평가를 약 7:3으로 분리하고 현재 URDF에서 목표 IK와 양 끝 충돌 여부를 확인한다. `prepared/status.json`의 `state`가 `complete`인지 확인한 다음, 먼저 실제 장면에서 5스텝 스모크 테스트를 한다.

```bash
PYTHONPATH=python /home/dev/miniforge/envs/flame/bin/python python/rl/run_rrt_range_experiment.py \
  --dataset debug/rrt_range_current/prepared/dataset.json \
  --config python/rl/rrt_range_cost.json --out debug/rrt_range_current/smoke --smoke
```

`rrt_range_cost.json`을 복제해 비용 비율을 바꾸고 각 실험의 `--out`을 다르게 지정한다. 다음 명령은 학습 후 홀드아웃 대상에서 같은 시드의 고정 범위 RRT 탐색과 학습 범위 탐색을 평가한다.

```bash
PYTHONPATH=python /home/dev/miniforge/envs/flame/bin/python python/rl/run_rrt_range_experiment.py \
  --dataset debug/rrt_range_current/prepared/dataset.json \
  --config python/rl/rrt_range_cost.json --out debug/rrt_range_current/baseline \
  --steps 10000
```

결과는 `config.json`, `convergence.jsonl`, `evaluation.json`, `summary.md`에 기록된다. 장애물·길이·부드러움 가중치를 바꿀 때 탐색 확장 허용량(`max_steps`)과 데이터셋은 동일하게 유지한다.
