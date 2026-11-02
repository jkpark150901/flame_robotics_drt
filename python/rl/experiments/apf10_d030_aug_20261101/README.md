# APF 범위·가중치 확대 + 스냅샷 내부 분할 + 장애물 증강

- 기존 residual SAC DDA 실험을 확장. APF 가중치 5 → 10, 영향 거리 0.15 → 0.30 m,
  비용 clip 50 → 200. 스텝당 APF 페널티 상한은 250 → 2000. 충돌 페널티 5000 유지.
- 시작/목표의 불가피한 APF 비용 보정 유지. 보정 범위 sigma=0.15는 관절 범위로 정규화한
  가중 관절 거리이며, 장애물 영향 거리(m)와 다른 값이다.
- 분할은 이전 `split_stratified_dda.json`을 복사하여 고정. 동일 목표의 모든 기준 경로와
  증강 결과는 같은 split에 남는다. 학습 데이터에 평가 목표의 rollout을 넣지 않는다.

| 스냅샷 | 학습 목표 | 평가 목표 |
|---|---:|---:|
| p1 | 4 | 2 |
| p2 | 5 | 2 |
| p3 | 15 | 3 |

사용 가능한 저장 IK 및 기준 경로가 있는 DDA 목표 31개 기준이다. 스냅샷 자체가 미관측인
평가가 아니라, 각 스냅샷 내 미학습 목표 평가이다. 기준 경로는 평가에도 입력된다.

## 증강

- 각 장면/회전 phase에서 에피소드마다 80% 확률로 1–3개 상자 배치 시도.
- 크기 축별 5–15 cm, 링크 주변 offset 10–35 cm. 시작/목표와 충돌하는 상자는 거부.
- 기준 경로를 막는 상자와 20 cm 이내 근접 상자를 혼합(block_prob=0.3).
- 배치 거부로 실제 장애물 수는 요청 수보다 적거나 0일 수 있다. 실제 개수를 기록한다.
- 기존 시작점 중간 선택(30%), 시작 관절 잡음(30%), 다양한 플래너 기준 경로 선택 유지.
- 거부된 상자의 tuple identity 비교 때문에 삭제 후에도 목록에 남던 버그 수정.

## 실행과 평가

`CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/dev/miniforge/envs/flame/bin/python -u python/rl/experiments/apf10_d030_aug_20261101/run.py`

- SAC 300,000 steps, 16 environments. 별도 출력 폴더 존재 시 재실행을 거부한다.
- 학습 후 자동 평가: 원본 장면 1 seed 및 증강 장면 3 seeds.
- 각 조건에서 SAC / SAC+shield / zero-policy 기준 경로 추종 비교.
- 평가 seed 810000, 910000, 920000, 930000에 target index × 1000을 더한다.
  학습 난수 seed와 분리되며, 정책/Shield 간 같은 목표에는 같은 장애물을 사용한다.
- 증강 평가는 배치 확률 100%로 시도하되, 배치 성공 여부를 `n_boxes`와 실제 center/size로 기록.
- 깨끗한 장면에서만 기존 RRTConnect/CHOMP/STOMP 벤치마크를 함께 표시한다.
  증강 장면에 기존 원본 장면 벤치마크 숫자를 섞지 않는다.
- 계산 시간은 rollout만 포함하며, 기준 경로 생성 및 학습 시간은 제외한다.
- 성공률과 `C_eff >= 0 && mandatory_ok` 경로 안전을 별도로 보고한다.
- 출력: `debug/rl/dda_apf10_d030_split_aug_20261101/`의 `status.json`, `train.log`,
  `eval_*.json`, `summary.md`. 평가 중에도 `.partial.json`에 목표별 결과 저장.
- 설정·소스 사본, 스냅샷/소스 해시, 실제 실행 명령은 결과 폴더에 보존한다.

원격 실제 로봇 명령은 보내지 않는다. 기존 학습 결과는 덮어쓰지 않는다.
