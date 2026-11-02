# SimTool 학습 모델 추론

실행: `make run PYTHON=/home/dev/miniforge/envs/flame/bin/python`.
이미 Viewer/ZProxy가 실행 중이면 SimTool만 재시작한다.

1. SimTool Operation 영역의 **SAC Inference…** 클릭.
2. 학습 모델 `.zip` 선택. 같은 실행 폴더의 `config.json`은 자동 선택된다.
   체크포인트 폴더의 모델도 선택 가능하다.
3. 저장된 IK 목표가 포함된 스냅샷 선택(예: `sample/planning_snapshot_p3_6.apfgoal.pkl`).
4. 기준 경로 폴더 선택(`summary.csv`와 목표별 `joint_states.csv` 필요).
   p1/p2/p3 스냅샷은 해당 `runs_apfgoal/RRTConnect_none_pN`을 자동 제안한다.
5. 목표 그룹 이름을 비우면 호환되는 전체 목표, 입력하면 해당 그룹만 추론한다.
6. Shield와 CPU/CUDA 선택 후 **추론 실행**. 별도 프로세스에서 실행하며 **중지** 가능.
7. 완료 후 **결과 불러오기**, 메인 화면의 **Start Simulation**으로 재생.

현재 모델은 residual SAC이므로 저장된 기준 경로가 필요하다. GUI의 현재 장면을 즉시 새로
계획하는 기능이 아니라 선택한 저장 스냅샷을 이용한 추론이며, 재생 시 그 장면을 복원한다.
기준 경로 끝점과 저장 IK 목표가 맞지 않는 목표는 제외하고 개수를 표시한다.
모델/학습 설정/로봇을 일치시켜야 한다. 모델 입출력 차원 불일치는 실행 오류로 표시된다.

학습용 목표 분할·장애물 증강·시작점 교란은 추론 시 해제된다. 모델은 결정론적으로 실행하고
각 결과의 시작 상태와 모든 edge를 충돌 엔진으로 재검증한다. 실패 경로는 실패 상태를 유지한
채 부분 경로로 저장되므로 성공 경로와 구분해서 확인할 수 있다. 실제 로봇 명령은 보내지 않는다.

출력: `debug/rl_inference/<timestamp>/`. 표준 `summary.csv`, `run.json`,
`failures.json`, `scene_snapshot.pkl`, 목표별 관절 CSV와 `inference.json`을 저장한다.
다른 로봇의 기존 경로를 SAC 결과에 섞지 않고 해당 모델의 로봇만 저장한다.
일반 **Load Result…**에서도 같은 폴더를 다시 불러올 수 있다.
