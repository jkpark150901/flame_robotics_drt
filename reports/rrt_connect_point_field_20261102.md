# 점군 거리장 기반 IK + RRT Connect 테스트

입력: sample/planning_snapshot_p3_10(90).pkl. 기존 Random 5 Points 함수, seed=20261102로 검사점 10, 2, 7, 6, 8 선택. 12개 자세쌍, DDA/RT 합계 24개 목표.

RRTConnect, step_size=0.1, max_iter=8000, planning_timeout=180초. collision_free IK의 격자 점수 및 널스페이스 보정 사용. 경로 최적화 없음.

파이프 정점을 점군으로 복셀화한 후 Euclidean distance transform으로 3cm 격자 생성. 원본 점군이 없는 구형 스냅샷이므로 메쉬 정점 164,792개 사용. 삼각형 거리 조회 없음. 새로운 로드 1/10 설정은 구형 스냅샷에 소급 적용되지 않음. 새 스냅샷은 필드용 점군도 저장.

포지셔너/가상벽은 큰 평면을 유지하기 위해 기존 메쉬 필드 사용. 파이프 삼각형 317,094개를 사용하는 PyBullet 충돌 검증은 유지. unsigned 근사 거리장이므로 내부 판정과 스캔 누락을 해결하는 방식은 아님.

첫 필드 전체 빌드: DDA/RT 각각 약 1.3초. 모든 격자에서 KD-tree로 검색한 초기 구현은 약 24초로, EDT 방식으로 대체함.

점군 거리, 격자 외부 cap, 좌표 평행이동, 잘못된 입력 검증: pytest 5개 통과.

전체 24개 목표 실행 완료. 목표별 결과는 debug/rrt_connect_voxel_random5/target_times.jsonl에 즉시 저장. 원시 프로파일은 target_XX.prof. cProfile 계측 부하 포함. 헤드리스 RobotCore 테스트이므로 GUI 렌더링/통신과 실제 포지셔너 동작 시간은 제외.

## 자세 결정 충돌 기준 통일

자세 결정(Worker/Viewer/증강)은 PyBulletURDFLinkCollisionChecker로 동일 URDF의 말단 링크를 로드하고 검사한다. 별도 메쉬 충돌체 생성 대신 URDF 충돌 origin, scale, primitive/compound 형상 및 importer 규칙을 그대로 사용한다. 검사 거리는 경로 계획의 말단 대 장애물과 동일한 0m이며, 초기화 실패 시 점군 근접 검사로 우회하지 않는다. 자기 충돌/다른 팔 링크의 충돌은 여전히 IK/경로 계획의 전체 로봇 검사에서 확인해야 한다.

실제 DDA/RT 자세 10개에서 경로 계획의 end-link 판정과 일치. 별도 primitive URDF의 관성 오프셋, 복수 충돌 형상, TCP 오프셋 테스트 통과. 원시 결과: debug/rrt_connect_voxel_random5/pose_consistency.json. 구형 스냅샷의 저장된 목표 자세는 자동 변경하지 않으며 자세 재결정이 필요하다.

## 전체 결과

24개 중 1개 성공, 22개 목표 말단 충돌, 1개 탐색 상한 도달(timeout_approximate). 목표 충돌은 경로 탐색 이전에 거부됨. 이 결과는 구형 스냅샷 자세를 그대로 테스트한 것으로 새 자세 결정기로 재생성한 자세의 성공률이 아님.

전체 요청 시간 합계: 1295.8초. 첫 4개 동일 목표 비교: 직접 거리 762.1초 → 점군 복셀 212.7초.

|번호|목표|전체 요청(초)|결과|
|---:|---|---:|---|
|0|Point 8 - Inspection pose 1:DDA|59.37|목표 충돌|
|1|Point 8 - Inspection pose 1:RT|53.58|목표 충돌|
|2|Point 7 - Inspection pose 1:DDA|50.09|목표 충돌|
|3|Point 7 - Inspection pose 1:RT|49.70|목표 충돌|
|4|Point 6 - Inspection pose 1:DDA|54.05|목표 충돌|
|5|Point 6 - Inspection pose 1:RT|15.44|성공|
|6|Point 2 - Inspection pose 2:DDA|52.02|목표 충돌|
|7|Point 2 - Inspection pose 2:RT|50.60|목표 충돌|
|8|Point 2 - Inspection pose 1:DDA|53.85|목표 충돌|
|9|Point 2 - Inspection pose 1:RT|38.37|목표 충돌|
|10|Point 6 - Inspection pose 2:DDA|57.91|목표 충돌|
|11|Point 6 - Inspection pose 2:RT|55.10|목표 충돌|
|12|Point 7 - Inspection pose 3:DDA|59.05|목표 충돌|
|13|Point 7 - Inspection pose 3:RT|57.48|목표 충돌|
|14|Point 7 - Inspection pose 2:DDA|58.22|목표 충돌|
|15|Point 7 - Inspection pose 2:RT|58.29|목표 충돌|
|16|Point 8 - Inspection pose 3:DDA|60.49|목표 충돌|
|17|Point 8 - Inspection pose 3:RT|57.88|목표 충돌|
|18|Point 8 - Inspection pose 2:DDA|60.65|목표 충돌|
|19|Point 8 - Inspection pose 2:RT|57.23|목표 충돌|
|20|Point 10 - Inspection pose 2:DDA|57.48|목표 충돌|
|21|Point 10 - Inspection pose 2:RT|63.14|탐색 상한|
|22|Point 10 - Inspection pose 1:DDA|58.60|목표 충돌|
|23|Point 10 - Inspection pose 1:RT|57.21|목표 충돌|

## 추가 확인된 실제 경로 불일치 원인

GUI의 _inspection_pose_process_snapshot이 파이프/포지셔너 충돌 메쉬 없이 점군만 Worker에 전달했다. Worker의 pose service는 장애물 메쉬가 없으면 PyBullet checker를 생성하지 않고 점군 근접 검사로 내려갔다. 요청 snapshot에 경로 계획에서 사용하는 world-frame mesh vertices/triangles를 추가하고 점군 fallback을 제거했다. 이 전송 경로와 optimizer 내부 메쉬 복사 참조 회귀 테스트 포함 3개 테스트 통과.

## 자세 재결정 스모크 테스트

기존 8번 검사점 위치에서 자세 재결정: 4.79초, 3개 DDA/RT 자세쌍 생성. 구형 스냅샷에 원본 점군이 없어 스냅샷 메쉬 표면에서 20만 점을 샘플링해 프로파일 입력으로 사용했으며, 충돌 장애물은 원본 메쉬 그대로 유지했다. 출력된 6개 말단 자세를 별도 URDF checker로 다시 검사했고 모두 비충돌. 전체 팔 IK/경로 성공을 의미하지는 않는다. 원시 결과는 pose_regeneration.json 및 regenerated_pose_checks.json.
