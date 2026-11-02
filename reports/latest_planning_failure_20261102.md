# 최신 경로 계획·재생 충돌 점검

대상: debug/plan_runs/20261102_214401_RRTConnect. 저장 집계는 40/40 성공, 계획 실패 없음. 재생에서 두 충돌이 별도로 기록됨.

1. 21:45:34, Point 4 - Inspection pose 1, DDA: segment 1(0부터 시작)의 dda_link_end와 pipe 충돌. 저장된 장면/경로에서 선형 보간 201점 재검사로 15점 충돌 재현.
2. 21:46:41, RT: Point 5 - Inspection pose 3이라는 스크린샷 이름이지만 실제로는 회전 전 복귀 동작. 이전 Point 1 - Inspection pose 1 종료 자세에서 linear track을 유지하고 다른 관절을 0으로 접는 두 점 직선 보간에서 201점 중 2점 충돌 재현. 회전 후 해당 검사 자세의 RT 첫 경로 구간은 같은 검사에서 충돌 0점.

최종 검증의 tight_verification_resolution은 planner.pin_collision_sample_resolution만 0.02로 바꾼다. 실제 collision_pairs_along_edge는 backend.check_edge_collision에 위임하며, backend는 handle.sample_resolution을 사용한다. 따라서 최종 검증 간격 변경이 실제 검사에 반영되지 않는 경로가 있다.

복귀 애니메이션 _build_retreat_animation_plans는 [current_q, target_q]를 직접 생성하며 해당 경로의 충돌 회피 계획/검증을 하지 않는다. 이 복귀 동작은 저장된 40개 계획의 성공 집계에 포함되지 않는다.

이전 21:26:39 실패 실행은 goal_collision 1개(8번 점 자세1 DDA, 51.04초)와 사용자 중단 45개였다. 46개 모두가 계획 알고리즘 실패는 아니다.

재현 결과: debug/latest_failure_review/recheck.json. 이번 점검에서 운영 코드는 변경하지 않음.
