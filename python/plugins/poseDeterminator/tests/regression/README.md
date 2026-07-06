# Regression tests

현장에서 발견된 버그의 재현·방지 테스트. 실제 데이터에 의존하므로 데이터 파일이 없는 머신에서는 자동 skip 된다.

## 데이터 의존성

| 테스트 | 필요 파일 |
|--------|-----------|
| `test_calculate_dda_rt_pose_svd.py` | `../../data/PIPE NO.3_fill.ply`, `../../data/PIPE NO.3_fill_testpoint.csv`, `../../data/robot_models v1.5/robots/rb10_1300e_{DDA,RT}.urdf` |

`data/` 디렉토리는 `.gitignore` 되어 있다(개발자 단말 별도 관리). 새 머신에서 회귀 검증을 돌리려면 데이터를 따로 받아두어야 한다.

## 새 회귀 테스트 추가

1. 모듈 docstring 에 **버그 발생 경위·원인·해결 방향**을 기록.
2. 데이터 부재 시 `pytest.skip(...)` 처리.
3. 검증은 단순 "raise 안 함" 부터 시작 — 시간이 지나 더 정밀한 검증이 필요해지면 별도 단위 테스트로 분리.
