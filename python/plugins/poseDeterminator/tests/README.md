# plugin_pdm tests

`EndEffectorPoseOptimizer` 및 부속 함수의 회귀 방지·강건성 검증 테스트.

## 디렉토리 구성

| 디렉토리 | 목적 | 외부 데이터 |
|----------|------|-------------|
| `unit/` | 단위 테스트. 외부 파일 의존 없이 합성 입력으로 메서드 단위 검증 | 없음 |
| `regression/` | 회귀 테스트. 현장 데이터에서 발견된 버그의 재현·방지 | `../data/` 의 PLY/CSV (있을 때만 실행, 부재 시 skip) |

향후 추가 예정 카테고리:
- `integration/` — 데모 노트북 핵심 흐름의 자동화
- `fixtures/` — 공용 픽스처/합성 데이터 생성기

## 실행

```bash
pip install pytest pandas
cd python/plugins/plugin_pdm
pytest tests/
```

카테고리별:
```bash
pytest tests/unit            # 빠르고 항상 실행 가능
pytest tests/regression      # 데이터가 있어야 의미 있음
```

특정 테스트만:
```bash
pytest tests/unit/test_pipe_direction_axis_robustness.py -v
```

## 작성 가이드

- **단위 테스트**: 외부 파일/네트워크 의존 금지. mock 또는 합성 데이터로만 동작.
- **회귀 테스트**: 데이터 부재 시 `pytest.skip` 사용. 무조건 fail 금지.
- 새 카테고리 추가 시 본 README 표에 한 줄 추가.
