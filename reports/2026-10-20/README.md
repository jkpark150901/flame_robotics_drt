# 경로 추종 성능 분석 문서 모음

정리일: 2026-10-20. 원본 문서는 수정하지 않고 복사했습니다.

[통합 PDF](all_reports.pdf)

| 문서 | 통합 PDF 쪽 | 범위 |
|---|---|---|
| [추종 성능 확인 요약 · 2페이지](sources/rb_servo_gui/docs/report/tracking_summary_2p.pdf) | 2–3 | 2026-10-20 / 평가 방법과 주요 결과 |
| [궤적 추종 성능 평가 전체 보고서](sources/rb_servo_gui/docs/report/report.pdf) | 4–34 | 10/08 방식 비교 및 10/14 servo 프로파일 실험 |
| [웨이포인트 분할·속도·경유점 도달 오차](sources/rb_servo_test/jb2_gui/report/report.pdf) | 35–46 | 10/16 jb2·servo 비교 |
| [동일 조건 반복 기록·오차 방향 분석](sources/rb_servo_gui/docs/report/repeatability/README.pdf) | 47–49 | 2026-10-20 / 기존 기록의 예비 분석 |
| [평가 지표 해설](sources/rb_servo_gui/docs/metrics.pdf) | 50–71 | 산출식·그래프 / 일부 DTW 설명은 이전 방식 |
| [외부 스크립트 인터페이스 정의서](sources/rb_servo_gui/docs/interface_spec.pdf) | 72–79 | 통신·명령 인터페이스 참고 |
| [결과 데이터 명세서](sources/rb_servo_gui/docs/report/RESULTS_SPEC.pdf) | 80–85 | 결과 CSV·궤적 CSV 정의 |
| [실험 데이터 명세서](sources/rb_servo_gui/docs/report/DATA_SPEC.pdf) | 86–97 | 원시 실험 데이터 구성 정의 |

## 포함 범위 및 주의

- 보고서 간 분석 코드·오차 정의가 다릅니다. 특히 10/08 DTW 매칭쌍 오차와 10/14 최단 경로 거리를 직접 비교하거나 개선 배율로 해석하지 않습니다.
- 반복성 분석은 저장된 실기 2회에서 확인한 예비 결과이며, 통계적 반복 정밀도나 작업공간 오차 방향을 확정하지 않습니다.
- sources에는 PDF·Markdown·HTML·그림·이미 포함된 집계 CSV/JSON 및 생성 스크립트를 보존했습니다. 외부 실행 폴더의 전체 원시 측정 데이터는 포함하지 않았습니다.
- 데이터 명세서의 results/, trajectories/, runs/ 등은 별도 데이터 패키지 설명입니다. 이 모음에 해당 전체 데이터가 포함된다는 뜻은 아닙니다.

문서 내부의 이미지 경로를 유지했습니다. `smoothness.md`는 metrics.md로 통합된 안내문으로, 소스에만 보존했습니다.
