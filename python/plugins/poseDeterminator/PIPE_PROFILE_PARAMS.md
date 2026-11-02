# 파이프 프로파일(원통) 피팅 파라미터

EF pose를 만들기 전에 검사점 주변의 파이프를 원통(축/중심/반지름)으로 피팅한다. 이 결과로 DDA 후보 위치
(`반지름 + distance_from_dda_to_surface + EF 앞면 길이`)가 정해지므로, **프로파일이 틀리면 포즈가 전부 틀린다.**

- 코드: [PipeProfileFitting.py](PipeProfileFitting.py) (파라미터/RANSAC/검증), [EndEffectorPoseOptimizer.py](EndEffectorPoseOptimizer.py) `calculate_pipe_profile`
- 설정: `viewervedo.cfg` → `ef_pose.pipe_profile` (아래 표). 예전 평면 키(`sampling_size_for_calculating_normal`,
  `radius_offset_for_sampling_points_in_sphere`)도 `pipe_profile` 값이 없을 때는 계속 인식한다.
- 단위: 길이 m, 각도 deg. 오타 키는 무시되고 Robot Core 로그에 `unknown key(s) ignored`로 표시된다.

## 방식 (`method`)

| 값 | 동작 | 비고 |
|---|---|---|
| `ransac` (기본) | 검사점 주변 점군에 RANSAC 원통 피팅(법선 2개로 축/반지름 가설) → 인라이어 최소제곱 재피팅 → 검증 | 법선 정확도에 의존하지 않음. 노이즈/이상치/이웃 형상에 강함 |
| `cluster` (legacy) | 법선 반대 방향 가는 ROI 실린더의 점을 클러스터링 → 2번 클러스터 끝점을 반대편 벽으로 가정 → 구 안의 점에 최소제곱 | 법선이 몇 도만 틀려도 반대편 벽을 놓쳐 mm 단위 가짜 원통을 만들 수 있음(검증이 걸러냄) |

## 파라미터

### 공통
| 키 | 기본 | 설명 | 조정 |
|---|---|---|---|
| `method` | `ransac` | 위 표 | |
| `normal_sampling_size` | 0.01 | 검사점 주변 이 크기의 상자 안 법선 중앙값을 로컬 법선으로 사용 | 스캔이 거칠면 키운다 (cluster 방식과 시각화 진단에 영향) |
| `validate` | true | 검증 실패 시 포즈 결정을 중단하고 사유를 반환 | false는 디버깅용. 틀린 프로파일로 포즈가 만들어진다 |

### RANSAC
| 키 | 기본 | 설명 | 증상 → 조정 |
|---|---|---|---|
| `seed` | 0 | RANSAC 난수 시드. `null`이면 매번 다름 | 결과가 실행마다 달라지면 고정 |
| `neighborhood_radius` | 0.25 | 피팅에 쓰는 검사점 주변 반경 | 파이프 **지름의 2배 이상** 권장. 작으면 원 둘레를 못 봐서 `angular coverage` 실패, 너무 크면 다른 형상이 섞여 속도 저하 |
| `max_neighborhood_points` | 30000 | RANSAC 점수 계산용 상한(균등 샘플) | 느리면 낮춘다 |
| `radius_min` / `radius_max` | 0.01 / 0.30 | 허용 파이프 반지름 범위. 가설과 검증 모두에 적용 | 현장 배관 규격에 맞춰 **좁힐수록** 오검출이 줄어든다 |
| `ransac_iterations` | 400 | 샘플 쌍 횟수 | 파이프가 이웃 점의 일부(<10%)뿐이면 늘린다 |
| `inlier_threshold` | 0.003 | `|축까지 거리 - 반지름|`이 이 값 미만이면 인라이어 | 스캔 노이즈 σ의 약 3배. 노이즈가 크면 키운다 |
| `normal_angle_deg` | 25 | 인라이어 법선과 반경 방향의 허용 각도 | 법선 추정이 거칠면 키운다 |
| `min_axis_angle_deg` | 8 | 두 샘플의 법선이 이보다 평행하면 건너뜀(평면 조각) | |
| `refine` | true | 인라이어로 최소제곱 재피팅(인라이어가 줄면 채택 안 함) | |

### 검증 (두 방식 공통)
| 키 | 기본 | 실패 사유 메시지 | 의미 / 조정 |
|---|---|---|---|
| `min_inliers` | 100 | `inliers N < min_inliers` | 피팅 점이 너무 적다. 점군이 성긴 구간이면 낮춘다 |
| `min_inlier_ratio` | 0.05 | `inlier ratio r < 0.05` | 이웃 점 중 원통에 붙는 비율. 직선 파이프가 아닌 곳(엘보/플랜지/밸브)에서 걸린다 |
| `min_angular_coverage_deg` | 90 | `angular coverage X deg < 90` | 인라이어가 축 둘레를 덮는 각도. 한쪽에서만 스캔한 파이프는 120~180°, 너무 작으면 반지름이 불안정 |
| `anchor_tolerance` | 0.01 | `surface point is X mm off the fitted surface` | 선택한 표면점이 피팅 표면에서 이만큼 이상 떨어지면 다른 형상을 피팅한 것 |
| `radius_min/max` | | `radius X mm outside [..]` | 반지름 범위 밖. mm 단위 값이면 거의 항상 잘못된 피팅 |
| `max_radius_mismatch` | 0.3 | `fit radius X disagrees with cluster estimate Y` | (cluster 방식) 최종 반지름과 클러스터 추정 반지름의 상대 차이 |

### legacy 클러스터 방식 전용
| 키 | 기본 | 설명 |
|---|---|---|
| `cluster_roi_radius` | 0.005 | 법선 반대 방향 ROI 실린더 반지름이자 클러스터 간격. **반대편 벽까지 거리(≈지름)×tan(법선 오차)보다 작으면 반대편 벽을 놓친다** (예: 지름 75mm, 법선 오차 4° → 5mm 초과) |
| `cluster_height_range` | [-0.1, 0.3] | 표면점에서 법선 반대 방향으로 ROI가 뻗는 범위. 파이프 지름보다 길어야 한다 |
| `sphere_radius_offset` | 0.003 | 추정 반지름에 더하는 재샘플링 구 여유 |

## 시각화 (자세 결정 후, 성공/실패 모두 표시)
`ef_pose.show_pipe_profile_debug`(기본 true)로 켜고 끈다.

| 표시 | 의미 |
|---|---|
| 초록 구 | 선택점에서 가장 가까운 스캔 표면점 |
| 옅은 회색 점 | (RANSAC) 피팅에 쓴 주변 점군 |
| 초록 점 | (RANSAC) 인라이어 |
| 회색 와이어 실린더 / 파랑·주황·보라 점 | 클러스터 ROI와 클러스터 (파랑=표면 쪽 0번, 주황=반대편으로 가정한 1번). 피팅이 틀린 이유 진단용 |
| 노란 구 | (cluster) 추정 중심/반지름으로 만든 재샘플링 구 |
| 청록 반투명 실린더 | 최종 피팅. **검증에 실패하면 빨강** |
| 3D 라벨 | `P<번호> [방식] r=반지름 inl=인라이어/이웃점`, 실패 시 `REJECTED: 사유` |

DDA 포즈의 불투명도는 `ef_pose.dda_alpha`(기본 0.6), 충돌로 탈락한 후보 표시는 `ef_pose.show_rejected_candidates`.

## 튜닝 순서
1. 반지름 범위(`radius_min/max`)를 현장 배관에 맞춘다 (가장 효과가 크다).
2. 라벨의 `inl`이 낮고 초록 점이 파이프 일부에만 있으면 `inlier_threshold`/`normal_angle_deg`를 키운다.
3. `angular coverage` 실패면 `neighborhood_radius`를 지름의 2배 이상으로 키운다.
4. 그래도 안 되면 그 지점이 직선 파이프가 아닌지(엘보/플랜지/밸브) 본다. 그곳은 거부되는 것이 정상이다.
