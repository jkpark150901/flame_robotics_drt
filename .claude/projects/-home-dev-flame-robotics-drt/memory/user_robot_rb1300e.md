---
name: user-robot-rb1300e
description: 사용자가 사용하는 로봇 모델 및 workspace 특성
metadata:
  type: user
---

Rainbow Robotics **RB1300e** 사용.

- 최대 도달거리: **1300mm** (어깨 조인트 기준)
- 현재 작업 TCP 추정: `[10, -258, 1330, 0.8, 0.0, 180.0]` (mm/deg)
- 원점 기준 현재 거리: ~1354mm → 이미 1300mm 공칭 도달 거리 초과
- 현재 방향: Rx≈0.8°, Ry=0°, **Rz=180°** (TCP가 기준 방향 반대 방향을 향함)

**Workspace 특성**:
- Z=1330mm는 최대 도달 한계에 매우 근접 → XY 이동이 조금만 커도 armstratch 발생
- 예제 코드 좌표(Z=200~400, Rz=0°)와 완전히 다른 영역에서 작업 중
- pb 궤적 설계 시 Z=1000~1200mm 대에서 작업하는 것이 안전
- 모든 TCP 좌표는 Rz=180을 유지해야 특이점 회피 가능
