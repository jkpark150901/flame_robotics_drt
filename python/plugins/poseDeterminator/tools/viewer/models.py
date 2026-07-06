"""tools/viewer 의 도메인 dataclass.

UI / worker / signal 이 공유하는 단일 자료형 정의. ``InspectionResult`` 는
``BatchWorker`` 가 emit 직전 deepcopy 해서 메인 스레드로 전달한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class PointStatus(Enum):
    UNRUN = "unrun"
    SUCCESS = "success"
    FAILURE = "failure"


class OptimizationMode(Enum):
    """엔드이펙터 자세 탐지 모드.

    - ``TWO_PAIR_90``: 기존 ``calculate_DDA_RT_pose_for_taking_xray`` (90°·2쌍).
      pose_group 의 키는 ``"0"`` / ``"90"``.
    - ``THREE_PAIR_120``: 신규 ``calculate_DDA_RT_pose_for_taking_xray_3pair_120``
      (120°·3쌍, 2쌍 폴백 허용). pose_group 의 키는 3쌍이면 ``"0"`` /
      ``"120"`` / ``"240"``, 폴백이면 ``"0"`` / ``"120"``.

    뷰어의 ``scene_builder.render_result`` 는 두 키 집합 모두를 ``_KEY_TO_COLOR``
    에서 매핑하므로, 모드는 batch worker 의 알고리즘 선택만 분기시키면 된다.
    """

    TWO_PAIR_90 = "2pair_90"
    THREE_PAIR_120 = "3pair_120"


@dataclass(frozen=True)
class PipeData:
    pipe_id: str  # 예: "PIPE NO.1"
    ply_path: Path
    csv_path: Path


@dataclass(frozen=True)
class InspectionPoint:
    index: int  # CSV 행 인덱스 (1-base, CSV 첫 열 그대로)
    position: tuple[float, float, float]  # (x, y, z)
    offset: tuple[float, float, float]  # (dx, dy, dz), 누락 시 (0,0,0)

    @property
    def position_with_offset(self) -> tuple[float, float, float]:
        return (
            self.position[0] + self.offset[0],
            self.position[1] + self.offset[1],
            self.position[2] + self.offset[2],
        )


@dataclass
class InspectionResult:
    point_index: int
    success: bool
    pose_groups: list[dict] = field(default_factory=list)
    debuging_info: dict | None = None
    error_message: str | None = None
    # 이 결과를 생성한 최적화 모드. 기본값은 기존 90°·2쌍 동작을 보존
    # (필드 미지정 호출자는 자동으로 TWO_PAIR_90 으로 인식됨).
    mode: OptimizationMode = OptimizationMode.TWO_PAIR_90
