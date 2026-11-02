"""모든 설정값(연결/궤적/모션 파라미터/판정 기준)을 한 곳에서 관리한다. JSON 으로 저장/불러오기."""
import json
import os
from dataclasses import asdict, dataclass, field, fields

DEFAULT_OUT_DIR = os.path.join(os.path.expanduser("~"), "rb_servo_gui_runs")
LAST_SETTINGS = os.path.join(os.path.expanduser("~"), ".rb_servo_gui.json")


@dataclass
class Settings:
    # ---- 연결 ----
    ip: str = "10.0.2.7"                 # 제어박스 IP (LCD 표시값)
    use_mock: bool = False               # True 면 내장 목업 로봇(127.0.0.1)에 연결 (로봇 없이 테스트)
    robot_mode: str = "simulation"       # "simulation" | "real" (real 은 저장되지 않으며 실행마다 확인 절차 필요)
    real_ack: bool = False               # 실기 실행 확인(GUI 확인 절차 통과 시에만 True). 저장되지 않음
    real_speed_cap: float = 30.0         # real 모드 속도 상한 [deg/s] (speed/home_speed 에 적용, 가속도는 2배까지). 0 이면 제한 없음
    revert_to_sim: bool = True           # real 실행이 끝나면 simulation 모드로 되돌림

    # ---- 궤적 ----
    traj_path: str = ""                  # 입력 궤적 CSV
    traj_kind: str = "waypoint"          # "waypoint": 웨이포인트 CSV(보간 필요) | "interp": 보간된 CSV (t, j0..j5 [rad])
    vmax: float = 20.0                   # 웨이포인트 시간 부여용 최대 관절속도 [deg/s]
    interp_rate: float = 50.0            # 보간 주파수 [Hz]

    # ---- 링크 표면 오차 (URDF 가 있을 때만) ----
    urdf_path: str = ""                  # 로봇 URDF. 지정하면 각도 오차가 링크 표면 위치 차이(mm)로 얼마나 나타나는지 계산
    surface_points: int = 600            # 링크당 표면 샘플점 수
    surface_samples: int = 200           # 추종 구간에서 비교할 매칭쌍 수 (균등 추출)

    # ---- 실행 ----
    mode: str = "jb2"                    # "jb2" | "servo" | "both"
    step: int = 10                       # 기본 추출 간격 (극값·정지 구간 경계는 추가 보존)
    out_dir: str = DEFAULT_OUT_DIR       # 실행마다 이 아래에 폴더가 생긴다
    home_first: bool = False             # 시작점 이동 전에 원점 복귀를 먼저 할지
    home_joints: list = field(default_factory=lambda: [0.0] * 6)
    home_speed: float = 20.0             # deg/s
    home_accel: float = 40.0             # deg/s^2

    # ---- move_j(시작 자세 이동) + move_jb2_add ----
    speed: float = 30.0                  # deg/s
    accel: float = 60.0                  # deg/s^2
    blend: float = 50.0                  # jb2 블렌드율 [%]

    # ---- move_servo_j ----
    servo_timing: str = "profile"        # "profile": 원본 궤적 위에서 speed/accel(관절 속도·가속도 한계) 이상 프로파일을 servo_period 주기로 샘플링해 전송 | "original": 지령점을 원본 궤적 시간 간격대로
    servo_basis: str = "max_joint"       # 프로파일 한계 기준: "max_joint"(관절마다 |속도| ≤ speed, |가속도| ≤ accel) | "l2"(6관절 벡터 크기)
    servo_period: float = 0.02           # profile 타이밍의 servo 전송 주기 [s] (0.002~0.2, 끝을 정확히 맞추려고 이 값 이하로 균등 분할)
    t1: float = 0.0                      # 도달 시간 [s] (0 이면 전송 간격, 최소 0.002)
    t2: float = 0.0                      # 유지 시간 [s] (0 이면 profile: 전송 주기의 2배, original: 전송 간격. 0.02~0.2 로 제한)
    gain: float = 0.1
    alpha: float = 0.5
    final_t2: float = 0.0                # servo 마지막 명령(과 반복 명령)에만 쓰는 t2 [s]. 0 이면 t2 와 같음, 최소 0.02 (오버슈트 = 마지막 속도 × t2 억제)
    final_hold: int = 2                  # servo 마지막 점을 이 횟수만큼 더 보냄 (마지막 지령 속도를 0으로 만들어 종료 오버슈트 억제, 0 이면 안 보냄)

    # ---- 대기/기록 ----
    reach_tol: float = 0.5               # 목표 도달 판정 허용 오차 [deg]
    reach_hold: float = 0.3              # 허용 오차 안에서 머무는 시간 [s]
    timeout: float = 20.0                # 움직임 없이 이 시간이 지나면 포기 [s] (움직이는 동안은 계속 기다림)
    settle_timeout: float = 15.0         # servo 마지막 지령 후 수렴 대기 포기 시간 [s]
    max_wait: float = 1800.0             # 한 번의 대기 절대 상한 [s]
    record_hz: float = 50.0              # 상태 기록 주파수 [Hz]

    # ---- 추종 성능 판정 기준 ----
    extrema_prominence: float = 0.05    # 극값의 최소 돌출도 [deg], 측정 노이즈 제외
    extrema_max_shift: float = 0.35     # 극값 대응의 정규화 시간 차이 상한 (0~1)
    th_path_p95: float = 1.0             # 궤적 추종 오차(DTW 매칭쌍 L2) p95 [deg]
    th_path_max: float = 2.0             # 궤적 추종 오차(DTW 매칭쌍 L2) 최대 [deg]
    th_cover: float = 2.0                # 커버리지(원본점->측정 경로) 최대 [deg]
    th_progress: float = 0.99            # 경로 진행도
    th_step: float = 5.0                 # 연속 측정 샘플 간 최대 이동 [deg]
    th_end: float = 0.5                  # 시작/종료 각 오차 [deg]  (종료 각은 마지막 안정 구간 기준, reach_tol 과 무관)
    th_ripple: float = 0.15              # 속도 리플(경로 따라 속력의 고주파 흔들림 / 평균 속력) 상한
    th_wobble: float = 0.30              # 횡방향 흔들림(궤적 추종 오차의 고주파 RMS) 상한 [deg]
    th_stall: float = 0.02               # 경로 중간 정체 시간 비율 상한 (속력 < 중앙값의 20%)
    th_lag: float = 0.03                 # 타이밍 지연 표시 기준 (정규화 시간 비율, 판정에는 미사용)
    move_thr: float = 0.2                # 모션 시작 판정 변화량 [deg]

    def to_dict(self):
        return asdict(self)

    def save(self, path=LAST_SETTINGS):
        """real 관련 값은 저장하지 않는다 (불러오면 항상 simulation 으로 시작)."""
        d = self.to_dict()
        d["robot_mode"] = "simulation"
        d["real_ack"] = False
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path=LAST_SETTINGS):
        s = cls()
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            return s
        names = {f_.name for f_ in fields(cls)}
        for k, v in d.items():
            if k in names:
                setattr(s, k, v)
        s.robot_mode, s.real_ack = "simulation", False
        return s
