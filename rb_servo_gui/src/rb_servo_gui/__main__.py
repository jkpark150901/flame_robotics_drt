"""실행: python -m rb_servo_gui [--selftest]"""
import argparse
import os
import sys
import tempfile
import threading


def _selftest():
    """로봇 없이 전체 흐름(목업 서버 -> 실행 -> 분석) 과 안전장치를 점검한다. 모든 통신은 127.0.0.1."""
    import numpy as np

    from . import trajectory
    from .mock_robot import MockRobot
    from .robot import NotSimulation
    from .runner import Runner
    from .settings import Settings

    tmp = tempfile.mkdtemp(prefix="rb_servo_selftest_")
    # 합성 웨이포인트 (6관절, rad)
    tt = np.linspace(0, 1, 12)
    wp = np.column_stack([np.arange(12)] + [0.6 * np.sin(2 * np.pi * tt * (k + 1) / 2) for k in range(6)])
    wp_path = os.path.join(tmp, "waypoints.csv")
    np.savetxt(wp_path, wp, delimiter=",", header="waypoint," + ",".join(f"j{i}" for i in range(6)), comments="", fmt="%.6f")

    mock = MockRobot(vmax=150.0).start()
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    try:
        base = dict(use_mock=True, traj_path=wp_path, traj_kind="waypoint", vmax=60.0, step=6, mode="both",
                    out_dir=tmp, speed=120, accel=240, blend=50, timeout=5, settle_timeout=5, record_hz=50)
        r = Runner(Settings(**base), log=lambda m: None)
        d = r.run()
        res = r.results or {}
        check("simulation: jb2/servo 실행 + 분석 완료", set(res) == {"jb2", "servo"})
        check("simulation: 모든 지령이 simulation 모드에서 수행됨 ('pgmode real' 없음)", "pgmode real" not in mock.log)
        check("simulation: 추종 시간 기록", all(res[m]["metrics"]["t_track"] > 0 and res[m]["metrics"]["t_total"] > 0 for m in res))
        check("simulation: 결과 파일 생성", all(os.path.exists(os.path.join(d, f)) for f in ("config.json", "traj_ref.csv"))
              and any(f.startswith("sim_follow") for f in os.listdir(d)) and any(f.startswith("smooth_map_follow") for f in os.listdir(d))
              and any(f.endswith("_after.png") for f in os.listdir(d)) and sorted(f for f in os.listdir(d) if f.endswith(".png")) == sorted(f for f in os.listdir(d) if f.endswith(".png") and (f.startswith(("sim_follow", "smooth_map_follow")) or f.endswith("_after.png"))))

        # ---- 실기 모드 안전장치 (목업 서버로만 시험) ----
        try:
            Runner(Settings(**base, robot_mode="real", real_ack=False), log=lambda m: None).run()
            check("real: 확인(real_ack) 없이는 거부", False)
        except PermissionError:
            check("real: 확인(real_ack) 없이는 거부", True)
        check("real: 거부 시 'pgmode real' 미전송", "pgmode real" not in mock.log)

        mock.queue = [[10.0] * 6, [20.0] * 6]  # 이동 중인 상태
        try:
            Runner(Settings(**base, robot_mode="real", real_ack=True), log=lambda m: None).run()
            check("real: 로봇이 움직이는 중이면 시작 거부", False)
        except NotSimulation:
            check("real: 로봇이 움직이는 중이면 시작 거부", True)
        mock.queue = []
        check("real: 거부 시 'pgmode real' 미전송", "pgmode real" not in mock.log)

        logs = []
        r2 = Runner(Settings(**{**base, "mode": "jb2"}, robot_mode="real", real_ack=True, real_speed_cap=40), log=logs.append)
        d2 = r2.run()
        check("real: 확인 후 실행 + 분석 완료", bool(r2.results))
        check("real: 속도 상한 적용 (120 -> 40)", any("속도 상한" in m for m in logs) and r2.s.speed == 40)
        check("real: 종료 후 simulation 으로 복귀", mock.sim == 1 and mock.log[-1] == "pgmode simulation")
        check("real: 실행 폴더에 REAL 표시", "_REAL_" in os.path.basename(d2))

        # ---- 원점 복귀 단독 ----
        hs = dict(use_mock=True, home_joints=[0.0] * 6, home_speed=120, home_accel=240, timeout=5)
        mock.ref = [30.0, -20.0, 10.0, 40.0, -50.0, 5.0]
        mock.queue = []
        Runner(Settings(**hs), log=lambda m: None).home()
        check("home: simulation 에서 원점 도달", max(abs(v) for v in mock.ref) < 0.5)
        check("home: simulation 에서 'pgmode real' 미전송", "pgmode real" not in mock.log[-6:])
        mock.ref = [20.0] * 6
        n_before = len(mock.log)
        try:
            Runner(Settings(**hs, robot_mode="real", real_ack=False), log=lambda m: None).home()
            check("home: real 은 확인 없이는 거부", False)
        except PermissionError:
            check("home: real 은 확인 없이는 거부", True)
        check("home: 거부 시 명령 미전송", len(mock.log) == n_before)
        mock.queue = [[5.0] * 6]
        try:
            Runner(Settings(**hs, robot_mode="real", real_ack=True), log=lambda m: None).home()
            check("home: real 에서 움직이는 중이면 거부", False)
        except NotSimulation:
            check("home: real 에서 움직이는 중이면 거부", True)
        mock.queue = []
        mock.ref = [20.0] * 6
        rl = []
        Runner(Settings(**hs, robot_mode="real", real_ack=True, real_speed_cap=25), log=rl.append).home()
        check("home: real 확인 후 원점 도달 + 속도 상한 적용", max(abs(v) for v in mock.ref) < 0.5 and any("속도 상한" in m for m in rl))
        check("home: real 종료 후 simulation 복귀", mock.sim == 1 and mock.log[-1] == "pgmode simulation")
    finally:
        mock.stop()
    print("\nselftest", "OK" if ok else "FAILED", "| 결과 폴더:", tmp)
    return 0 if ok else 1


def _fix_tk_environment(argv):
    """tkinter 가 글꼴(Xft)을 못 읽는 Tk 를 불러오면(예: conda/ROS 의 LD_LIBRARY_PATH 가 시스템 Tk 대신 다른 libtk 를 잡는 경우)
    한글이 깨진다. 그런 경우 LD_LIBRARY_PATH 를 제거하고 한 번만 자기 자신을 다시 실행한다."""
    if os.environ.get("RB_SERVO_NO_REEXEC") or not os.environ.get("LD_LIBRARY_PATH"):
        return
    try:
        import tkinter
        import tkinter.font as tkfont
        root = tkinter.Tk()
        root.withdraw()
        n = len(set(tkfont.families(root)))
        root.destroy()
    except Exception:  # noqa: BLE001
        return
    if n >= 60:  # 정상 Tk(Xft) 는 시스템 글꼴을 모두 보여준다
        return
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env["RB_SERVO_NO_REEXEC"] = "1"
    print("[rb_servo_gui] 시스템 글꼴을 읽지 못하는 Tk 로 감지 -> LD_LIBRARY_PATH 없이 다시 시작합니다", file=sys.stderr)
    os.execve(sys.executable, [sys.executable, "-m", "rb_servo_gui", *argv], env)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="rb_servo_gui")
    ap.add_argument("--selftest", action="store_true", help="로봇 없이 목업 서버로 전체 흐름과 안전장치를 점검")
    ap.add_argument("--check-fonts", action="store_true", help="GUI 글꼴(한글) 설정 확인 후 종료")
    args = ap.parse_args(argv)
    if args.selftest:
        sys.exit(_selftest())
    _fix_tk_environment(argv)
    if args.check_fonts:
        import tkinter as tk
        from .gui import setup_fonts
        root = tk.Tk()
        root.withdraw()
        print(setup_fonts(root))
        root.destroy()
        return
    from .gui import run_gui
    run_gui()


if __name__ == "__main__":
    main()
