#!/usr/bin/env python3
"""
trajectory_blend_test.py
========================
동일한 waypoints 궤적을 여러 blending 값으로 실행하고
실제 시계열 데이터를 기록해 비교 플롯을 생성합니다.

모드:
  jb2  move_jb2_add/run  관절 공간 blending  (blending_value 0.0~1.0)
  pb   move_pb_add/run   TCP 공간 path blending  (ratio 0.0~1.0 또는 distance mm)
  lb   move_lb_add/run   TCP 공간 linear blending  (blend_distance mm)

사용 예:
  python trajectory_blend_test.py --robot_ip 10.0.2.7
  python trajectory_blend_test.py --robot_ip 10.0.2.7 --mode jb2 --blending 0.0 0.3 0.7 1.0
  python trajectory_blend_test.py --robot_ip 10.0.2.7 --mode pb  --blending 0.05 0.3 0.7
  python trajectory_blend_test.py --robot_ip 10.0.2.7 --mode lb  --blending 10 30 80
  python trajectory_blend_test.py --robot_ip 10.0.2.7 --mode jb2 --speed 80 --accel 150 \\
      --waypoints "20,10,0,0,0,0" "-10,20,5,0,0,0" "0,0,0,0,0,0"
"""

import argparse
import asyncio
import json
import logging
import pathlib
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import rbpodo as rb

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
log = logging.getLogger('blend_test')

# ── 기본 waypoints (현재 위치 기준 상대값) ──────────────────────────────────
# jb2: 관절각 델타 (deg), shape (N, 6)
_DEFAULT_JB2_DELTAS = np.array([
    [ 20,  10,   0, 0, 0, 0],
    [-15,  25,   5, 0, 0, 0],
    [ 25,   5,  -5, 0, 0, 0],
    [-10,  30,   0, 0, 0, 0],
    [  0,   0,   0, 0, 0, 0],   # 원위치
], dtype=float)

# pb / lb: TCP 델타 (mm, deg), shape (N, 6)
_DEFAULT_PB_DELTAS = np.array([
    [150,    0,   0, 0, 0, 0],
    [150,  120,   0, 0, 0, 0],
    [  0,  120,   0, 0, 0, 0],
    [  0,    0,   0, 0, 0, 0],   # 원위치
], dtype=float)


# ── 유틸리티 ─────────────────────────────────────────────────────────────────

async def _read_state(data_ch: rb.asyncio.CobotData):
    data = await asyncio.wait_for(data_ch.request_data(), timeout=3.0)
    return (
        np.array(data.sdata.tcp_ref,  dtype=float),
        np.array(data.sdata.jnt_ref,  dtype=float),
    )


async def _record_loop(data_ch: rb.asyncio.CobotData,
                       records: list,
                       stop_event: asyncio.Event,
                       hz: float = 50.0):
    """robot state를 hz 주파수로 샘플링하는 백그라운드 태스크."""
    interval = 1.0 / hz
    while not stop_event.is_set():
        t0 = time.perf_counter()
        try:
            data = await asyncio.wait_for(data_ch.request_data(), timeout=0.5)
            records.append({
                't':      time.time(),
                'tcp':    list(data.sdata.tcp_ref),
                'joints': list(data.sdata.jnt_ref),
            })
        except Exception:
            pass
        dt = time.perf_counter() - t0
        await asyncio.sleep(max(0.0, interval - dt))


# ── 이동 헬퍼 ────────────────────────────────────────────────────────────────

async def _move_to_start(robot, rc, start_joints, speed, accel):
    log.info("시작 자세로 복귀 중 …")
    await robot.move_j(rc, start_joints, speed, accel)
    await robot.flush(rc)
    res = await robot.wait_for_move_started(rc, 5.0)
    if res.type() == rb.ReturnType.Success:
        await robot.wait_for_move_finished(rc)
    rc.error().throw_if_not_empty()
    log.info("시작 자세 도착.")
    await asyncio.sleep(0.3)


async def _run_jb2(robot, rc, waypoints, speed, accel, blending):
    await robot.move_jb2_clear(rc)
    await robot.flush(rc)
    for wp in waypoints:
        await robot.move_jb2_add(rc, wp, speed, accel, blending)
    await robot.flush(rc)
    await robot.move_jb2_run(rc)
    await robot.flush(rc)


async def _run_pb(robot, rc, waypoints, speed, accel, blending, blend_opt):
    await robot.move_pb_clear(rc)
    await robot.flush(rc)
    for wp in waypoints:
        await robot.move_pb_add(rc, wp, speed, blend_opt, blending)
    await robot.flush(rc)
    await robot.move_pb_run(rc, accel, rb.MovePBOption.Intended)
    await robot.flush(rc)


async def _run_lb(robot, rc, waypoints, speed, accel, blend_dist):
    await robot.move_lb_clear(rc)
    await robot.flush(rc)
    for wp in waypoints:
        await robot.move_lb_add(rc, wp, blend_dist)
    await robot.flush(rc)
    await robot.move_lb_run(rc, speed, accel, rb.MoveLBOption.Intended)
    await robot.flush(rc)


# ── 실행 + 기록 ──────────────────────────────────────────────────────────────

async def _execute_with_recording(robot, data_ch, rc, run_coro_fn, record_hz):
    """run_coro_fn()을 실행하면서 동시에 robot state를 기록."""
    records: list = []
    stop_event = asyncio.Event()

    record_task = asyncio.create_task(
        _record_loop(data_ch, records, stop_event, hz=record_hz)
    )

    t_start = time.time()

    await run_coro_fn()   # 명령 전송 (non-blocking)

    res = await robot.wait_for_move_started(rc, 10.0)
    if res.type() == rb.ReturnType.Success:
        await robot.wait_for_move_finished(rc)
    else:
        log.warning("move_started 타임아웃")

    t_end = time.time()

    stop_event.set()
    await record_task

    # t=0 을 모션 시작 기준으로 정규화
    for r in records:
        r['t'] -= t_start

    log.info("기록 완료: %d 샘플, %.2f 초", len(records), t_end - t_start)
    return records


# ── 플롯 ─────────────────────────────────────────────────────────────────────

def _plot_results(all_results: dict, mode: str, output_dir: pathlib.Path):
    bv_list = sorted(all_results.keys())
    colors   = cm.plasma(np.linspace(0.1, 0.9, len(bv_list)))

    blend_label = 'blending_value' if mode in ('jb2', 'pb') else 'blend_dist (mm)'

    if mode == 'jb2':
        # ① J1, J2 vs time
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('MoveJB2: blending 값에 따른 관절각도 변화', fontsize=13)
        for ax, j_idx, ylabel in zip(axes, [0, 1], ['J1 (deg)', 'J2 (deg)']):
            for bv, color in zip(bv_list, colors):
                recs = all_results[bv]
                t  = [r['t']          for r in recs]
                jv = [r['joints'][j_idx] for r in recs]
                ax.plot(t, jv, color=color, lw=1.8, label=f'{blend_label}={bv}')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        _save(fig, output_dir / 'jb2_joints_time.png')

        # ② J1-J2 공간 궤적
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title('MoveJB2: J1-J2 궤적 (blending 비교)')
        for bv, color in zip(bv_list, colors):
            recs = all_results[bv]
            j1 = [r['joints'][0] for r in recs]
            j2 = [r['joints'][1] for r in recs]
            ax.plot(j1, j2, color=color, lw=1.8, label=f'{blend_label}={bv}')
            if recs:
                ax.scatter(j1[0], j2[0], color=color, marker='o', s=60, zorder=5)
        ax.set_xlabel('J1 (deg)')
        ax.set_ylabel('J2 (deg)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        _save(fig, output_dir / 'jb2_j1j2_traj.png')

    else:  # pb or lb
        # ① TCP XY 궤적
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.set_title(f'Move{mode.upper()}: TCP XY 궤적 (blending 비교)')
        for bv, color in zip(bv_list, colors):
            recs = all_results[bv]
            x = [r['tcp'][0] for r in recs]
            y = [r['tcp'][1] for r in recs]
            ax.plot(x, y, color=color, lw=1.8, label=f'{blend_label}={bv}')
            if recs:
                ax.scatter(x[0], y[0], color=color, marker='o', s=60, zorder=5)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        _save(fig, output_dir / f'{mode}_tcp_xy.png')

        # ② TCP X, Y, Z vs time
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Move{mode.upper()}: blending에 따른 TCP 위치 변화', fontsize=13)
        for ax, idx, ylabel in zip(axes, [0, 1, 2], ['X (mm)', 'Y (mm)', 'Z (mm)']):
            for bv, color in zip(bv_list, colors):
                recs = all_results[bv]
                t  = [r['t']       for r in recs]
                vs = [r['tcp'][idx] for r in recs]
                ax.plot(t, vs, color=color, lw=1.8, label=f'{blend_label}={bv}')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        _save(fig, output_dir / f'{mode}_tcp_time.png')

    log.info("플롯 저장 완료: %s", output_dir.resolve())


def _save(fig, path: pathlib.Path):
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("  → %s", path.name)


# ── 메인 ─────────────────────────────────────────────────────────────────────

async def _main(args):
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blend_opt = (rb.BlendingOption.Ratio
                 if args.blending_option == 'ratio'
                 else rb.BlendingOption.Distance)

    log.info("로봇 연결 중: %s", args.robot_ip)
    robot   = rb.asyncio.Cobot(args.robot_ip)
    data_ch = rb.asyncio.CobotData(args.robot_ip)
    rc      = rb.ResponseCollector()

    await robot.set_operation_mode(rc, rb.OperationMode.Real)
    await robot.flush(rc)
    rc.error().throw_if_not_empty()
    await robot.set_speed_bar(rc, args.speed_bar)
    await robot.flush(rc)
    rc.error().throw_if_not_empty()

    # 현재 상태 읽기
    cur_tcp, cur_joints = await _read_state(data_ch)
    log.info("현재 TCP   : %s mm/deg", np.round(cur_tcp,    2).tolist())
    log.info("현재 관절  : %s deg",    np.round(cur_joints, 2).tolist())

    # 웨이포인트 (현재 위치 기준 상대값)
    if args.waypoints:
        deltas = np.array([list(map(float, d.split(','))) for d in args.waypoints])
    else:
        deltas = _DEFAULT_JB2_DELTAS if args.mode == 'jb2' else _DEFAULT_PB_DELTAS

    if args.mode == 'jb2':
        waypoints    = [cur_joints + d for d in deltas]
        start_joints = cur_joints.copy()
    else:
        waypoints    = [cur_tcp + d for d in deltas]
        start_joints = cur_joints.copy()

    log.info("웨이포인트 %d 개 (델타 기준):", len(waypoints))
    for i, (d, wp) in enumerate(zip(deltas, waypoints)):
        log.info("  WP%d: delta=%s  →  %s", i + 1,
                 np.round(d, 2).tolist(), np.round(wp, 2).tolist())

    all_results: dict = {}

    for bv in args.blending:
        log.info("=" * 60)
        log.info("▶ blending=%.3f  시작", bv)

        await _move_to_start(robot, rc, start_joints, args.speed, args.accel)

        if args.mode == 'jb2':
            async def run_fn(bv=bv):
                await _run_jb2(robot, rc, waypoints, args.speed, args.accel, bv)
        elif args.mode == 'pb':
            async def run_fn(bv=bv):
                await _run_pb(robot, rc, waypoints, args.speed, args.accel, bv, blend_opt)
        else:  # lb
            async def run_fn(bv=bv):
                await _run_lb(robot, rc, waypoints, args.speed, args.accel, bv)

        records = await _execute_with_recording(robot, data_ch, rc, run_fn, args.record_hz)
        all_results[bv] = records

        save_path = output_dir / f'blend_{bv:.3f}_{args.mode}.json'
        with open(save_path, 'w') as f:
            json.dump({
                'blending': bv,
                'mode': args.mode,
                'waypoints': [w.tolist() for w in waypoints],
                'records': records,
            }, f)
        log.info("데이터 저장: %s", save_path.name)

    try:
        await robot.disconnect(rc)
    except Exception:
        pass

    _plot_results(all_results, args.mode, output_dir)
    log.info("완료. 결과: %s", output_dir.resolve())


def main():
    p = argparse.ArgumentParser(
        description='blending 값에 따른 실제 궤적 비교',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--robot_ip',        default='10.0.2.7')
    p.add_argument('--mode',            choices=['jb2', 'pb', 'lb'], default='jb2',
                   help='이동 모드 (기본: jb2)')
    p.add_argument('--blending',        type=float, nargs='+',
                   default=[0.0, 0.3, 0.7, 1.0],
                   help='테스트할 blending 값 목록')
    p.add_argument('--blending_option', choices=['ratio', 'distance'], default='ratio',
                   help='pb 모드 blending 옵션 (기본: ratio)')
    p.add_argument('--speed',           type=float, default=60.0,
                   help='속도 (jb2: deg/s, pb/lb: mm/s, 기본: 60)')
    p.add_argument('--accel',           type=float, default=100.0,
                   help='가속도 (jb2: deg/s², pb/lb: mm/s², 기본: 100)')
    p.add_argument('--speed_bar',       type=float, default=0.3,
                   help='speed bar (0~1, 기본: 0.3)')
    p.add_argument('--record_hz',       type=float, default=50.0,
                   help='기록 샘플링 주파수 (Hz, 기본: 50)')
    p.add_argument('--waypoints',       nargs='*',
                   help='커스텀 웨이포인트 (델타값, "j1,j2,j3,j4,j5,j6" 형식)')
    p.add_argument('--output_dir',      default='blend_results',
                   help='결과 저장 디렉토리 (기본: blend_results)')
    args = p.parse_args()

    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        log.info("사용자 중단.")


if __name__ == '__main__':
    main()
