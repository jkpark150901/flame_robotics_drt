"""
check_robot_data.py
===================
Rainbow Robotics rbpodo 통신 테스트.

로봇에 연결해 TCP 위치/자세 및 관절 각도를 주기적으로 출력합니다.

사용 예:
  python check_robot_data.py --robot_ip 10.0.2.7
  python check_robot_data.py --robot_ip 10.0.2.7 --interval 0.5 --duration 30
"""

import argparse
import asyncio
import logging
import time

import numpy as np

import rbpodo as rb

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S',
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def _fmt(values, fmt='.3f', unit=''):
    return '[' + ', '.join(f'{v:{fmt}}' for v in values) + ']' + (f' {unit}' if unit else '')


async def _run(args):
    log.info("로봇 연결 중: %s", args.robot_ip)
    data_channel = rb.asyncio.CobotData(args.robot_ip)
    log.info("CobotData 연결 완료.")

    t_start = time.time()
    t_end = t_start + args.duration if args.duration > 0 else float('inf')
    count = 0

    try:
        while time.time() < t_end:
            t0 = time.time()
            data = await data_channel.request_data()

            tcp = list(data.sdata.tcp_ref)        # [x, y, z, rx, ry, rz]  mm / deg
            joints = list(data.sdata.jnt_ref)      # [J1..J6]  deg

            tcp_pos_mm = tcp[:3]
            tcp_rot_deg = tcp[3:]

            count += 1
            elapsed = time.time() - t_start
            log.info(
                "[%.1f s | #%d]  TCP pos: %s mm  rot: %s deg  |  joints: %s deg",
                elapsed,
                count,
                _fmt(tcp_pos_mm, '.2f'),
                _fmt(tcp_rot_deg, '.2f'),
                _fmt(joints, '.2f'),
            )

            dt = time.time() - t0
            sleep = max(0.0, args.interval - dt)
            await asyncio.sleep(sleep)

    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("사용자 중단.")

    log.info("총 %d 샘플 수집 완료 (%.1f 초).", count, time.time() - t_start)


def main():
    p = argparse.ArgumentParser(description='Rainbow Robotics 로봇 TCP/관절 데이터 수신 테스트')
    p.add_argument('--robot_ip', default='10.0.2.7', help='로봇 IP 주소')
    p.add_argument('--interval', type=float, default=0.2, help='샘플 출력 주기 (초, 기본 0.2)')
    p.add_argument('--duration', type=float, default=0,
                   help='실행 시간(초). 0이면 Ctrl+C 때까지 계속 (기본 0)')
    args = p.parse_args()

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
