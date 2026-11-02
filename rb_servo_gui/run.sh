#!/usr/bin/env bash
# 설치 없이 바로 실행: ./run.sh [--selftest]
# conda/ROS 의 LD_LIBRARY_PATH 가 시스템 Tk 대신 다른 libtk 를 잡아 한글이 깨지는 문제를 피하려고 비운다
cd "$(dirname "$0")" && PYTHONPATH=src exec env -u LD_LIBRARY_PATH python3 -m rb_servo_gui "$@"
