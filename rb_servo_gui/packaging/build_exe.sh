#!/usr/bin/env bash
# 단일 실행 파일(onedir) 빌드: Python 이 없는 PC 에 배포할 때 사용. (빌드한 OS 용 결과물이 나온다)
set -e
cd "$(dirname "$0")/.."
python3 -m pip install -r requirements.txt pyinstaller
PYTHONPATH=src python3 -m PyInstaller --noconfirm --clean --windowed --name rb_servo_gui \
    --paths src --collect-submodules rb_servo_gui \
    --add-data "examples:examples" \
    src/rb_servo_gui/__main__.py
echo "결과물: dist/rb_servo_gui/"
