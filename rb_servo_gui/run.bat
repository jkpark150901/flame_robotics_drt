@echo off
rem 설치 없이 바로 실행: run.bat [--selftest]
cd /d "%~dp0"
set PYTHONPATH=src
python -m rb_servo_gui %*
