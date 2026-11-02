@echo off
rem 단일 폴더 실행 파일 빌드 (Windows)
cd /d "%~dp0\.."
python -m pip install -r requirements.txt pyinstaller
set PYTHONPATH=src
python -m PyInstaller --noconfirm --clean --windowed --name rb_servo_gui --paths src --collect-submodules rb_servo_gui --add-data "examples;examples" src\rb_servo_gui\__main__.py
echo 결과물: dist\rb_servo_gui\
