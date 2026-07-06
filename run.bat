@echo off
setlocal

rem Set this to a specific interpreter when using conda or a custom venv.
rem Example: set "PYTHON_CONFIG=C:\Users\admin\miniforge3\envs\drt\python.exe"
set "PYTHON_CONFIG="

set "VENV_DIR=%~dp0venv"
if defined PYTHON_CONFIG (
    set "PYTHON=%PYTHON_CONFIG%"
) else if exist "%VENV_DIR%\Scripts\python.exe" (
    set "PYTHON=%VENV_DIR%\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

if "%1"=="" goto help

if /I "%1"=="monitor" goto monitor
if /I "%1"=="viewer" goto viewer
if /I "%1"=="controller" goto controller
if /I "%1"=="zproxy" goto zproxy
if /I "%1"=="simtool" goto simtool
if /I "%1"=="verifycobot" goto verifycobot
if /I "%1"=="verifypositioner" goto verifypositioner
if /I "%1"=="run" goto run

echo Unknown target: %1
goto help

:monitor
"%PYTHON%" monitor.py --config drt.cfg
goto end

:viewer
"%PYTHON%" python\viewer.py --config "%~dp0python\viewer.cfg"
goto end

:controller
"%PYTHON%" python\controller.py --config "%~dp0python\controller.cfg"
goto end

:zproxy
"%PYTHON%" python\zproxy.py --config "%~dp0python\zproxy.cfg"
goto end

:simtool
"%PYTHON%" python\simtool.py --config "%~dp0python\simtool.cfg"
goto end

:verifycobot
"%PYTHON%" python\verifycobot.py --config "%~dp0python\verifycobot.cfg"
goto end

:verifypositioner
"%PYTHON%" python\verifypositioner.py --config "%~dp0python\verifypositioner.cfg"
goto end

:run
rem Start background processes
start "zproxy" "%PYTHON%" python\zproxy.py --config "%~dp0python\zproxy.cfg"
start "viewer" "%PYTHON%" python\viewer.py --config "%~dp0python\viewer.cfg"
rem Run the last one in the foreground
"%PYTHON%" python\simtool.py --config "%~dp0python\simtool.cfg"
goto end

:help
echo Usage: run.bat [target]
echo Targets: monitor, viewer, controller, zproxy, simtool, verifycobot, verifypositioner, run
goto end

:end
endlocal
