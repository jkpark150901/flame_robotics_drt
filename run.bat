@echo off
setlocal

rem PYTHON_CONFIG can override the default Miniforge drt environment.
rem Example: set "PYTHON_CONFIG=C:\path\to\python.exe"

set "VENV_DIR=%~dp0venv"
set "MINIFORGE_DRT=%USERPROFILE%\miniforge3\envs\drt\python.exe"
if defined PYTHON_CONFIG (
    set "PYTHON=%PYTHON_CONFIG%"
) else if exist "%MINIFORGE_DRT%" (
    set "PYTHON=%MINIFORGE_DRT%"
) else if exist "%VENV_DIR%\Scripts\python.exe" (
    set "PYTHON=%VENV_DIR%\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

if "%1"=="" goto help

set "TARGET=%~1"
shift
set "EXTRA_ARGS="

:collect_args
if "%~1"=="" goto dispatch
set "EXTRA_ARGS=%EXTRA_ARGS% %1"
shift
goto collect_args

:dispatch

if /I "%TARGET%"=="monitor" goto monitor
if /I "%TARGET%"=="viewer" goto viewer
if /I "%TARGET%"=="robotcore" goto robotcore
if /I "%TARGET%"=="controller" goto controller
if /I "%TARGET%"=="zproxy" goto zproxy
if /I "%TARGET%"=="simtool" goto simtool
if /I "%TARGET%"=="verifycobot" goto verifycobot
if /I "%TARGET%"=="verifypositioner" goto verifypositioner
if /I "%TARGET%"=="run" goto run
if /I "%TARGET%"=="run-embedded" goto run_embedded

echo Unknown target: %TARGET%
goto help

:monitor
"%PYTHON%" monitor.py --config drt.cfg %EXTRA_ARGS%
goto end

:viewer
"%PYTHON%" python\viewervedo.py --config "%~dp0python\viewervedo.cfg" %EXTRA_ARGS%
goto end

:robotcore
"%PYTHON%" python\robotcore.py --config "%~dp0python\viewervedo.cfg" %EXTRA_ARGS%
goto end

:controller
"%PYTHON%" python\controller.py --config "%~dp0python\controller.cfg" %EXTRA_ARGS%
goto end

:zproxy
"%PYTHON%" python\zproxy.py --config "%~dp0python\zproxy.cfg" %EXTRA_ARGS%
goto end

:simtool
"%PYTHON%" python\simtool.py --config "%~dp0python\simtool.cfg" %EXTRA_ARGS%
goto end

:verifycobot
"%PYTHON%" python\verifycobot.py --config "%~dp0python\verifycobot.cfg" %EXTRA_ARGS%
goto end

:verifypositioner
"%PYTHON%" python\verifypositioner.py --config "%~dp0python\verifypositioner.cfg" %EXTRA_ARGS%
goto end

:run
rem Start the standalone robot-core topology.
start "zproxy" "%PYTHON%" python\zproxy.py --config "%~dp0python\zproxy.cfg"
start "robotcore" "%PYTHON%" python\robotcore.py --config "%~dp0python\viewervedo.cfg"
start "viewer" "%PYTHON%" python\viewervedo.py --config "%~dp0python\viewervedo.cfg" --robot_core_mode external
rem SimTool owns the workflow and remains in the foreground.
"%PYTHON%" python\simtool.py --config "%~dp0python\simtool.cfg" %EXTRA_ARGS%
goto end

:run_embedded
rem Viewer owns a child robot-core process in embedded mode.
start "zproxy" "%PYTHON%" python\zproxy.py --config "%~dp0python\zproxy.cfg"
start "viewer" "%PYTHON%" python\viewervedo.py --config "%~dp0python\viewervedo.cfg" --robot_core_mode embedded
"%PYTHON%" python\simtool.py --config "%~dp0python\simtool.cfg" %EXTRA_ARGS%
goto end

:help
echo Usage: run.bat [target] [additional arguments]
echo Targets:
echo   robotcore       Standalone pose/path planning service
echo   viewer          3D visualization process
echo   simtool         Workflow and UI process
echo   zproxy          ZAPI message proxy
echo   run             zproxy + standalone robotcore + viewer + simtool
echo   run-embedded    zproxy + viewer with child robotcore + simtool
echo   monitor, controller, verifycobot, verifypositioner
echo.
echo Examples:
echo   set "PYTHON_CONFIG=C:\path\to\python.exe"
echo   run.bat viewer --verbose_level DEBUG
echo   run.bat robotcore --endpoint tcp://127.0.0.1:5557
echo   run.bat viewer --robot_core_mode external
echo   run.bat simtool --verbose_level INFO
echo   run.bat run
echo   run.bat run-embedded
goto end

:end
endlocal
