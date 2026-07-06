@echo off
setlocal

rem Export only the files used by python\verifypositioner.py and python\verifytool.
rem Usage:
rem   export_verifytool.bat
rem   export_verifytool.bat D:\verifytool_export

set "ROOT=%~dp0"
set "DEST=%~1"
if "%DEST%"=="" set "DEST=%ROOT%verifytool_export"

if not exist "%DEST%" mkdir "%DEST%"

call :copy_file "python\verifypositioner.py"
call :copy_file "python\verifypositioner.cfg"
call :copy_file "python\resource\NanumSquareR.ttf"
call :copy_file "python\verifytool\README.md"
call :copy_file "python\verifytool\__init__.py"
call :copy_file "python\verifytool\verifypositioner.py"
call :copy_file "python\verifytool\verifypositioner.ui"
call :copy_file "python\verifytool\workers\__init__.py"
call :copy_file "python\verifytool\workers\natnet_worker.py"
call :copy_file "tools\NatNet\NatNetClient.py"

echo.
echo Export complete:
echo   %DEST%
echo.
echo Run from the exported folder:
echo   cd /d "%DEST%"
echo   python python\verifypositioner.py
echo.
exit /b 0

:copy_file
set "REL=%~1"
set "SRC=%ROOT%%REL%"
set "OUT=%DEST%\%REL%"

if not exist "%SRC%" (
    echo [WARN] Missing: %REL%
    exit /b 0
)

for %%I in ("%OUT%") do if not exist "%%~dpI" mkdir "%%~dpI"
copy /Y "%SRC%" "%OUT%" >nul
echo [OK] %REL%
exit /b 0
