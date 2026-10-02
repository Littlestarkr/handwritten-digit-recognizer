@echo off
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py predict_gui.py
) else (
    python predict_gui.py
)

if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to close this window.
    pause >nul
)
