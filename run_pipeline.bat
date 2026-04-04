@echo off
echo ============================================================
echo   LM Studio Local Multi-Model Pipeline
echo ============================================================
echo.

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo         Install Python or add it to your PATH.
    pause
    exit /b 1
)

REM Check lms CLI is available
lms --version >nul 2>&1
if errorlevel 1 (
    echo [WARN] lms CLI not found. Model load/unload may fail.
    echo        Make sure LM Studio CLI is installed.
    echo.
)

REM Run the pipeline
echo [START] Running pipeline ...
echo.
python local_model_router.py %*

echo.
echo [DONE] Pipeline finished. Check the outputs/ folder.
pause
