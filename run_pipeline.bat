@echo off
echo ============================================================
echo   LM Studio Local Multi-Model Pipeline
echo ============================================================
echo.
echo   Usage:
echo     run_pipeline.bat                          (default task)
echo     run_pipeline.bat "Your custom task"       (custom task)
echo     run_pipeline.bat --report report.txt      (analyze a report)
echo     run_pipeline.bat -r report.txt -t "Find trade ideas"
echo.

REM Use Anaconda Python (system python may not be in PATH)
set PYTHON=C:\Users\Admin\anaconda3\python.exe

REM Check Python is available
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found at %PYTHON%
    echo         Edit this file and set PYTHON to your python.exe path.
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
%PYTHON% local_model_router.py %*

echo.
echo [DONE] Pipeline finished. Check the outputs/ folder.
pause
