@echo off
:: Vetinari Startup Script (Windows)
:: Usage: start.bat [options]
::   start.bat                          -- Start with dashboard
::   start.bat --goal "My goal here"    -- Run a specific goal
::   start.bat --no-dashboard           -- CLI only
::   start.bat serve                    -- Dashboard only

setlocal

:: Change to script directory
cd /d "%~dp0"

:: Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

:: Set default environment variables if not already set
if not defined LM_STUDIO_HOST (
    set LM_STUDIO_HOST=http://100.78.30.7:1234
)

echo ============================================================
echo  VETINARI AI Orchestration System
echo  Host: %LM_STUDIO_HOST%
echo ============================================================

:: Run Vetinari
if "%1"=="" (
    python -m vetinari start %*
) else if "%1"=="serve" (
    python -m vetinari serve %2 %3 %4 %5
) else if "%1"=="run" (
    python -m vetinari run %2 %3 %4 %5 %6
) else if "%1"=="status" (
    python -m vetinari status
) else if "%1"=="review" (
    python -m vetinari review
) else (
    python -m vetinari start %*
)

pause
