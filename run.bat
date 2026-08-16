@echo off
REM Photo Translator -- put photos in input\, run this, collect output\.
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo First run: creating the environment. This takes a few minutes.
    python -m venv venv || goto :nopython
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

python PhotoTranslator.py %*
goto :eof

:nopython
echo.
echo Python was not found. Install it from https://python.org
echo and tick "Add python.exe to PATH" during setup.
pause
