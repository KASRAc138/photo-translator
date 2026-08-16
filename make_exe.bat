@echo off
REM Build PhotoTranslator.exe -- one portable file, no dependencies for whoever runs it.
REM Double-click this. It takes a few minutes the first time.
cd /d "%~dp0"
setlocal

if not exist "venv\Scripts\python.exe" (
    echo Creating the build environment...
    python -m venv venv || goto :nopython
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt pyinstaller || goto :fail

REM Raqm needs fribidi.dll on Windows to activate. Without it the app falls back
REM to arabic-reshaper, which is verified to render identically -- so this is a
REM note, not a failure.
python build_portable.py --onefile || goto :fail

echo.
echo ============================================================
echo   Done.  dist\PhotoTranslator.exe
echo   Copy that one file anywhere and double-click it.
echo   First run downloads the language pack (about 100 MB, once).
echo ============================================================
echo.
pause
exit /b 0

:nopython
echo.
echo Python was not found. Install it from https://python.org
echo and tick "Add python.exe to PATH" during setup.
pause
exit /b 1

:fail
echo.
echo Build failed -- see the messages above.
pause
exit /b 1
