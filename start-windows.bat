@echo off
rem TabForge one-click launcher: installs the free local helper and opens the app.
cd /d "%~dp0"
title TabForge

where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% --version >nul 2>nul || (
  echo Python 3.10+ is required: https://www.python.org/downloads/  ^(tick "Add Python to PATH"^)
  pause
  exit /b 1
)

echo Installing / updating the helper (first run takes a moment)...
%PY% -m pip install --quiet --upgrade -r helper\requirements.txt

if not exist helper\.separation-asked (
  echo.
  set /p SEP="Also install AI stem separation (better accuracy, ~2 GB download)? [y/N] "
  echo asked> helper\.separation-asked
)
if /i "%SEP%"=="y" (
  %PY% -m pip install --quiet -r helper\requirements-separation.txt
  %PY% -m pip install --quiet --no-deps demucs==4.0.1
)

%PY% helper\tabforge_helper.py
pause
