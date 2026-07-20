@echo off
chcp 65001 >nul
title Xiangqi Assistant EXE Builder

echo =======================================================
echo   Windows Xiangqi Desktop Assistant Builder
echo =======================================================

python --version >nul 2>&1
if errorlevel 1 goto ERROR_NO_PYTHON

echo [1/3] Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 goto ERROR_PIP_FAILED

echo [2/3] Building desktop GUI...
python -m PyInstaller --noconfirm --clean --onedir --windowed ^
  --add-data "pikafish.nnue;." ^
  --add-binary "pikafish.exe;." ^
  app.py -n "XiangqiAssistant"
if errorlevel 1 goto ERROR_PACK_FAILED

echo [3/3] Build completed: dist\XiangqiAssistant\XiangqiAssistant.exe
pause
exit /b 0

:ERROR_NO_PYTHON
echo [ERROR] Python 3.10+ is not installed or not in PATH.
pause
exit /b 1

:ERROR_PIP_FAILED
echo [ERROR] Dependency installation failed.
pause
exit /b 1

:ERROR_PACK_FAILED
echo [ERROR] EXE build failed.
pause
exit /b 1
