@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto failed
echo.
echo 安装完成。打开微信游戏后，运行 start.bat。
pause
exit /b 0
:failed
echo 安装失败。请安装 64 位 Python 3.13，并确认网络可用。
pause
exit /b 1
