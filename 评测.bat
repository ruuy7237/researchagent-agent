@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
title ResearchAgent 调研 评测

python -V >nul 2>nul
if errorlevel 1 goto NOPY

if not exist ".venv\Scripts\python.exe" goto INSTALL
".venv\Scripts\python.exe" -c "import fastapi" >nul 2>nul
if errorlevel 1 goto INSTALL
goto RUN

:INSTALL
echo [安装] 首次运行需要装依赖，大约 1-2 分钟
if not exist ".venv\Scripts\python.exe" python -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple >nul 2>nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 ".venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
echo.

:RUN
set "PATH=%CD%\.venv\Scripts;%PATH%"
".venv\Scripts\python.exe" eval\run_eval.py
echo.
echo 结果已写入 data\metrics.json
pause
exit /b 0

:NOPY
echo.
echo [错误] 没找到 python 命令，请先装 Python 3.10+
pause
exit /b 1
