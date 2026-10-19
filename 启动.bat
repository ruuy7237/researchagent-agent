@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
title ResearchAgent 调研

echo ============================================
echo  ResearchAgent 调研
echo ============================================
echo.

python -V >nul 2>nul
if errorlevel 1 goto NOPY

if not exist ".venv\Scripts\python.exe" goto INSTALL
".venv\Scripts\python.exe" -c "import fastapi" >nul 2>nul
if errorlevel 1 goto INSTALL
goto RUN

:INSTALL
echo [安装] 首次运行需要装依赖，大约 1-2 分钟，只做一次
echo.
if not exist ".venv\Scripts\python.exe" (
  echo   创建虚拟环境 .venv ...
  python -m venv .venv
)
echo   升级 pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
echo   安装依赖 ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if not errorlevel 1 goto INSTALLED
echo.
echo   清华镜像失败，换阿里云重试 ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
if errorlevel 1 goto DEPFAIL

:INSTALLED
echo.
echo [安装] 完成
echo.

:RUN
set "PATH=%CD%\.venv\Scripts;%PATH%"
echo [准备] 建立检索索引 ...
echo [启动] 服务地址 http://127.0.0.1:8013
echo [启动] 浏览器会自己打开，没打开就把上面的地址复制进去
echo [启动] 关掉这个窗口就停止服务
echo.
start /min cmd /c "ping -n 4 127.0.0.1 >nul & start "" http://127.0.0.1:8013"
".venv\Scripts\python.exe" -m uvicorn app.server:app --port 8013
pause
exit /b 0

:DEPFAIL
echo.
echo [失败] 依赖没装上，多半是网络问题。可以手动执行：
echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
pause
exit /b 1

:NOPY
echo.
echo [错误] 没找到 python 命令
echo 请先装 Python 3.10 或更高版本，安装时勾选 "Add Python to PATH"
pause
exit /b 1
