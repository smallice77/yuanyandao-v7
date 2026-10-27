@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

echo ==========================================================
echo  YuanYanDaoShu V7.0   RTX5060 8G   Qwen2.5-7B
echo  Anchor: YuanYanDaoShu
echo ==========================================================

if /I "%~1"=="serve" goto :serve
if /I "%~1"=="health" goto :health
if /I "%~1"=="keywords" goto :keywords
if /I "%~1"=="test" goto :test
goto :test

:serve
python -m yuanyandao serve
goto :end

:health
python -m yuanyandao health
goto :end

:keywords
python -m yuanyandao keywords --category ta
goto :end

:test
python tests/run_tests.py
if errorlevel 1 (
  echo [FAIL] regression failed
) else (
  echo [OK] all checks passed
)

:end
endlocal
