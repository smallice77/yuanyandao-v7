<#
.SYNOPSIS
    元衍道枢 YuanYanDaoShu V7.0 · RTX5060 8G 一键启动 / 自检

.EXAMPLE
    .\run_rtx5060.ps1              # 全量回归（六条内置指令 + 7 项自检）
    .\run_rtx5060.ps1 -Health      # 只做部署自检
    .\run_rtx5060.ps1 -Serve       # 启动 HTTP 服务（127.0.0.1:18080）
    .\run_rtx5060.ps1 -Serve -Port 18081
#>
param(
    [switch]$Serve,
    [switch]$Test,
    [switch]$Health,
    [int]$Port = 18080
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " 元衍道枢 YuanYanDaoShu V7.0 · RTX5060 8G · Qwen2.5-7B" -ForegroundColor Cyan
Write-Host " 核心锚点密钥：元衍道枢" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

python -V
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Health) {
    python -m yuanyandao health
    exit $LASTEXITCODE
}

if ($Serve) {
    Write-Host "[..] 自检中（Ollama 不可用会自动降级，不影响启动）" -ForegroundColor Yellow
    python -m yuanyandao health *> $null
    Write-Host "[OK] 服务启动：http://127.0.0.1:$Port/v1/invoke" -ForegroundColor Green
    python -m yuanyandao serve --port $Port
    exit $LASTEXITCODE
}

python tests/run_tests.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] 回归未通过，请检查上方输出" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[OK] 六条内置指令与全部自检通过" -ForegroundColor Green
Write-Host "下一步：.\run_rtx5060.ps1 -Serve   启动 HTTP 服务" -ForegroundColor DarkGray
