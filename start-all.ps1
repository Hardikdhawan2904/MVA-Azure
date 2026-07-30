# Starts the shared Postgres instance plus all four services, each in its own
# terminal window (so logs stay separate and readable). Run once from the repo root:
#   powershell -File start-all.ps1
#
# Postgres is a native Windows instance (not Docker) as of the Docker-removal
# migration -- see the plan doc for why. Data dir: C:\PGData\mva-pipeline,
# port 5433 (same port Docker used, so no other config changed). It isn't
# registered as a Windows Service (no admin rights were available when this
# was set up), so this script starts it the same way it starts everything
# else: explicitly, every time, idempotently.

$root = $PSScriptRoot
$pgBin = "C:\Program Files\PostgreSQL\17\bin"
$pgData = "C:\PGData\mva-pipeline"

Write-Host "Starting shared Postgres (native, port 5433)..." -ForegroundColor Cyan
$pgStatus = & "$pgBin\pg_ctl.exe" -D $pgData status
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Already running." -ForegroundColor DarkGray
} else {
    & "$pgBin\pg_ctl.exe" -D $pgData -l "$pgData\startup.log" start
}

Start-Sleep -Seconds 2

$py = "$root\venv\Scripts\python.exe"

Write-Host "Starting Agent 1 (Schema Intelligence Layer) on :8000..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$root\Schema-Intelligence-Layer'; & '$py' -m uvicorn app.main:app --port 8000 --reload"

Write-Host "Starting Agent 2 (Data Profiling Layer) on :8001..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$root\Data-Profiling-Agent'; & '$py' -m uvicorn app.main:app --port 8001 --reload"

Write-Host "Starting Orchestrator on :8002..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$root\Agent-Orchestrator'; & '$py' -m uvicorn app.main:app --port 8002 --reload"

Write-Host "Starting Agent 3 (Analytics Agent) on :8003..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$root\Analytics-Agent'; & '$py' -m uvicorn app.main:app --port 8003 --reload"

Write-Host ""
Write-Host "All services launching. Give them a few seconds, then check:" -ForegroundColor Green
Write-Host "  Agent 1:      http://127.0.0.1:8000/health"
Write-Host "  Agent 2:      http://127.0.0.1:8001/api/v1/health"
Write-Host "  Orchestrator: http://127.0.0.1:8002/health"
Write-Host "  Agent 3:      http://127.0.0.1:8003/health"
Write-Host "  Full pipeline: http://127.0.0.1:8002/docs"
