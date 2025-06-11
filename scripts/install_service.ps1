# Install MT5 Trading System as Windows Service
# Requires NSSM (Non-Sucking Service Manager) to be installed

$serviceName = "MT5QuantSystem"
$pythonPath = "python"
$scriptPath = Join-Path $PSScriptRoot "..\main.py"
$workingDir = Split-Path $scriptPath

# Check if NSSM is installed
$nssmPath = "C:\Program Files\nssm\win64\nssm.exe"
if (-not (Test-Path $nssmPath)) {
    Write-Host "NSSM not found. Please install NSSM first from https://nssm.cc/download"
    exit 1
}

# Remove existing service if it exists
& $nssmPath stop $serviceName
& $nssmPath remove $serviceName confirm

# Install the service
& $nssmPath install $serviceName $pythonPath "$scriptPath"
& $nssmPath set $serviceName AppDirectory $workingDir
& $nssmPath set $serviceName DisplayName "MT5 Quant Trading System"
& $nssmPath set $serviceName Description "Automated MT5 Trading System with Multi-Timeframe Analysis"
& $nssmPath set $serviceName Start SERVICE_AUTO_START
& $nssmPath set $serviceName AppStdout "$workingDir\logs\service_stdout.log"
& $nssmPath set $serviceName AppStderr "$workingDir\logs\service_stderr.log"
& $nssmPath set $serviceName AppRotateFiles 1
& $nssmPath set $serviceName AppRotateOnline 1
& $nssmPath set $serviceName AppRotateSeconds 86400
& $nssmPath set $serviceName AppRotateBytes 10485760

# Create logs directory if it doesn't exist
New-Item -ItemType Directory -Force -Path "$workingDir\logs"

# Start the service
Start-Service $serviceName

Write-Host "Service installed and started. Check logs in $workingDir\logs"
Write-Host "To manage service:"
Write-Host "  Start: Start-Service $serviceName"
Write-Host "  Stop: Stop-Service $serviceName"
Write-Host "  Status: Get-Service $serviceName" 