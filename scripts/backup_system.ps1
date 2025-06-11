# Backup script for MT5 Quant Trading System
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = "..\backups\mt5_quant_system_backup_$timestamp"
$sourceDir = ".."
$zipFile = "$backupDir.zip"

# Create backup directory
New-Item -ItemType Directory -Force -Path $backupDir | Out-Null

# Files to backup (excluding logs, cache, and temporary files)
$filesToBackup = @(
    "main.py",
    "display_manager.py",
    "requirements.txt",
    "config.py",
    "trade_manager.py",
    "strategy.py",
    "indicators.py",
    "mt5_utils.py"
)

# Directories to backup
$dirsToBackup = @(
    "core",
    "scripts"
)

# Copy files
foreach ($file in $filesToBackup) {
    if (Test-Path "$sourceDir\$file") {
        Copy-Item "$sourceDir\$file" "$backupDir\$file" -Force
        Write-Host "Backed up: $file"
    }
}

# Copy directories
foreach ($dir in $dirsToBackup) {
    if (Test-Path "$sourceDir\$dir") {
        Copy-Item "$sourceDir\$dir" "$backupDir\$dir" -Recurse -Force
        Write-Host "Backed up directory: $dir"
    }
}

# Create a README with backup info
$backupInfo = @"
MT5 Quant Trading System Backup
Created: $(Get-Date)
Version: 1.0
Backup Timestamp: $timestamp

Contents:
- Main trading system files
- Core modules
- Configuration
- Scripts
- Requirements

Note: Logs and cache files are excluded from backup.
"@

$backupInfo | Out-File "$backupDir\README.txt" -Encoding UTF8

# Create zip archive
Compress-Archive -Path $backupDir -DestinationPath $zipFile -Force

# Clean up the temporary directory
Remove-Item $backupDir -Recurse -Force

Write-Host "`nBackup completed successfully!"
Write-Host "Backup file: $zipFile"
Write-Host "`nBackup contents:"
Get-ChildItem $zipFile | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize 