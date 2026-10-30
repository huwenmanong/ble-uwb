$ErrorActionPreference = 'Stop'
$Runner = Join-Path $PSScriptRoot 'run_bluekey.ps1'
$Scripts = Join-Path $PSScriptRoot 'scripts'

& $Runner (Join-Path $Scripts 'check_bluekey_environment.py')
& $Runner (Join-Path $Scripts 'audit_d1_stage0.py')
& $Runner (Join-Path $Scripts 'investigate_d1_env3_mismatch.py')
& $Runner (Join-Path $Scripts 'freeze_d1_loso_splits.py')

Write-Host 'Stage-0 pipeline completed under bluekey.'
