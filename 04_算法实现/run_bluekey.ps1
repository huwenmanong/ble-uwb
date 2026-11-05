param(
    [Parameter(Mandatory = $true)]
    [string]$ScriptPath,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArguments
)

# The user-facing environment name is blue-key; the installed Conda name is bluekey.
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = 'D:\anaconda\envs\bluekey\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "bluekey Python was not found at $Python"
}

if (-not [System.IO.Path]::IsPathRooted($ScriptPath)) {
    $ScriptPath = Join-Path $ProjectRoot $ScriptPath
}
if (-not (Test-Path -LiteralPath $ScriptPath)) {
    throw "Script was not found: $ScriptPath"
}

Push-Location $ProjectRoot
try {
    & $Python $ScriptPath @ScriptArguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
