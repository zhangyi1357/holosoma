[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$NumEnvs = 4096,
    [int]$Seed = 1,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:OMNI_KIT_ACCEPT_EULA = "YES"

$rootDir = Split-Path -Parent $PSScriptRoot
$defaultPython = Join-Path $env:USERPROFILE ".holosoma_deps\miniconda3\envs\hssim\python.exe"
$python = if ($env:HOLOSOMA_ISAACSIM_PYTHON) { $env:HOLOSOMA_ISAACSIM_PYTHON } else { $defaultPython }

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Isaac Sim Python was not found at '$python'. Set HOLOSOMA_ISAACSIM_PYTHON to the Python executable in your Isaac Sim 5.1 environment."
}

& $python (Join-Path $rootDir "src\holosoma\holosoma\train_agent.py") `
    "exp:a3-t3d0-wbt-fast-sac" `
    "--training.num-envs" $NumEnvs `
    "--training.seed" $Seed `
    @ExtraArgs

exit $LASTEXITCODE
