param(
  [switch]$SkipRust,
  [switch]$SkipJava
)

$ErrorActionPreference = "Stop"

function Run-Step($label, $cmd) {
  Write-Host ""
  Write-Host "==> $label"
  Write-Host "    $cmd"
  iex $cmd
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

Run-Step "Python stdlib tests (Step 12)" "python tests/integration/run_stdlib_step12_unittest.py"

Run-Step "Python REPL smoke test (Step 13)" "python -m toolchain.nova_repl --exec ""1+2"""

if (-not $SkipRust) {
  if (Get-Command cargo -ErrorAction SilentlyContinue) {
    Run-Step "Rust workspace tests (compiler)" "cd compiler; cargo test"
    Set-Location $RepoRoot
  } else {
    Write-Host ""
    Write-Host "==> Rust tests skipped (cargo not found)"
  }
}

if (-not $SkipJava) {
  if (Get-Command mvn -ErrorAction SilentlyContinue) {
    Run-Step "Java interface validator tests" "cd compiler/interface_validator; mvn test"
    Set-Location $RepoRoot
  } else {
    Write-Host ""
    Write-Host "==> Java tests skipped (mvn not found)"
  }
}

Write-Host ""
Write-Host "All selected test steps completed."

