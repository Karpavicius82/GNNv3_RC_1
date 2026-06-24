# Fast dev-loop test runner. Three wins over run_tests.ps1:
#   1. INCREMENTAL  -- only recompiles a .cpp whose source (or the shared substrate
#      header) is newer than its .exe. Unchanged tests are reused. A 1-2 file edit
#      goes from ~10 min to a few seconds.
#   2. PARALLEL     -- compiles the stale set across (cores-2) cl.exe processes.
#   3. SKIP-SLOW    -- skips the long-running *diagnostic* tests by default (they are
#      data-gathering, not pass/fail gates). Use -WithSlow to include them.
# Flags: -All forces a full rebuild; -WithSlow also builds/runs the diagnostics.
# The canonical, authoritative full check remains scripts\run_tests.ps1.
param([switch]$All, [switch]$WithSlow)
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$vcvars = (Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if (-not $vcvars) { Write-Error "vcvars64.bat not found"; exit 1 }
$build = Join-Path $root "build"; New-Item -ItemType Directory -Force -Path $build | Out-Null
$headerT = (Get-Item "$root\tools\graph_wave_substrate.hpp").LastWriteTimeUtc
$tests = Get-ChildItem "$root\tools\*.cpp" | Sort-Object Name

# ---- decide which need rebuilding (incremental) ----
$stale = @()
foreach ($t in $tests) {
  $exe = Join-Path $build "$($t.BaseName).exe"
  $need = $All -or -not (Test-Path $exe)
  if (-not $need) {
    $exeT = (Get-Item $exe).LastWriteTimeUtc
    if ($t.LastWriteTimeUtc -gt $exeT -or $headerT -gt $exeT) { $need = $true }
  }
  if ($need) { $stale += $t }
}
Write-Host ("incremental: {0} of {1} need rebuild" -f $stale.Count, $tests.Count) -ForegroundColor Cyan

# ---- compile the stale set in parallel (load MSVC env once) ----
if ($stale.Count -gt 0) {
  $envLines = cmd /c "`"$vcvars`" >nul 2>&1 && set"
  foreach ($line in $envLines) { if ($line -match '^(.*?)=(.*)$') { Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2] -ErrorAction SilentlyContinue } }
  $cores = [Math]::Max(1, [int]$env:NUMBER_OF_PROCESSORS - 2)
  $running = @()
  foreach ($t in $stale) {
    $exe = Join-Path $build "$($t.BaseName).exe"
    $a = "/nologo /O2 /EHsc /std:c++20 /I `"$root\tools`" `"$($t.FullName)`" /Fe:`"$exe`" /Fo:`"$build\\`""
    $running += Start-Process cl -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput "$build\$($t.BaseName).out" -RedirectStandardError "$build\$($t.BaseName).err"
    while (($running | Where-Object { -not $_.HasExited }).Count -ge $cores) { Start-Sleep -Milliseconds 80 }
  }
  $running | Wait-Process -ErrorAction SilentlyContinue
}

# ---- run sequentially and report (exit code is the gate; sequential = isolated/correct) ----
$pass = 0; $fail = 0; $broken = 0; $skip = 0
foreach ($t in $tests) {
  if (-not $WithSlow -and $t.BaseName -match 'diagnostic') { $skip++; continue }
  $exe = Join-Path $build "$($t.BaseName).exe"
  if (-not (Test-Path $exe)) { Write-Host ("BUILD-FAIL  {0}" -f $t.BaseName) -ForegroundColor Yellow; $broken++; continue }
  $null = & $exe 2>&1
  if ($LASTEXITCODE -eq 0) { $pass++ } else { Write-Host ("FAIL  {0}" -f $t.BaseName) -ForegroundColor Red; $fail++ }
}
Write-Host ""
Write-Host ("==== PASS=$pass  FAIL=$fail  BUILD-FAIL=$broken  SKIPPED-slow=$skip  (of $($tests.Count)) ====") -ForegroundColor Cyan
if ($fail -gt 0 -or $broken -gt 0) { exit 1 }
