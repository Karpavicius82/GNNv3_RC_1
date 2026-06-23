# Build and run every contract test in tools/, then print a pass/fail summary.
# Usage (from repo root, in a shell that can find MSVC, or it will locate vcvars):
#   powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# locate vcvars64.bat
$vcvars = Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $vcvars) { Write-Error "vcvars64.bat not found"; exit 1 }

$build = Join-Path $root "build"
New-Item -ItemType Directory -Force -Path $build | Out-Null
$tests = Get-ChildItem "$root\tools\*.cpp" | Sort-Object Name

$pass = 0; $fail = 0; $broken = 0
foreach ($t in $tests) {
  $name = $t.BaseName
  $exe  = Join-Path $build "$name.exe"
  $bat  = @"
call "$($vcvars.FullName)" >nul 2>&1
cl /nologo /O2 /EHsc /std:c++20 /I "$root\tools" "$($t.FullName)" /Fe:"$exe" /Fo:"$build\\" >nul 2>"$build\err.log"
"@
  $bf = Join-Path $build "b.bat"; Set-Content $bf $bat -Encoding ascii
  cmd /c $bf | Out-Null
  if (-not (Test-Path $exe)) { Write-Host ("BUILD-FAIL  {0}" -f $name) -ForegroundColor Yellow; $broken++; continue }
  $out = & $exe 2>&1
  $res = ($out | Select-String "RESULT" | Select-Object -Last 1).ToString().Trim()
  if ($LASTEXITCODE -eq 0) { Write-Host ("PASS  {0,-50} {1}" -f $name, $res) -ForegroundColor Green; $pass++ }
  else { Write-Host ("FAIL  {0,-50} {1}" -f $name, $res) -ForegroundColor Red; $fail++ }
}
Write-Host ""
Write-Host ("==== PASS=$pass  FAIL=$fail  BUILD-FAIL=$broken  (of $($tests.Count)) ====") -ForegroundColor Cyan
if ($fail -gt 0 -or $broken -gt 0) { exit 1 }
