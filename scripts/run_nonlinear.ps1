# run_nonlinear.ps1 -- production runner for the NONLINEAR (Kerr) engine tests.
# Auto-locates MSVC, builds and runs every nonlinear test, checks pass/fail, and
# exits 0 only if all pass. No Developer prompt needed.
#   powershell -ExecutionPolicy Bypass -File scripts\run_nonlinear.ps1
[CmdletBinding()] param([string]$VcVars)

$ErrorActionPreference = 'Stop'
$root  = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root 'build\nonlinear'
New-Item -ItemType Directory -Force -Path $build | Out-Null

# --- 1. locate vcvars64.bat (x64) --------------------------------------------
if (-not $VcVars -or -not (Test-Path $VcVars)) {
  $vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
  if ($vs) { $VcVars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat' }
  if (-not $VcVars -or -not (Test-Path $VcVars)) {
    $VcVars = (Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat" -EA SilentlyContinue | Select-Object -First 1).FullName
  }
}
if (-not $VcVars -or -not (Test-Path $VcVars)) { Write-Host "FATAL: vcvars64.bat not found (pass -VcVars <path>)" -ForegroundColor Red; exit 2 }
Write-Host "compiler env: $VcVars" -ForegroundColor DarkGray

# --- 2. import the MSVC environment ONCE (so cl is on PATH for all builds) ----
# Hardened against a PATH/Path case-duplicate (seen in container/CI envs): we UNION
# every PATH-like entry case-insensitively, so the VC tools dir survives regardless
# of which casing or dump order wins. (Reported by an independent Codex run.)
$pathParts = @()
cmd /c "call `"$VcVars`" >nul 2>&1 && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') {
    $k = $matches[1]; $v = $matches[2]
    if ($k -ieq 'PATH') { $pathParts += ($v -split ';') } else { Set-Item -Path "env:$k" -Value $v }
  }
}
if ($pathParts.Count) {
  $seen = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
  $env:Path = (($pathParts | Where-Object { $_ -and $seen.Add($_) }) -join ';')
}
if (-not (Get-Command cl -EA SilentlyContinue)) { Write-Host "FATAL: cl still not on PATH after vcvars (check VC x64 tools installed)" -ForegroundColor Red; exit 2 }

# --- 3. the nonlinear test set (file, kind, success check) --------------------
# contracts return exit 0 on full pass; probes are checked by a metric in output.
$tests = @(
  @{ name='graph_wave_horizon_densification_contract_test';  dir='tools';    kind='contract' }
  @{ name='graph_wave_streaming_densification_contract_test'; dir='tools';   kind='contract' }
  @{ name='graph_wave_nonlinear_compute_contract_test';      dir='tools';    kind='contract' }
  @{ name='probe_streaming_compression';                     dir='research'; kind='probe'; needs='compression='; minx=1.5 }
  @{ name='probe_nonlinear_soliton';                         dir='research'; kind='probe' }
  @{ name='probe_blackhole_compress';                        dir='research'; kind='probe' }
)

$pass = 0; $fail = 0; $results = @()
foreach ($t in $tests) {
  $src = Join-Path $root "$($t.dir)\$($t.name).cpp"
  $exe = Join-Path $build "$($t.name).exe"
  if (-not (Test-Path $src)) { $results += [pscustomobject]@{ test=$t.name; status='MISSING'; detail=$src }; $fail++; continue }

  $errlog = Join-Path $build "$($t.name).err"
  cl /nologo /O2 /EHsc /std:c++20 /I "$root\tools" $src /Fe:$exe /Fo:"$build\\" 1>$null 2>$errlog
  if (-not (Test-Path $exe)) { $results += [pscustomobject]@{ test=$t.name; status='BUILD-FAIL'; detail=(Get-Content $errlog -Tail 1) }; $fail++; continue }

  Push-Location $root
  $out = & $exe 2>&1 | Out-String
  $code = $LASTEXITCODE
  Pop-Location

  $ok = $true; $detail = ''
  if ($t.kind -eq 'contract') {
    $ok = ($code -eq 0)
    $m = ([regex]'RESULT\s*:\s*(\d+)\s*/\s*(\d+)').Match($out); if ($m.Success) { $detail = "$($m.Groups[1].Value)/$($m.Groups[2].Value)" }
  } else {
    $ok = ($code -eq 0)
    if ($t.needs) { $ok = $ok -and ($out -match [regex]::Escape($t.needs)) }
    $cm = ([regex]'compression=([\d.]+)x').Match($out)
    if ($cm.Success) { $detail = "compression=$($cm.Groups[1].Value)x"; if ($t.minx) { $ok = $ok -and ([double]$cm.Groups[1].Value -ge $t.minx) } }
  }
  $results += [pscustomobject]@{ test=$t.name; status=$(if ($ok) {'PASS'} else {'FAIL'}); detail=$detail }
  if ($ok) { $pass++ } else { $fail++ }
}

# --- 4. summary --------------------------------------------------------------
Write-Host ''
$results | ForEach-Object {
  $c = switch ($_.status) { 'PASS' {'Green'} default {'Red'} }
  Write-Host ("  {0,-7} {1,-48} {2}" -f $_.status, $_.test, $_.detail) -ForegroundColor $c
}
Write-Host ''
Write-Host ("==== NONLINEAR: PASS=$pass FAIL=$fail (of $($tests.Count)) ====") -ForegroundColor Cyan
exit ($(if ($fail -eq 0) { 0 } else { 1 }))
