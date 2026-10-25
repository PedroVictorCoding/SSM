# Native Windows benchmark; never writes to an existing run directory.
param(
    [Parameter(Mandatory=$true)][string]$Binary,
    [Parameter(Mandatory=$true)][string]$Out,
    [ValidateSet('train','stream-train','run')][string]$Mode = 'train',
    [string]$Model = '',
    [int]$Tokens = 128,
    [int]$Steps = 30,
    [int]$BatchSize = 8,
    [int]$SeqLen = 64,
    [int]$DModel = 64,
    [int]$NLayers = 3,
    [int]$DState = 8,
    [int]$VizEvery = 0,
    [switch]$Profile,
    [switch]$DeviceClip,
    [switch]$SkipMemory,
    [switch]$TreeScan
)
$ErrorActionPreference = 'Stop'
$Binary = (Resolve-Path $Binary).ProviderPath
$Out = [System.IO.Path]::GetFullPath($Out)
if (Test-Path $Out) { throw "Benchmark requires a new output directory: $Out" }
New-Item -ItemType Directory -Path $Out | Out-Null
$sailArgs = @($Mode, '--out', $Out, '--steps', "$Steps", '--batch-size', "$BatchSize",
    '--seq-len', "$SeqLen", '--d-model', "$DModel", '--n-layers', "$NLayers", '--d-state', "$DState",
    '--d-conv', '4', '--attn-layers', '1', '--n-heads', '4', '--mlp-ratio', '2',
    '--moe-experts', '4', '--moe-top-k', '1', '--lr', '0.0002', '--min-lr', '0.00005',
    '--warmup', '0', '--save-every', "$Steps", '--eval-every', "$Steps", '--eval-windows', '8',
    '--log-every', '5', '--viz-every', "$VizEvery", '--viz-max-values', '0',
    '--sample-chars', '0', '--seed', '42')
if ($Mode -eq 'train') { $sailArgs += @('--data', 'data/tinyshakespeare.txt') }
elseif ($Mode -eq 'stream-train') {
    $incoming = Join-Path $Out 'incoming'
    New-Item -ItemType Directory -Path $incoming | Out-Null
    $sailArgs += @('--initial-data', 'data/tinyshakespeare.txt', '--incoming', $incoming)
}
if ($Profile) { $sailArgs += '--profile' }
if ($DeviceClip) { $sailArgs += '--device-clip' }
if ($TreeScan) { $sailArgs += '--work-efficient-scan' }
if ($Mode -eq 'run') {
    $Model = (Resolve-Path $Model).ProviderPath
    $sailArgs = @('run', '--model', $Model, '--prompt', 'ROMEO:', '--tokens', "$Tokens", '--temperature', '0', '--seed', '42')
}
$quotedArgs = ($sailArgs | ForEach-Object { '"' + $_.Replace('"','\"') + '"' }) -join ' '
$timer = [System.Diagnostics.Stopwatch]::StartNew()
$process = Start-Process -FilePath $Binary -ArgumentList $quotedArgs -PassThru -NoNewWindow `
    -RedirectStandardOutput (Join-Path $Out 'stdout.log') -RedirectStandardError (Join-Path $Out 'stderr.log')
# Retain the native handle before Refresh/HasExited; otherwise Windows PowerShell
# can lose ExitCode when Start-Process observes a short-lived child terminating.
$retainedHandle = $process.Handle
$samples = @()
$counterError = $null
while (-not $process.HasExited) {
    if (-not $SkipMemory) { try {
        # Windows counters may update less often than this sampling interval.
        $counters = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory `
            -OperationTimeoutSec 5 -Filter "Name LIKE 'pid_$($process.Id)[_]%'" -ErrorAction Stop)
        $dedicated = ($counters | Measure-Object -Property DedicatedUsage -Sum).Sum
        $shared = ($counters | Measure-Object -Property SharedUsage -Sum).Sum
        $samples += [pscustomobject]@{ seconds=$timer.Elapsed.TotalSeconds; dedicated_bytes=[double]$dedicated; shared_bytes=[double]$shared; instances=$counters.Count }
    } catch { $counterError = $_.Exception.Message } }
    Start-Sleep -Milliseconds 250
    $process.Refresh()
}
$process.WaitForExit()
$timer.Stop()
$samples | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $Out 'memory_samples.json') -Encoding UTF8
$metrics = @()
if (Test-Path (Join-Path $Out 'metrics.csv')) { $metrics = @(Import-Csv (Join-Path $Out 'metrics.csv')) }
$rates = @($metrics | Where-Object { $_.tokens_per_sec } | ForEach-Object { [double]$_.tokens_per_sec })
# Retain all windows; the first includes cold kernels. Report warm windows separately.
$warmRates = @($rates | Select-Object -Skip 1)
$inference = [regex]::Match((Get-Content (Join-Path $Out 'stdout.log') -Raw), 'prefill ([\d.]+) ms \| ([\d.]+) tok/s')
$workTokens = if ($Mode -eq 'run') { $Tokens } else { $Steps * $BatchSize * $SeqLen }
$summary = [ordered]@{
    binary=$Binary; arguments=$sailArgs; exit_code=$process.ExitCode;
    process_wall_seconds=$timer.Elapsed.TotalSeconds;
    end_to_end_tok_s=if ($process.ExitCode -eq 0) { $workTokens / $timer.Elapsed.TotalSeconds } else { $null };
    inference_prefill_ms=if ($inference.Success) { [double]$inference.Groups[1].Value } else { $null };
    inference_decode_tok_s=if ($inference.Success) { [double]$inference.Groups[2].Value } else { $null };
    all_window_tok_s=$rates; warm_mean_tok_s=($warmRates | Measure-Object -Average).Average;
    peak_sampled_dedicated_bytes=($samples | Measure-Object -Property dedicated_bytes -Maximum).Maximum;
    peak_sampled_shared_bytes=($samples | Measure-Object -Property shared_bytes -Maximum).Maximum;
    memory_counter_error=$counterError; memory_samples=$samples.Count; memory_sampling_disabled=[bool]$SkipMemory;
    device_request=$env:CUBECL_WGPU_DEFAULT_DEVICE; timestamp_utc=[DateTime]::UtcNow.ToString('o');
    os=[Environment]::OSVersion.VersionString; logical_processors=[Environment]::ProcessorCount;
    binary_sha256=(Get-FileHash $Binary -Algorithm SHA256).Hash;
    dataset_sha256=(Get-FileHash 'data/tinyshakespeare.txt' -Algorithm SHA256).Hash;
    model_sha256=if ($Mode -eq 'run') { (Get-FileHash $Model -Algorithm SHA256).Hash } else { $null };
    validation=@($metrics | Where-Object { $_.val_loss } | ForEach-Object { [double]$_.val_loss })
}
$summary | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Out 'benchmark.json') -Encoding UTF8
$summary | ConvertTo-Json -Depth 5
if ($process.ExitCode -ne 0) { throw "Sail command failed; see $Out\stderr.log" }
