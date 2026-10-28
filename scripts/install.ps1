param(
    [ValidateSet("ndarray", "wgpu", "cuda")]
    [string]$Backend
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw "Rust/Cargo is required. Install it from https://rustup.rs, restart PowerShell, and run this installer again."
}

if (-not $Backend) {
    Write-Host "Choose the backend to install:"
    Write-Host "  1. CPU (portable default)"
    Write-Host "  2. WGPU (AMD/Intel/NVIDIA GPU)"
    Write-Host "  3. CUDA (NVIDIA only)"
    $choice = Read-Host "Backend [1]"
    switch ($choice) {
        ""  { $Backend = "ndarray" }
        "1" { $Backend = "ndarray" }
        "2" { $Backend = "wgpu" }
        "3" { $Backend = "cuda" }
        default { throw "Invalid backend selection." }
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Write-Host "Installing SAIL with the '$Backend' backend..."
& cargo install --path $repoRoot --locked --force --no-default-features --features $Backend
if ($LASTEXITCODE -ne 0) {
    throw "cargo install failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Installed. Open a new terminal and run: sail"
if (-not (Get-Command sail -ErrorAction SilentlyContinue)) {
    Write-Host "If 'sail' is not found, add `$env:USERPROFILE\.cargo\bin to your user PATH."
}
