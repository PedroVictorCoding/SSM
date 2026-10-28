#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

if ! command -v cargo >/dev/null 2>&1; then
    echo "Rust/Cargo is required. Install it from https://rustup.rs, then run this installer again." >&2
    exit 1
fi

backend=${1:-}
if [ -z "$backend" ] && [ -t 0 ]; then
    printf '%s\n' "Choose the backend to install:" "  1. CPU (portable default)" "  2. WGPU (AMD/Intel/NVIDIA GPU)" "  3. CUDA (NVIDIA only)"
    printf 'Backend [1]: '
    IFS= read -r choice
    case "${choice:-1}" in
        1) backend=ndarray ;;
        2) backend=wgpu ;;
        3) backend=cuda ;;
        *) echo "Invalid backend selection." >&2; exit 2 ;;
    esac
fi
backend=${backend:-ndarray}

case "$backend" in
    ndarray|wgpu|cuda) ;;
    *) echo "Backend must be ndarray, wgpu, or cuda." >&2; exit 2 ;;
esac

echo "Installing SAIL with the '$backend' backend..."
cargo install --path "$repo_root" --locked --force --no-default-features --features "$backend"

echo
echo "Installed. Run: sail"
if ! command -v sail >/dev/null 2>&1; then
    echo "If 'sail' is not found in a new terminal, add \$HOME/.cargo/bin to PATH."
fi
