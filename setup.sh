#!/usr/bin/env bash
# One-click environment setup for SpyRL.
#
#   conda create -n spyrl python=3.10 -y && conda activate spyrl
#   bash setup.sh
#
# Knobs:
#   PYTHON_BIN=python3.11      interpreter to install into (default: python3)
#   INSTALL_MODE=copy          non-editable install (default: editable)
#   INSTALL_FLASH_ATTN=0       skip flash-attn, which is slow to build (default: 1)
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_MODE="${INSTALL_MODE:-editable}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"

echo "==> Interpreter: $("${PYTHON_BIN}" --version) at $(command -v "${PYTHON_BIN}")"
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel

echo "==> Installing SpyRL (verl core + recipe dependencies)"
if [ "${INSTALL_MODE}" = "editable" ]; then
    "${PYTHON_BIN}" -m pip install -e ".[spyrl]"
else
    "${PYTHON_BIN}" -m pip install ".[spyrl]"
fi

if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then
    echo "==> Installing flash-attn (built against the torch that was just installed)"
    if ! "${PYTHON_BIN}" -m pip install flash-attn --no-build-isolation; then
        echo "!!  flash-attn failed to build. Training still works without it, just slower."
        echo "!!  Re-run with INSTALL_FLASH_ATTN=0 to skip this step."
    fi
fi

echo "==> Verifying the install"
"${PYTHON_BIN}" spyrl/check_install.py
