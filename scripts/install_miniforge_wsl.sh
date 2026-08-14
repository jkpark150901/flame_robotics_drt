#!/bin/bash
# Install Miniforge (conda + mamba, conda-forge default channel) inside WSL.
set -euo pipefail

MINIFORGE_VERSION="Miniforge3-Linux-x86_64.sh"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/${MINIFORGE_VERSION}"
INSTALL_DIR="${MINIFORGE_PREFIX:-$HOME/miniforge3}"
DOWNLOAD_PATH="/tmp/${MINIFORGE_VERSION}"

if [ -d "$INSTALL_DIR" ]; then
    echo "Miniforge already installed at $INSTALL_DIR — skipping."
else
    echo "Downloading Miniforge installer..."
    curl -fsSL -o "$DOWNLOAD_PATH" "$MINIFORGE_URL"

    echo "Installing to $INSTALL_DIR..."
    bash "$DOWNLOAD_PATH" -b -p "$INSTALL_DIR"
    rm -f "$DOWNLOAD_PATH"
fi

echo "Initializing shell (bash)..."
"$INSTALL_DIR/bin/conda" init bash

# Disable auto-activation of the base environment on new shells.
"$INSTALL_DIR/bin/conda" config --set auto_activate_base false

echo
echo "Miniforge installed at: $INSTALL_DIR"
echo "Restart your shell (or run: source ~/.bashrc) to start using conda/mamba."
