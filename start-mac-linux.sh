#!/usr/bin/env bash
# TabForge one-click launcher: installs the free local helper and opens the app.
set -e
cd "$(dirname "$0")"
PY=python3
command -v $PY >/dev/null || { echo "Python 3.10+ is required: https://www.python.org/downloads/"; exit 1; }

echo "Installing / updating the helper (first run takes a moment)..."
$PY -m pip install --quiet --upgrade -r helper/requirements.txt

if [ ! -f helper/.separation-asked ]; then
  read -r -p "Also install AI stem separation (better accuracy, ~2 GB download)? [y/N] " SEP
  touch helper/.separation-asked
  if [[ "$SEP" =~ ^[Yy]$ ]]; then
    $PY -m pip install --quiet -r helper/requirements-separation.txt
    $PY -m pip install --quiet --no-deps demucs==4.0.1
  fi
fi

exec $PY helper/tabforge_helper.py
