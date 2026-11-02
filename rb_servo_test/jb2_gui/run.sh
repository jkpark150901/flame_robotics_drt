#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [[ -n "${PYTHON:-}" ]]; then
    interpreter="$PYTHON"
elif [[ -x ../../venv/bin/python ]]; then
    interpreter=../../venv/bin/python
else
    interpreter=python3
fi
exec env -u LD_LIBRARY_PATH "$interpreter" app.py "$@"
