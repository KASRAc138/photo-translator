#!/usr/bin/env bash
# Photo Translator -- put photos in input/, run this, collect output/.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d venv ]; then
    echo "First run: creating the environment."
    python3 -m venv venv
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r requirements.txt
fi
exec ./venv/bin/python PhotoTranslator.py "$@"
