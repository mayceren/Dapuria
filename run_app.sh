#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -x .venv/bin/python ]]; then
    exec .venv/bin/python -m streamlit run app.py "$@"
fi
exec python3 -m streamlit run app.py "$@"
