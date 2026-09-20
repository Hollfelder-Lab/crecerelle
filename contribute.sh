#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -f setup.py || ! -f requirements-docs.txt ]]; then
    echo "Place contribute.sh in the Crecerelle repository root." >&2
    exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
    if [[ -n "${PYTHON:-}" ]]; then
        python_bin="$PYTHON"
    elif command -v python3.12 >/dev/null 2>&1; then
        python_bin=python3.12
    else
        python_bin=python3
    fi
    "$python_bin" -c 'import sys; sys.exit("Python 3.12 or newer is required; Python 3.12 is recommended.") if sys.version_info < (3, 12) else None'
    "$python_bin" -m venv .venv
fi

.venv/bin/python -c 'import sys; sys.exit("The existing .venv requires Python 3.12 or newer.") if sys.version_info < (3, 12) else None'
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e . -r requirements-docs.txt

echo "Development environment ready. Activate it with: source .venv/bin/activate"
echo "Run checks: python -m pytest tests/test_notebook_smoke.py tests/test_public_api_smoke.py"
echo "Build documentation: python -m mkdocs build --strict"
