#!/usr/bin/env bash
# Regenerate widget screenshots and rebuild the documentation site.
#
# Usage:
#     documentation/scripts/build.sh                # screenshots + build
#     documentation/scripts/build.sh --skip-shots   # build only
#     documentation/scripts/build.sh --serve        # screenshots + live-serve
#
# Expects a Python venv at <repo>/.venv with mkdocs, mkdocs-material, pyyaml,
# napari, and the plugin itself installed.
set -euo pipefail

SKIP_SHOTS=0
SERVE=0
for arg in "$@"; do
    case "$arg" in
        --skip-shots) SKIP_SHOTS=1 ;;
        --serve)      SERVE=1 ;;
        -h|--help)
            sed -n '1,12p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOC_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$DOC_ROOT")"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PY="$REPO_ROOT/.venv/bin/python"
elif [[ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then
    PY="$REPO_ROOT/.venv/Scripts/python.exe"
else
    echo "Could not find a python executable in $REPO_ROOT/.venv" >&2
    exit 1
fi

MKDOCS_YML="$DOC_ROOT/mkdocs.yml"

if [[ "$SKIP_SHOTS" -eq 0 ]]; then
    echo "==> Capturing widget screenshots..."
    "$PY" "$SCRIPT_DIR/capture_widgets.py"
fi

if [[ "$SERVE" -eq 1 ]]; then
    echo "==> Serving site at http://127.0.0.1:8000 (Ctrl+C to stop)"
    "$PY" -m mkdocs serve -f "$MKDOCS_YML"
else
    echo "==> Building static site..."
    "$PY" -m mkdocs build -f "$MKDOCS_YML"
    echo "Built. Open: $DOC_ROOT/site/index.html"
fi
