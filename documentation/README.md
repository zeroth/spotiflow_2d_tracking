# napari-spotiflow-tracking — documentation

HTML user documentation for the plugin, built with
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

## Quick commands

All commands assume a Python venv at `<repo>/.venv` with the dev dependencies
installed:

```powershell
.venv\Scripts\python.exe -m pip install -r documentation\requirements.txt
```

| Task | Command (Windows / PowerShell)                                                 |
|---|---|
| Capture all widget screenshots         | `documentation\scripts\build.ps1 -SkipShots:$false` (default) |
| Build static HTML to `documentation\site\` | `documentation\scripts\build.ps1` |
| Live-preview at <http://127.0.0.1:8000>  | `documentation\scripts\build.ps1 -Serve` |
| Build without re-capturing screenshots | `documentation\scripts\build.ps1 -SkipShots` |

macOS / Linux equivalents use `documentation/scripts/build.sh` with
`--skip-shots` / `--serve`.

## Direct invocation

```powershell
# Capture widget screenshots
.venv\Scripts\python.exe documentation\scripts\capture_widgets.py

# Build site
.venv\Scripts\python.exe -m mkdocs build -f documentation\mkdocs.yml

# Live preview
.venv\Scripts\python.exe -m mkdocs serve -f documentation\mkdocs.yml
```

## Layout

```
documentation/
├── mkdocs.yml              # Site config + nav
├── requirements.txt        # mkdocs + theme + pyyaml
├── scripts/
│   ├── capture_widgets.py  # Headless-ish napari widget screenshotter
│   ├── build.ps1           # Windows: capture + build / serve
│   └── build.sh            # macOS / Linux: capture + build / serve
├── docs/
│   ├── index.md
│   ├── installation.md
│   ├── workflow.md
│   ├── maintaining.md
│   ├── widgets/{preprocessing,detection,tracking,papa}.md
│   └── assets/screenshots/ # Auto-generated PNGs
└── site/                   # Built HTML output (gitignored)
```

See [docs/maintaining.md](docs/maintaining.md) for the full workflow,
including how to add a new widget.
