# Maintaining these docs

This site is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).
All documentation source lives under `documentation/docs/` as plain Markdown.

```
documentation/
├── mkdocs.yml                  # Site config + navigation
├── requirements.txt            # mkdocs + theme
├── scripts/
│   ├── capture_widgets.py      # Auto-screenshots every plugin widget
│   ├── build.ps1               # Windows: capture + build
│   └── build.sh                # macOS / Linux: capture + build
└── docs/
    ├── index.md
    ├── installation.md
    ├── workflow.md
    ├── maintaining.md          # (this page)
    ├── widgets/                # One page per widget
    │   ├── preprocessing.md
    │   ├── detection.md
    │   ├── tracking.md
    │   └── papa.md
    └── assets/
        └── screenshots/        # Auto-generated PNGs (gitignored optional)
```

---

## Daily commands

All commands below assume you are in the **repository root** and that the
project virtual environment is `.venv`.

### Live-preview the site

```powershell
.venv\Scripts\python.exe -m mkdocs serve -f documentation/mkdocs.yml
```

Then open <http://127.0.0.1:8000> in a browser. Pages reload on save.

### Regenerate every widget screenshot

```powershell
.venv\Scripts\python.exe documentation/scripts/capture_widgets.py
```

A short-lived napari viewer window will pop up, open each widget in turn,
grab a PNG of the widget panel, and write to
`documentation/docs/assets/screenshots/`.

### Build the static HTML site

```powershell
.venv\Scripts\python.exe -m mkdocs build -f documentation/mkdocs.yml
```

HTML is written to `documentation/site/` (gitignored by default).

### One-shot: screenshots + build

```powershell
documentation/scripts/build.ps1              # Windows
./documentation/scripts/build.sh             # macOS / Linux
```

---

## Adding a new widget

When you register a new widget in
`src/napari_spotiflow_tracking/napari.yaml`, the docs system is set up so you
only need to add a markdown page and a nav entry:

1. **Add the widget** to `napari.yaml` as usual.
2. **Regenerate screenshots** — `capture_widgets.py` auto-discovers widgets
   from `napari.yaml`, so it will pick up the new one without any code
   changes and produce one PNG per tab (or a single PNG if the widget is not
   tabbed). The output filename is `<slug>.png` or
   `<slug>--<tab-slug>.png`.
3. **Write the markdown page** at
   `documentation/docs/widgets/<your-widget>.md`. Use the existing pages
   as templates — typically a one-paragraph purpose statement, then a
   table per group-box, then an *Outputs* section.
4. **Add the page to `mkdocs.yml`** under the `Widgets` section of `nav`.
5. **Preview** with `mkdocs serve`, then **build** to publish.

---

## Updating existing widget docs

When you change a widget's UI:

1. Run `capture_widgets.py` to refresh the PNGs.
2. Open the widget's markdown page and update any field tables / behavior
   notes that changed.
3. Preview, then build.

The auto-discovery means **renaming a widget's tab** automatically changes
the screenshot filename — so an old PNG may be orphaned. Delete it from
`documentation/docs/assets/screenshots/` and update the `![…](…)` reference
in the markdown.

---

## How the screenshot script works

`capture_widgets.py`:

- Reads `src/napari_spotiflow_tracking/napari.yaml` to discover widgets and
  the plugin's package name.
- Launches a real napari viewer (`napari.Viewer(show=True)`) so layouts are
  computed against actual window metrics.
- For each widget, calls
  `viewer.window.add_plugin_dock_widget(plugin_name, widget_name=...)`,
  resizes the dock to a consistent width (`DOCK_WIDTH = 440`), and grabs the
  dock widget content via `QWidget.grab()`.
- If the widget contains a `QTabWidget`, every tab is clicked in turn and
  captured to its own PNG. This is how Detection / Tracking / PAPA produce
  their multi-tab screenshots without any per-widget config.

Constants you might want to tweak (top of the file):

| Constant | Default | Purpose |
|---|---|---|
| `DOCK_WIDTH` | `440` | Width in pixels each dock is resized to before grab. |
| `SETTLE_MS` | `250` | How long the Qt event loop is pumped after each action so layout/painting catch up. |

---

## Why MkDocs (and not Sphinx)?

- **Markdown source** is easier for end-user docs (Sphinx prefers reST).
- **Material theme** ships first-class layout, search, dark mode, code
  copy buttons, admonitions, and tabs — no extra configuration.
- **`mkdocs serve` live reload** is one of the nicest writing experiences
  available.
- For an API-heavy project we'd want Sphinx + autodoc; here we're documenting
  a UI, so user-facing prose + screenshots dominates.

If the project ever grows API reference needs, MkDocs has
[`mkdocstrings`](https://mkdocstrings.github.io/) which can be added without
disturbing the existing content.
