# Typical workflow

A complete pipeline through the four widgets, in the order most users run them.

```
[ Raw image stack ]
        │
        ▼
┌────────────────────┐
│  Pre-Processing    │  crop · subtract · background · denoise · blur · avg
└────────────────────┘
        │  cleaned Image layer
        ▼
┌────────────────────┐
│  Spot Detection    │  Spotiflow / LoG · optional Gaussian fit · masks
└────────────────────┘
        │  "Detected Spots" Points layer  (+ "Spot Masks" Labels)
        ▼
┌────────────────────┐
│  Spot Tracking     │  trackpy · MSD · step detection · intensity
└────────────────────┘
        │  "Tracked Spots" Tracks layer  +  per-track CSV
        ▼
       (optional)
┌────────────────────┐
│  PAPA              │  cycle counts · group ratios
└────────────────────┘
```

## Step 1 — Pre-Processing

Open **Plugins → Spotiflow 2D Tracking → Pre-Processing**.

1. Pick the input Image layer.
2. (Optional) Crop with a Shapes layer rectangle.
3. (Optional) Subtract background mean (Calculation → Subtract → Mean → Apply),
   then use **High-pass Gaussian** to remove slowly-varying background.
4. (Optional) Denoise with Noise2Void if your signal is noisy.
5. (Optional) Gaussian blur and/or walking average to smooth temporally.

Each step produces a **new image layer** (suffixed with the operation), so you
can compare to the original and roll back by selecting an earlier layer.

## Step 2 — Spot Detection

Open **Plugins → Spotiflow 2D Tracking → Spot Detection**.

1. Pick the pre-processed Image layer.
2. Choose a method:
    - **Spotiflow** (recommended): pick a pretrained model or a custom one,
      tune *Probability threshold* (or check **Auto**), *Min distance*.
    - **LoG**: classical fallback, tune sigmas and threshold.
3. (Optional) Tick **Generate mask during detection** to fit a 2D Gaussian to
   each spot and produce a per-spot Labels layer with region properties.
4. Press **Detect Spots** (or **Quick Preview** for the current frame only).
5. Use the **Filter** panel to drop unwanted spots by any region property
   (area, intensity, FWHM, solidity, …).
6. Switch to the **Analysis** tab to see the filtered table and a
   spots-per-frame bar chart.
7. **Export Blobs to CSV** when you're happy.

## Step 3 — Spot Tracking

Open **Plugins → Spotiflow 2D Tracking → Spot Tracking**.

1. Pick the Points layer from Detection. (Optionally also the Image layer if
   you want per-vertex intensity sampling.)
2. Set **Search range** (max pixel jump between frames) and **Memory** (max
   frames a spot can disappear before the track is broken).
3. Press **Track** — a new Tracks layer **Tracked Spots** appears in napari.
4. In the **Analysis** tab:
    - Filter tracks by properties (length, mean speed, etc.).
    - Click a row to see per-track **MSD** (with anomalous-diffusion fit),
      **step detection**, and **intensity** plots.
5. **Export Tracks to CSV** to save the full per-vertex DataFrame.

## Step 4 — PAPA (optional)

For cyclical / alternating experiments only.

1. Open **Plugins → Spotiflow 2D Tracking → PAPA**.
2. Pick the detected Points layer.
3. Set your cycle pattern: *Group 1 size*, *Group 2 size*, *Skip frames*.
4. Label the groups (default: `DR`, `PAPA`).
5. Press **Plot** to see particles-per-frame and per-cycle bar charts.
6. **Export CSV** for the annotated per-frame table.

---

## Tips

- **Layer naming**: each widget adds new layers with descriptive suffixes —
  the chain of layers tells you the processing history.
- **Cancel buttons**: long-running operations (Detect, Track, Denoise, GPU fit)
  expose **Cancel** buttons and napari progress bars. Use them.
- **GPU**: tick **Use GPU (CUDA)** in Pre-Processing and Detection when
  available — Spotiflow inference and Gaussian fitting both benefit
  significantly.
- **Filtering is non-destructive**: the Filter panels only hide rows; the
  underlying Points / Tracks layers keep all data until you export.
