# napari-spotiflow-tracking

A [napari](https://napari.org) plugin for 2D spot detection and particle tracking using [Spotiflow](https://github.com/weigertlab/spotiflow) and [trackpy](https://github.com/soft-matter/trackpy).

## Features

- **Spot Detection** -- detect spots in 2D images or T,Y,X stacks using pretrained Spotiflow models
- **2D Gaussian Fitting** -- fit elliptical Gaussians to each detected spot and paint labeled masks
- **Particle Tracking** -- link spots across frames with trackpy, displayed as napari Tracks layer
- **CSV Export** -- export detected blobs and tracked particles to CSV

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Install PyTorch

Install PyTorch **before** other dependencies. Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) and select the install command for your OS and CUDA version.

For example (CPU only):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Or with CUDA 12.4:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install napari

```bash
pip install "napari[all]"
```

### 4. Install Qt backend

napari needs a Qt backend. Install one via qtpy:

```bash
pip install qtpy
```


### 5. Install remaining dependencies

```bash
pip install spotiflow numpy pandas trackpy scikit-image scipy qtpy
```

### 6. Install the plugin

#### Option A: From Git (development mode)

```bash
git clone https://github.com/your-username/napari-spotiflow-tracking.git
cd napari-spotiflow-tracking
pip install -e .
```

#### Option B: Without Git

Download and extract the repository as a ZIP, then:

```bash
cd napari-spotiflow-tracking
pip install .
```

## Usage

Launch napari:

```bash
napari
```

The plugin registers two widgets under **Plugins > Spotiflow 2D Tracking**:

### Spot Detection

1. Open an image (2D or T,Y,X stack)
2. Select the image layer, choose a Spotiflow model, and adjust detection parameters
3. Click **Detect Spots**
4. Results appear as a Points layer and (optionally) a Labels mask layer
5. Use **Generate Mask from Points** to create masks from an existing Points layer
6. Use **Export Blobs to CSV** to save spot coordinates

### Spot Tracking

1. Select the Points layer produced by detection (must have frame, y, x columns)
2. Set search range and memory parameters
3. Click **Track**
4. Results appear as a napari Tracks layer
5. Use **Export Tracks to CSV** to save tracked particles

## Running Tests

```bash
pip install -e ".[testing]"
pytest
```
