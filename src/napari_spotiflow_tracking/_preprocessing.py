from __future__ import annotations

import numpy as np


def _get_device(use_gpu: bool = False) -> str:
    """Return 'cuda' if requested and available, else 'cpu'."""
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
    return "cpu"


def _rebaseline(residual: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Shift a zero-centred residual onto the original's mean, in its dtype.

    Subtracting the low-pass background leaves the result centred on zero, so
    roughly half of it is negative. Adding the original mean back puts the
    signal on a positive baseline again.

    Integer dtypes are clipped to their representable range before the cast:
    a residual of -3 written straight into ``uint16`` wraps to 65533, turning
    the darkest pixels into the brightest in the frame. Floats are cast
    without clipping, so they keep their sign.
    """
    out = residual + float(np.mean(original))
    if np.issubdtype(original.dtype, np.integer):
        info = np.iinfo(original.dtype)
        out = np.clip(out, info.min, info.max)
    return out.astype(original.dtype)


def _rebaseline_stack(residual: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Per-frame :func:`_rebaseline`, so a stack matches frame-by-frame 2D."""
    out = np.empty(original.shape, dtype=original.dtype)
    for t in range(original.shape[0]):
        out[t] = _rebaseline(residual[t], original[t])
    return out


def remove_background_high_pass(
    image: np.ndarray,
    sigma: float = 10.0,
    device: str = "cpu",
) -> np.ndarray:
    """Remove background by subtracting a Gaussian low-pass filtered image.

    Uses PyTorch separable convolution (same approach as spotiflow's
    BackgroundRemover) for GPU acceleration. Falls back to scipy on CPU
    when torch is not available.

    The subtraction leaves the result centred on zero, so the image's own mean
    is added back to return it to a positive baseline (see :func:`_rebaseline`).

    Args:
        image: 2D image (Y, X).
        sigma: standard deviation of the Gaussian filter. Larger values
               remove broader background structures.
        device: 'cuda' for GPU, 'cpu' for CPU.

    Returns:
        Background-subtracted image, re-baselined on the input mean and cast
        back to the input dtype. Integer results are clipped to the dtype's
        range.
    """
    try:
        import torch
        import torch.nn.functional as F
        return _remove_background_torch(image, sigma, device, torch, F)
    except ImportError:
        from scipy.ndimage import gaussian_filter
        background = gaussian_filter(image.astype(np.float64), sigma=sigma)
        return _rebaseline(image.astype(np.float64) - background, image)


def _remove_background_torch(image, sigma, device, torch, F):
    """Torch-based separable Gaussian background subtraction."""
    # Build 1D Gaussian kernel (span -2σ to +2σ like spotiflow)
    radius = max(int(np.ceil(sigma * 2)), 1)
    half = radius
    kernel_size = 2 * half + 1
    t = torch.linspace(-2, 2, kernel_size, dtype=torch.float32)
    h = torch.exp(-t ** 2)
    h = h / h.sum()

    wy = h.reshape(1, 1, kernel_size, 1)
    wx = h.reshape(1, 1, 1, kernel_size)

    # Image to tensor (1, 1, H, W)
    x = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    if device == "cuda":
        x = x.cuda()
        wy = wy.cuda()
        wx = wx.cuda()

    # Separable convolution with reflect padding
    y = F.pad(x, pad=(half, half, half, half), mode="reflect")
    y = F.conv2d(y, weight=wy, stride=1, padding="valid")
    y = F.conv2d(y, weight=wx, stride=1, padding="valid")
    result = x - y

    residual = result.squeeze().cpu().numpy().astype(np.float64)
    return _rebaseline(residual, image)


def remove_background_high_pass_stack(
    image_stack: np.ndarray,
    sigma: float = 10.0,
    device: str = "cpu",
) -> np.ndarray:
    """Remove background from a T,Y,X stack using batch processing.

    Processes all frames in a single batched torch operation for speed.
    Falls back to per-frame scipy if torch is unavailable.

    Args:
        image_stack: 3D array (T, Y, X).
        sigma: Gaussian filter sigma.
        device: 'cuda' or 'cpu'.

    Returns:
        Background-subtracted stack, each frame re-baselined on its own mean
        and cast back to the input dtype — so the result matches applying
        :func:`remove_background_high_pass` to each frame in turn.
    """
    if image_stack.ndim != 3:
        raise ValueError(f"Expected 3D (T,Y,X) stack, got {image_stack.ndim}D")

    try:
        import torch
        import torch.nn.functional as F
        return _remove_background_stack_torch(image_stack, sigma, device, torch, F)
    except ImportError:
        from scipy.ndimage import gaussian_filter
        residual = np.empty(image_stack.shape, dtype=np.float64)
        for t in range(image_stack.shape[0]):
            bg = gaussian_filter(image_stack[t].astype(np.float64), sigma=sigma)
            residual[t] = image_stack[t].astype(np.float64) - bg
        return _rebaseline_stack(residual, image_stack)


def _remove_background_stack_torch(image_stack, sigma, device, torch, F):
    """Batch torch-based background subtraction for entire stack."""
    radius = max(int(np.ceil(sigma * 2)), 1)
    half = radius
    kernel_size = 2 * half + 1
    t = torch.linspace(-2, 2, kernel_size, dtype=torch.float32)
    h = torch.exp(-t ** 2)
    h = h / h.sum()

    wy = h.reshape(1, 1, kernel_size, 1)
    wx = h.reshape(1, 1, 1, kernel_size)

    # Stack to tensor (T, 1, H, W)
    x = torch.from_numpy(image_stack.astype(np.float32)).unsqueeze(1)

    if device == "cuda":
        x = x.cuda()
        wy = wy.cuda()
        wx = wx.cuda()

    # Process in batches to avoid OOM on large stacks
    batch_size = 64
    n_frames = x.shape[0]
    results = []

    for i in range(0, n_frames, batch_size):
        batch = x[i:i + batch_size]
        y = F.pad(batch, pad=(half, half, half, half), mode="reflect")
        y = F.conv2d(y, weight=wy, stride=1, padding="valid")
        y = F.conv2d(y, weight=wx, stride=1, padding="valid")
        results.append((batch - y).cpu())

    residual = torch.cat(results, dim=0).squeeze(1).numpy().astype(np.float64)
    return _rebaseline_stack(residual, image_stack)


# ── Morphological erosion/reconstruction background removal ──────────


def remove_background_erosion(
    image: np.ndarray,
    disk_size: int = 10,
) -> np.ndarray:
    """Remove background using morphological reconstruction.

    Erodes the image with a disk structuring element, reconstructs by
    dilation, and subtracts the reconstructed background from the original.

    Args:
        image: 2D image (Y, X).
        disk_size: radius of the disk structuring element for erosion.

    Returns:
        Background-subtracted image (float64).
    """
    from skimage import morphology

    seed = morphology.erosion(image, morphology.disk(disk_size))
    background = morphology.reconstruction(seed, image, method="dilation")
    return image.astype(np.float64) - background


# ── Noise2Void denoising ─────────────────────────────────────────────


def _select_train_frames(image: np.ndarray, n_frames: int) -> np.ndarray:
    """Select n_frames evenly spaced frames from a stack for training."""
    total = image.shape[0]
    if n_frames >= total:
        return image
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    indices = list(dict.fromkeys(indices))
    return image[indices]


def denoise_n2v(
    image: np.ndarray,
    model_path: str | None = None,
    n_epochs: int = 100,
    patch_size: int = 64,
    mode: str = "2D",
    train_frames: int | None = None,
    predict_image: np.ndarray | None = None,
) -> np.ndarray:
    """Denoise an image using Noise2Void (CAREamics).

    If a model path is provided, loads a pretrained model.
    Otherwise, trains a new N2V model on the input image (self-supervised).

    Args:
        image: 2D image (Y, X) or 3D stack (T, Y, X) / volume (Z, Y, X).
               Used for training (and prediction if predict_image is None).
        model_path: path to a pretrained CAREamics model (.zip or .ckpt).
                    If None, trains a new model on the input data.
        n_epochs: number of training epochs (only used when training).
        patch_size: patch size for training/prediction.
        mode: '2D' or '3D'.
              2D: for stacks, each frame is an independent sample (axes=SYX).
              3D: stack is treated as a single volume (axes=ZYX).
        train_frames: number of evenly spaced frames to train on.
                      None or 0 = use all frames. Useful for large stacks.
        predict_image: if provided, train on `image` but predict on this array.
                       Useful for preview (train on one frame, predict on another).

    Returns:
        Denoised image with same shape as input.
    """
    from careamics import CAREamist
    from careamics.config import create_n2v_configuration

    is_3d = mode.upper() == "3D"

    if model_path is not None:
        careamist = CAREamist(model_path)
    else:
        # Select training subset if requested
        if train_frames and train_frames > 0 and image.ndim >= 3:
            train_data = _select_train_frames(image, train_frames)
        else:
            train_data = image

        # Determine axes from training data shape
        if train_data.ndim == 2:
            axes = "YX"
        elif train_data.ndim == 3:
            axes = "ZYX" if is_3d else "SYX"
        else:
            raise ValueError(f"Expected 2D or 3D image, got {train_data.ndim}D")

        # Patch size: 3D needs a Z patch dimension
        if is_3d and train_data.ndim == 3:
            ps = [min(patch_size, train_data.shape[0]), patch_size, patch_size]
        else:
            ps = [patch_size, patch_size]

        config = create_n2v_configuration(
            experiment_name="n2v_denoise",
            data_type="array",
            axes=axes,
            patch_size=ps,
            batch_size=1,
            num_epochs=n_epochs,
            train_dataloader_params={"shuffle": True, "num_workers": 0},
            val_dataloader_params={"num_workers": 0},
        )
        careamist = CAREamist(config)
        careamist.train(train_source=train_data.astype(np.float32))

    # Use predict_image if provided, otherwise predict on training image
    pred_img = predict_image if predict_image is not None else image

    # Tiling for prediction — overlap must be >= half the tile size
    # to avoid boundary artifacts from the UNet's receptive field
    if is_3d and pred_img.ndim == 3:
        tile_size = (min(patch_size, pred_img.shape[0]), patch_size, patch_size)
        tile_overlap = (
            min(patch_size // 2, pred_img.shape[0] // 2),
            patch_size // 2,
            patch_size // 2,
        )
    else:
        tile_size = (patch_size, patch_size)
        tile_overlap = (patch_size // 2, patch_size // 2)

    pred_kwargs = dict(
        source=pred_img.astype(np.float32),
        data_type="array",
        dataloader_params={"num_workers": 0},
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    if pred_img.ndim == 2:
        pred_kwargs["axes"] = "YX"
    elif pred_img.ndim == 3:
        pred_kwargs["axes"] = "ZYX" if is_3d else "SYX"

    prediction = careamist.predict(**pred_kwargs)

    if isinstance(prediction, list):
        # CAREamics returns a list of arrays (one per sample for SYX)
        prediction = np.stack([np.squeeze(p) for p in prediction], axis=0)
    else:
        prediction = np.squeeze(np.asarray(prediction))

    # If prediction input was 2D, squeeze out any extra dims
    if pred_img.ndim == 2:
        prediction = np.squeeze(prediction)

    return prediction


# ── Walking average ──────────────────────────────────────────────────


def walking_average(image_stack: np.ndarray, window: int = 3) -> np.ndarray:
    """Apply a walking (rolling) average along the time axis.

    For a T,Y,X stack, each frame is replaced by the mean of the surrounding
    *window* frames (centered). Frames near the edges use a truncated window.

    Args:
        image_stack: 3D array (T, Y, X).
        window: number of frames in the averaging window (must be odd).

    Returns:
        Smoothed stack with same shape and dtype as input.
    """
    if image_stack.ndim != 3:
        raise ValueError(f"Expected 3D (T,Y,X) stack, got {image_stack.ndim}D")
    if window < 1:
        raise ValueError(f"Window must be >= 1, got {window}")
    if window % 2 == 0:
        window += 1  # ensure odd

    n_frames = image_stack.shape[0]
    half = window // 2
    result = np.empty_like(image_stack, dtype=np.float64)

    # Cumulative sum for efficient rolling mean
    cumsum = np.cumsum(image_stack.astype(np.float64), axis=0)
    cumsum = np.concatenate([np.zeros_like(cumsum[:1]), cumsum], axis=0)

    for t in range(n_frames):
        lo = max(t - half, 0)
        hi = min(t + half + 1, n_frames)
        result[t] = (cumsum[hi] - cumsum[lo]) / (hi - lo)

    return result.astype(image_stack.dtype)


# ── Gaussian blur ─────────────────────────────────────────────────────


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a per-frame Gaussian blur.

    For a 2D ``(Y, X)`` image a single Gaussian filter is applied. For a 3D
    ``(T, Y, X)`` stack each frame is blurred independently (no smoothing
    across the time axis). The computation is done in float64 and the result
    is returned as float64.

    Args:
        image: 2D ``(Y, X)`` image or 3D ``(T, Y, X)`` stack.
        sigma: standard deviation of the Gaussian kernel, in pixels. A
            non-positive value returns the input unchanged.

    Returns:
        Blurred array (float64), or the input unchanged when ``sigma <= 0``.

    Raises:
        ValueError: if ``image`` is neither 2D nor 3D.
    """
    from scipy.ndimage import gaussian_filter

    if sigma <= 0:
        return image
    if image.ndim == 2:
        return gaussian_filter(image.astype(np.float64), sigma=sigma).astype(image.dtype)
    if image.ndim == 3:
        frames = [
            gaussian_filter(image[t].astype(np.float64), sigma=sigma)
            for t in range(image.shape[0])
        ]
        return np.stack(frames, axis=0).astype(image.dtype)
    raise ValueError(f"Expected 2D (Y,X) or 3D (T,Y,X) image, got {image.ndim}D")


# ── Crop via Shapes layer rectangle ──────────────────────────────────


def first_rectangle_bbox(shapes_data, shape_types, image_shape):
    """Compute axis-aligned bbox of the first rectangle in a Shapes layer.

    Iterates through the shapes list, picks the first one whose type is
    ``"rectangle"``, takes the last two coordinate dims as (Y, X), and
    returns the AABB clipped to the image's Y/X bounds.

    Args:
        shapes_data: list of (N, D) corner arrays, as stored on
            ``napari.layers.Shapes.data``.
        shape_types: parallel list of shape-type strings, as stored on
            ``napari.layers.Shapes.shape_type``.
        image_shape: shape tuple of the target image. The last two entries
            must be ``(H, W)``.

    Returns:
        Tuple ``(y_min, y_max, x_min, x_max)`` suitable for slicing.

    Raises:
        ValueError: no rectangle is present, or the clipped bbox is empty.
    """
    h, w = image_shape[-2], image_shape[-1]
    for data, stype in zip(shapes_data, shape_types):
        if stype != "rectangle":
            continue
        corners = np.asarray(data, dtype=float)
        ys = corners[:, -2]
        xs = corners[:, -1]
        y_min = max(0, int(round(float(ys.min()))))
        y_max = min(int(h), int(round(float(ys.max()))))
        x_min = max(0, int(round(float(xs.min()))))
        x_max = min(int(w), int(round(float(xs.max()))))
        if y_max - y_min <= 0 or x_max - x_min <= 0:
            raise ValueError(
                "Crop region is empty (rectangle outside image bounds)."
            )
        return (y_min, y_max, x_min, x_max)
    raise ValueError("No rectangle found in shapes layer.")


def crop_image(image, bbox):
    """Slice an image using a ``(y_min, y_max, x_min, x_max)`` bbox.

    Works for both 2D ``(Y, X)`` and 3D ``(T, Y, X)`` inputs because the
    slice indexes only the last two dimensions.

    Args:
        image: 2D or 3D numpy array.
        bbox: tuple returned by :func:`first_rectangle_bbox`.

    Returns:
        Cropped array (a view into ``image``).
    """
    y_min, y_max, x_min, x_max = bbox
    return image[..., y_min:y_max, x_min:x_max]


# ── Scalar calculation (add / subtract) ──────────────────────────────


def apply_calculation(image: np.ndarray, op: str, value: int) -> np.ndarray:
    """Add or subtract an integer scalar from an image.

    Returns a new array with the *same dtype as the input*. For integer
    input dtypes, underflow and overflow are clipped to the dtype's
    representable range (e.g. a ``uint16`` image clipped to ``[0, 65535]``).
    For floating-point inputs the math is performed and cast back to the
    input dtype with no clipping.

    Args:
        image: 2D or 3D numpy array.
        op: ``"add"`` or ``"subtract"``.
        value: integer scalar (the widget restricts this to ``>= 0``; the
            helper itself accepts any int).

    Returns:
        New array of the same shape and dtype as ``image``. Always a copy
        — never a view.

    Raises:
        ValueError: if ``op`` is not one of ``"add"`` or ``"subtract"``.
    """
    if op not in ("add", "subtract"):
        raise ValueError(f"Unknown op: {op!r}")

    sign = 1 if op == "add" else -1

    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        # Compute in int64 to avoid intermediate overflow during the math.
        wide = image.astype(np.int64) + sign * int(value)
        clipped = np.clip(wide, info.min, info.max)
        return clipped.astype(image.dtype)

    # Floating-point input: do the math, then cast back to the input dtype.
    return (image + sign * value).astype(image.dtype, copy=True)
