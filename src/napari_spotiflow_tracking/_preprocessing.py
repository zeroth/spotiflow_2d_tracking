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


def remove_background_high_pass(
    image: np.ndarray,
    sigma: float = 10.0,
    device: str = "cpu",
) -> np.ndarray:
    """Remove background by subtracting a Gaussian low-pass filtered image.

    Uses PyTorch separable convolution (same approach as spotiflow's
    BackgroundRemover) for GPU acceleration. Falls back to scipy on CPU
    when torch is not available.

    Args:
        image: 2D image (Y, X).
        sigma: standard deviation of the Gaussian filter. Larger values
               remove broader background structures.
        device: 'cuda' for GPU, 'cpu' for CPU.

    Returns:
        Background-subtracted image (float64).
    """
    try:
        import torch
        import torch.nn.functional as F
        return _remove_background_torch(image, sigma, device, torch, F)
    except ImportError:
        from scipy.ndimage import gaussian_filter
        background = gaussian_filter(image.astype(np.float64), sigma=sigma)
        return image.astype(np.float64) - background


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

    return result.squeeze().cpu().numpy().astype(np.float64)


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
        Background-subtracted stack (float64).
    """
    if image_stack.ndim != 3:
        raise ValueError(f"Expected 3D (T,Y,X) stack, got {image_stack.ndim}D")

    try:
        import torch
        import torch.nn.functional as F
        return _remove_background_stack_torch(image_stack, sigma, device, torch, F)
    except ImportError:
        from scipy.ndimage import gaussian_filter
        result = np.empty_like(image_stack, dtype=np.float64)
        for t in range(image_stack.shape[0]):
            bg = gaussian_filter(image_stack[t].astype(np.float64), sigma=sigma)
            result[t] = image_stack[t].astype(np.float64) - bg
        return result


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

    result = torch.cat(results, dim=0).squeeze(1).numpy().astype(np.float64)
    return result


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
) -> np.ndarray:
    """Denoise an image using Noise2Void (CAREamics).

    If a model path is provided, loads a pretrained model.
    Otherwise, trains a new N2V model on the input image (self-supervised).

    Args:
        image: 2D image (Y, X) or 3D stack (T, Y, X) / volume (Z, Y, X).
        model_path: path to a pretrained CAREamics model (.zip or .ckpt).
                    If None, trains a new model on the input data.
        n_epochs: number of training epochs (only used when training).
        patch_size: patch size for training/prediction.
        mode: '2D' or '3D'.
              2D: for stacks, each frame is an independent sample (axes=SYX).
              3D: stack is treated as a single volume (axes=ZYX).
        train_frames: number of evenly spaced frames to train on.
                      None or 0 = use all frames. Useful for large stacks.

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

    # Tiling for prediction
    if is_3d and image.ndim == 3:
        tile_size = (min(patch_size, image.shape[0]), patch_size, patch_size)
        tile_overlap = (
            min(patch_size // 4, image.shape[0] // 2),
            patch_size // 4,
            patch_size // 4,
        )
    else:
        tile_size = (patch_size, patch_size)
        tile_overlap = (patch_size // 4, patch_size // 4)

    pred_kwargs = dict(
        source=image.astype(np.float32),
        data_type="array",
        dataloader_params={"num_workers": 0},
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    if image.ndim == 2:
        pred_kwargs["axes"] = "YX"
    elif image.ndim == 3:
        pred_kwargs["axes"] = "ZYX" if is_3d else "SYX"

    prediction = careamist.predict(**pred_kwargs)

    if isinstance(prediction, list):
        prediction = prediction[0]
    return np.squeeze(np.asarray(prediction))


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
