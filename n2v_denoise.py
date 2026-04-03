"""Standalone Noise2Void denoising CLI script.

No dependencies on the napari plugin. Requires: careamics, tifffile, numpy.

Usage:
    # Train on 3 evenly spaced frames, predict on entire 600-frame stack
    python n2v_denoise.py input.tif output.tif --epochs 100 --train-frames 3

    # Train on all frames (default)
    python n2v_denoise.py input.tif output.tif --epochs 100

    # 3D mode (Z,Y,X volume)
    python n2v_denoise.py input.tif output.tif --mode 3D --epochs 50

    # Use a pretrained model (skip training entirely)
    python n2v_denoise.py input.tif output.tif --model path/to/model.zip

    # Adjust patch size and batch size
    python n2v_denoise.py input.tif output.tif --epochs 200 --patch-size 128 --batch-size 4
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


def _select_train_frames(image: np.ndarray, n_frames: int) -> np.ndarray:
    """Select n_frames evenly spaced frames from a stack for training.

    Args:
        image: 3D+ array where axis 0 is the time/sample dimension.
        n_frames: number of frames to select.

    Returns:
        Subset array with shape (n_frames, ...).
    """
    total = image.shape[0]
    if n_frames >= total:
        return image
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    # Remove duplicates while preserving order
    indices = list(dict.fromkeys(indices))
    print(f"Selected {len(indices)} training frames: {indices}")
    return image[indices]


def _get_prediction_params(image, patch_size, is_3d):
    """Compute tile_size, tile_overlap, and axes for prediction."""
    if is_3d and image.ndim >= 3:
        z_dim = image.shape[-3]
        tile_size = (min(patch_size, z_dim), patch_size, patch_size)
        tile_overlap = (
            min(patch_size // 4, max(z_dim // 2, 1)),
            patch_size // 4,
            patch_size // 4,
        )
    else:
        tile_size = (patch_size, patch_size)
        tile_overlap = (patch_size // 4, patch_size // 4)

    if image.ndim == 2:
        axes = "YX"
    elif image.ndim == 3:
        axes = "ZYX" if is_3d else "SYX"
    elif image.ndim == 4:
        axes = "SZYX" if is_3d else "STYX"
    else:
        axes = None

    return tile_size, tile_overlap, axes


def train_n2v(
    train_data: np.ndarray,
    n_epochs: int = 100,
    patch_size: int = 64,
    batch_size: int = 1,
    mode: str = "2D",
):
    """Train a N2V model on the given data.

    Returns the trained CAREamist instance.
    """
    from careamics import CAREamist
    from careamics.config import create_n2v_configuration

    is_3d = mode.upper() == "3D"

    if train_data.ndim == 2:
        axes = "YX"
    elif train_data.ndim == 3:
        axes = "ZYX" if is_3d else "SYX"
    elif train_data.ndim == 4:
        axes = "SZYX" if is_3d else "STYX"
    else:
        raise ValueError(f"Unsupported data dimensions: {train_data.ndim}D")

    if is_3d and train_data.ndim >= 3:
        z_dim = train_data.shape[-3]
        ps = [min(patch_size, z_dim), patch_size, patch_size]
    else:
        ps = [patch_size, patch_size]

    print(f"Training N2V — mode={mode}, axes={axes}, "
          f"shape={train_data.shape}, epochs={n_epochs}, "
          f"patch_size={ps}, batch_size={batch_size}")

    config = create_n2v_configuration(
        experiment_name="n2v_denoise",
        data_type="array",
        axes=axes,
        patch_size=ps,
        batch_size=batch_size,
        num_epochs=n_epochs,
        train_dataloader_params={"shuffle": True, "num_workers": 0},
        val_dataloader_params={"num_workers": 0},
    )
    careamist = CAREamist(config)
    careamist.train(train_source=train_data.astype(np.float32))
    return careamist


def predict_n2v(
    careamist,
    image: np.ndarray,
    patch_size: int = 64,
    mode: str = "2D",
    batch_frames: int = 50,
) -> np.ndarray:
    """Run N2V prediction on the full image/stack.

    For large 2D stacks, processes in batches of `batch_frames` to avoid OOM.

    Returns denoised array with same shape as input.
    """
    is_3d = mode.upper() == "3D"

    # For large 2D stacks, process in batches to avoid OOM
    if not is_3d and image.ndim == 3 and image.shape[0] > batch_frames:
        return _predict_batched(careamist, image, patch_size, batch_frames)

    tile_size, tile_overlap, axes = _get_prediction_params(image, patch_size, is_3d)
    print(f"Predicting — shape={image.shape}, tile_size={tile_size}, axes={axes}")

    pred_kwargs = dict(
        source=image.astype(np.float32),
        data_type="array",
        dataloader_params={"num_workers": 0},
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    if axes is not None:
        pred_kwargs["axes"] = axes

    prediction = careamist.predict(**pred_kwargs)

    if isinstance(prediction, list):
        prediction = prediction[0]
    return np.squeeze(np.asarray(prediction))


def _predict_batched(
    careamist, image: np.ndarray, patch_size: int, batch_frames: int
) -> np.ndarray:
    """Predict on a large 2D stack in batches to avoid OOM."""
    import gc

    n_frames = image.shape[0]
    tile_size = (patch_size, patch_size)
    tile_overlap = (patch_size // 4, patch_size // 4)
    result = np.empty_like(image, dtype=np.float32)

    for start in range(0, n_frames, batch_frames):
        end = min(start + batch_frames, n_frames)
        batch = image[start:end]
        print(f"Predicting frames {start}-{end-1} of {n_frames}...")

        pred = careamist.predict(
            source=batch.astype(np.float32),
            data_type="array",
            axes="SYX",
            dataloader_params={"num_workers": 0},
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        if isinstance(pred, list):
            pred = pred[0]
        result[start:end] = np.squeeze(np.asarray(pred))

        # Free memory between batches
        del pred
        gc.collect()

    return result


def _pad_to_divisible(image: np.ndarray, divisor: int = 16) -> np.ndarray:
    """Pad spatial dimensions to be divisible by divisor (for UNet compatibility)."""
    pad_widths = []
    for i, s in enumerate(image.shape):
        if i >= image.ndim - 2:  # only pad Y, X (last 2 dims)
            remainder = s % divisor
            pad_widths.append((0, (divisor - remainder) % divisor))
        else:
            pad_widths.append((0, 0))
    return np.pad(image, pad_widths, mode="reflect")


def save_model(careamist, save_path: str, input_image: np.ndarray, input_name: str):
    """Save trained model to BioImage Model Zoo format."""
    try:
        ref = input_image[:1] if input_image.ndim >= 3 else input_image
        ref = _pad_to_divisible(ref)
        careamist.export_to_bmz(
            path_to_archive=save_path,
            friendly_model_name="N2V Denoised Model",
            input_array=ref,
            authors=[{"name": "CellphyLab"}],
            general_description="Noise2Void denoising model trained with CAREamics",
            data_description=f"Trained on {input_name}",
        )
        print(f"Model saved: {save_path}")
    except Exception as e:
        print(f"Warning: could not save model: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Noise2Void denoising for microscopy images (TIFF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Input TIFF file")
    parser.add_argument("output", help="Output TIFF file (denoised)")
    parser.add_argument(
        "--model", default=None,
        help="Path to pretrained CAREamics model (.zip or .ckpt). "
             "Skips training and goes straight to prediction.",
    )
    parser.add_argument(
        "--mode", choices=["2D", "3D"], default="2D",
        help="2D: each frame is independent (SYX). 3D: stack is a volume (ZYX). Default: 2D",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--train-frames", type=int, default=None, metavar="N",
        help="Number of evenly spaced frames to train on (default: all frames). "
             "E.g. --train-frames 3 selects 3 frames spread across the stack.",
    )
    parser.add_argument(
        "--patch-size", type=int, default=64,
        help="Patch size for training/prediction (default: 64)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for training (default: 1)",
    )
    parser.add_argument(
        "--batch-frames", type=int, default=50,
        help="Number of frames to predict at once to avoid OOM (default: 50). "
             "Lower this if you run out of memory during prediction.",
    )

    args = parser.parse_args()

    import tifffile

    # ── Load ──────────────────────────────────────────────────────────
    print(f"Loading: {args.input}")
    image = tifffile.imread(args.input)
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")

    t0 = time.perf_counter()

    # ── Train or load model ───────────────────────────────────────────
    if args.model is not None:
        from careamics import CAREamist
        print(f"Loading pretrained model: {args.model}")
        careamist = CAREamist(args.model)
    else:
        # Select training subset
        if args.train_frames is not None and image.ndim >= 3:
            train_data = _select_train_frames(image, args.train_frames)
        else:
            train_data = image

        careamist = train_n2v(
            train_data,
            n_epochs=args.epochs,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            mode=args.mode,
        )

        # Auto-save model next to input file
        input_path = Path(args.input)
        n_train = args.train_frames if args.train_frames else image.shape[0] if image.ndim >= 3 else 1
        model_name = f"{input_path.stem}_n2v_{n_train}frames.zip"
        model_save_path = str(input_path.parent / model_name)
        save_model(careamist, model_save_path, image, args.input)

    train_time = time.perf_counter() - t0
    print(f"Training/loading complete in {train_time:.1f}s")

    # ── Predict on full stack ─────────────────────────────────────────
    t1 = time.perf_counter()
    denoised = predict_n2v(
        careamist, image,
        patch_size=args.patch_size,
        mode=args.mode,
        batch_frames=args.batch_frames,
    )
    pred_time = time.perf_counter() - t1
    print(f"Prediction complete in {pred_time:.1f}s")

    # ── Save output ───────────────────────────────────────────────────
    tifffile.imwrite(args.output, denoised.astype(image.dtype), bigtiff=True)
    print(f"Saved: {args.output}")
    print(f"Total time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
