"""Standalone Noise2Void denoising CLI script.

No dependencies on the napari plugin. Requires: careamics, tifffile, numpy.

Usage:
    # Train from scratch and denoise (2D mode, T,Y,X stack)
    python n2v_denoise.py input.tif output.tif --epochs 100

    # 3D mode (Z,Y,X volume)
    python n2v_denoise.py input.tif output.tif --mode 3D --epochs 50

    # Use a pretrained model
    python n2v_denoise.py input.tif output.tif --model path/to/model.zip

    # Adjust patch size and batch size
    python n2v_denoise.py input.tif output.tif --epochs 200 --patch-size 128 --batch-size 4

    # Save the trained model for reuse
    python n2v_denoise.py input.tif output.tif --epochs 100 --save-model model_output.zip
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def denoise_n2v(
    image: np.ndarray,
    model_path: str | None = None,
    n_epochs: int = 100,
    patch_size: int = 64,
    batch_size: int = 1,
    mode: str = "2D",
) -> tuple[np.ndarray, object]:
    """Denoise an image using Noise2Void (CAREamics).

    Args:
        image: 2D (Y,X), 3D (T,Y,X or Z,Y,X), or 4D (T,Z,Y,X) array.
        model_path: path to pretrained model (.zip or .ckpt). None = train from scratch.
        n_epochs: training epochs (ignored if model_path is set).
        patch_size: spatial patch size for training and tiled prediction.
        batch_size: batch size for training.
        mode: '2D' or '3D'.

    Returns:
        (denoised_image, careamist) — the denoised array and trained model object.
    """
    from careamics import CAREamist
    from careamics.config import create_n2v_configuration

    is_3d = mode.upper() == "3D"

    if model_path is not None:
        print(f"Loading pretrained model: {model_path}")
        careamist = CAREamist(model_path)
    else:
        # Determine axes
        if image.ndim == 2:
            axes = "YX"
        elif image.ndim == 3:
            axes = "ZYX" if is_3d else "SYX"
        elif image.ndim == 4:
            axes = "SZYX" if is_3d else "STYX"
        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}D")

        print(f"Training N2V model — mode={mode}, axes={axes}, "
              f"epochs={n_epochs}, patch_size={patch_size}, batch_size={batch_size}")

        # Patch size
        if is_3d and image.ndim >= 3:
            z_dim = image.shape[-3] if image.ndim >= 3 else image.shape[0]
            ps = [min(patch_size, z_dim), patch_size, patch_size]
        else:
            ps = [patch_size, patch_size]

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
        careamist.train(train_source=image.astype(np.float32))

    # Tiling for prediction
    if is_3d and image.ndim >= 3:
        z_dim = image.shape[-3]
        tile_size = (min(patch_size, z_dim), patch_size, patch_size)
        tile_overlap = (
            min(patch_size // 4, z_dim // 2),
            patch_size // 4,
            patch_size // 4,
        )
    else:
        tile_size = (patch_size, patch_size)
        tile_overlap = (patch_size // 4, patch_size // 4)

    # Determine prediction axes
    if image.ndim == 2:
        pred_axes = "YX"
    elif image.ndim == 3:
        pred_axes = "ZYX" if is_3d else "SYX"
    elif image.ndim == 4:
        pred_axes = "SZYX" if is_3d else "STYX"
    else:
        pred_axes = None

    print(f"Predicting with tile_size={tile_size}, tile_overlap={tile_overlap}")

    pred_kwargs = dict(
        source=image.astype(np.float32),
        data_type="array",
        dataloader_params={"num_workers": 0},
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    if pred_axes is not None:
        pred_kwargs["axes"] = pred_axes

    prediction = careamist.predict(**pred_kwargs)

    if isinstance(prediction, list):
        prediction = prediction[0]

    return np.squeeze(np.asarray(prediction)), careamist


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
             "If not provided, trains a new model on the input data.",
    )
    parser.add_argument(
        "--save-model", default=None, metavar="PATH",
        help="Save the trained model to this path (BioImage Model Zoo .zip format)",
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
        "--patch-size", type=int, default=64,
        help="Patch size for training/prediction (default: 64)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for training (default: 1)",
    )

    args = parser.parse_args()

    # Load image
    import tifffile
    print(f"Loading: {args.input}")
    image = tifffile.imread(args.input)
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")

    # Denoise
    t0 = time.perf_counter()
    denoised, careamist = denoise_n2v(
        image,
        model_path=args.model,
        n_epochs=args.epochs,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        mode=args.mode,
    )
    elapsed = time.perf_counter() - t0
    print(f"Denoising complete in {elapsed:.1f}s")

    # Save output
    tifffile.imwrite(args.output, denoised.astype(image.dtype))
    print(f"Saved: {args.output}")

    # Save model if requested
    if args.save_model:
        try:
            careamist.export_to_bmz(
                path_to_archive=args.save_model,
                friendly_model_name="N2V Denoised Model",
                input_array=image[:1] if image.ndim >= 3 else image,
                authors=["CellphyLab"],
                general_description="Noise2Void denoising model trained with CAREamics",
                data_description=f"Trained on {args.input}",
            )
            print(f"Model saved: {args.save_model}")
        except Exception as e:
            print(f"Warning: could not save model: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
