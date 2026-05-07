from __future__ import annotations

import traceback

import numpy as np
from qtpy.QtCore import QThread, Signal

from napari_spotiflow_tracking._fitting import fit_and_mask_2d
from napari_spotiflow_tracking._preprocessing import remove_background_high_pass
from napari_spotiflow_tracking._segmentation import detect_spots, detect_spots_log, sample_point_intensities


class DetectionWorker(QThread):
    """Background worker for spot detection + Gaussian fitting.

    Handles both single 2D images and T,Y,X stacks.
    Supports Spotiflow and LoG detection methods.
    """

    progress = Signal(str, int, int)  # (stage, current, total)
    finished = Signal(object, object, object)  # (all_points, all_masks, all_intensities)
    errored = Signal(str)

    def __init__(
        self,
        image: np.ndarray,
        model=None,
        prob_thresh: float | None = None,
        min_distance: int = 2,
        generate_mask: bool = True,
        fit_backend: str | None = None,
        patch_radius: int = 4,
        fallback_radius: float = 2.0,
        method: str = "spotiflow",
        remove_bg: bool = False,
        bg_sigma: float = 10.0,
        log_min_sigma: float = 2,
        log_max_sigma: float = 10,
        log_num_sigma: int = 10,
        log_threshold: float = 0.09,
        parent=None,
    ):
        super().__init__(parent)
        self._image = image
        self._model = model
        self._prob_thresh = prob_thresh
        self._min_distance = min_distance
        self._generate_mask = generate_mask
        self._fit_backend = fit_backend
        self._patch_radius = patch_radius
        self._fallback_radius = fallback_radius
        self._method = method
        self._remove_bg = remove_bg
        self._bg_sigma = bg_sigma
        self._log_min_sigma = log_min_sigma
        self._log_max_sigma = log_max_sigma
        self._log_num_sigma = log_num_sigma
        self._log_threshold = log_threshold

    def _process_frame(self, frame_image: np.ndarray):
        """Detect spots and optionally fit Gaussians for a single 2D frame."""
        if self._remove_bg:
            frame_image = remove_background_high_pass(frame_image, sigma=self._bg_sigma)

        if self._method == "spotiflow":
            points, details = detect_spots(
                frame_image, self._model,
                prob_thresh=self._prob_thresh,
                min_distance=self._min_distance,
            )
            try:
                intens = np.asarray(details.intens, dtype=float).ravel()
            except AttributeError:
                intens = sample_point_intensities(frame_image, points)
        else:
            points = detect_spots_log(
                frame_image,
                min_sigma=self._log_min_sigma,
                max_sigma=self._log_max_sigma,
                num_sigma=self._log_num_sigma,
                threshold=self._log_threshold,
            )
            intens = sample_point_intensities(frame_image, points)

        if self._generate_mask:
            result = fit_and_mask_2d(
                frame_image, points,
                patch_radius=self._patch_radius,
                fallback_radius=self._fallback_radius,
                backend=self._fit_backend,
            )
            return points, result.mask, intens
        return points, None, intens

    def run(self):
        try:
            if self._image.ndim == 2:
                self.progress.emit("Detecting spots", 0, 1)
                points, mask, intens = self._process_frame(self._image)
                self.progress.emit("Detecting spots", 1, 1)
                self.finished.emit(points, mask, intens)

            elif self._image.ndim == 3:
                n_frames = self._image.shape[0]
                all_points = []
                all_masks = []
                all_intens = []

                for t in range(n_frames):
                    self.progress.emit("Detecting spots", t + 1, n_frames)
                    points, mask, intens = self._process_frame(self._image[t])

                    if len(points) > 0:
                        frame_col = np.full((len(points), 1), t)
                        points_with_frame = np.hstack([frame_col, points])
                        all_points.append(points_with_frame)
                        all_intens.append(intens)

                    if mask is not None:
                        all_masks.append(mask)

                if all_points:
                    combined_points = np.vstack(all_points)
                    combined_intens = np.concatenate(all_intens)
                else:
                    combined_points = np.empty((0, 3))
                    combined_intens = np.empty((0,))
                combined_masks = np.stack(all_masks, axis=0) if all_masks else None

                self.finished.emit(combined_points, combined_masks, combined_intens)
            else:
                self.errored.emit(
                    f"Expected 2D or 3D (T,Y,X) image, got ndim={self._image.ndim}"
                )
        except Exception:
            self.errored.emit(traceback.format_exc())


def _n2v_subprocess(image, model_path, n_epochs, patch_size, mode, train_frames, predict_image, result_file, error_queue):
    """Run N2V in a separate process (cleanly killable).

    Writes result to a .npy temp file to avoid Queue size limits.
    """
    try:
        from napari_spotiflow_tracking._preprocessing import denoise_n2v
        result = denoise_n2v(image, model_path=model_path,
                             n_epochs=n_epochs, patch_size=patch_size,
                             mode=mode, train_frames=train_frames,
                             predict_image=predict_image)
        np.save(result_file, result)
        error_queue.put(("ok", None))
    except ImportError:
        error_queue.put(("import_error", "CAREamics not installed. Run: pip install careamics"))
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print(tb, flush=True)
        error_queue.put(("error", tb))


class N2VWorker(QThread):
    """Background worker for Noise2Void denoising.

    Runs N2V in a child process so it can be cleanly cancelled without
    corrupting PyTorch/Lightning state.
    """

    progress = Signal(str)  # status message
    finished = Signal(object)  # denoised image
    errored = Signal(str)

    def __init__(
        self,
        image: np.ndarray,
        model_path: str | None = None,
        n_epochs: int = 100,
        patch_size: int = 64,
        mode: str = "2D",
        train_frames: int | None = None,
        predict_image: np.ndarray | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._image = image
        self._model_path = model_path
        self._n_epochs = n_epochs
        self._patch_size = patch_size
        self._mode = mode
        self._train_frames = train_frames
        self._predict_image = predict_image
        self._process = None

    def run(self):
        import multiprocessing as mp
        import tempfile

        if self._model_path is None:
            self.progress.emit(f"Training N2V model ({self._n_epochs} epochs)...")
        else:
            self.progress.emit("Denoising with pretrained model...")

        # Use temp file for result transfer (avoids Queue size limits)
        tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        result_file = tmp.name
        tmp.close()

        ctx = mp.get_context("spawn")
        error_queue = ctx.Queue()
        self._process = ctx.Process(
            target=_n2v_subprocess,
            args=(self._image, self._model_path, self._n_epochs,
                  self._patch_size, self._mode, self._train_frames,
                  self._predict_image, result_file, error_queue),
        )
        self._process.start()
        self._process.join()

        if self._process.exitcode != 0 and error_queue.empty():
            self._cleanup_tmp(result_file)
            self.errored.emit("N2V process was cancelled or crashed.")
            return

        if error_queue.empty():
            self._cleanup_tmp(result_file)
            self.errored.emit("N2V process returned no status.")
            return

        status, payload = error_queue.get()
        if status == "ok":
            try:
                result = np.load(result_file)
                self.finished.emit(result)
            except Exception as e:
                self.errored.emit(f"Failed to load result: {e}")
            finally:
                self._cleanup_tmp(result_file)
        else:
            self._cleanup_tmp(result_file)
            self.errored.emit(payload)

    @staticmethod
    def _cleanup_tmp(path):
        import os
        try:
            os.unlink(path)
        except OSError:
            pass

    def cancel(self):
        """Kill the child process cleanly."""
        if self._process is not None and self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=3)


class GpuFitMaskWorker(QThread):
    """Run GPU Gaussian fit + mask generation as a background batch.

    Takes a 3D image stack and a (N, 3) points array (with columns
    frame, y, x) and produces a (T, Y, X) stack of 2D labels per frame
    plus a per-spot props DataFrame whose columns include ``area``,
    ``mean_intensity``, ``max_intensity``, ``spot_radius`` (matching the
    FilterPanel whitelist where possible).
    """

    progress = Signal(str)
    finished = Signal(object, object)  # (mask, props_df)
    errored = Signal(str)

    def __init__(
        self,
        image: np.ndarray,
        points: np.ndarray,
        win: int = 2,
        max_radius: float | None = None,
        max_iter: int = 50,
        chunk_size: int = 20000,
        parent=None,
    ):
        super().__init__(parent)
        self._image = image
        self._points = points
        self._win = win
        # Default: cap spot radius at the fit half-window so the mask and
        # the fit footprint stay consistent.
        self._max_radius = float(win) if max_radius is None else float(max_radius)
        self._max_iter = max_iter
        self._chunk_size = chunk_size

    def run(self):
        try:
            import pandas as pd

            from napari_spotiflow_tracking._fitting import (
                fit_all_frames_gpu,
                make_gpu_spot_mask,
            )

            self.progress.emit("Preparing points DataFrame...")
            pts = np.asarray(self._points)
            if pts.ndim != 2 or pts.shape[0] == 0:
                self.errored.emit("No detections to fit.")
                return
            if pts.shape[1] == 3:
                # (frame, y, x); convert frame to 1-indexed time as expected
                # by fit_all_frames_gpu's contract.
                df = pd.DataFrame({
                    "time": pts[:, 0].astype(int) + 1,
                    "y": pts[:, 1].astype(float),
                    "x": pts[:, 2].astype(float),
                })
            else:
                self.errored.emit(
                    f"Expected (N, 3) points array; got shape {pts.shape}"
                )
                return

            stack = self._image
            if stack.ndim == 2:
                stack = stack[np.newaxis, ...]

            self.progress.emit("GPU Gaussian fitting...")
            fitted = fit_all_frames_gpu(
                stack, df, win=self._win,
                max_iter=self._max_iter, chunk_size=self._chunk_size,
            )
            # make_gpu_spot_mask reads x_fit/y_fit-like columns as x/y after
            # the fit completes; in our flow we keep the original (rounded)
            # x/y as the centers since the points come from the detector.
            # The fitted x_fit/y_fit are sub-pixel refinements; using the
            # original integer-rounded x/y for the mask placement is
            # consistent with the existing detection flow.
            self.progress.emit("Building mask + props...")
            mask, props_df = make_gpu_spot_mask(
                stack_shape=stack.shape,
                df=fitted,
                intensity_stack=stack,
                max_radius=self._max_radius,
            )
            # Add a `frame` column (0-indexed) so it matches the regionprops
            # convention used by _merge_regionprops_into_points (which keys
            # on frame, y, x).
            props_df = props_df.copy()
            props_df["frame"] = props_df["time"].astype(int) - 1
            self.finished.emit(mask, props_df)
        except Exception:
            import traceback
            self.errored.emit(traceback.format_exc())


class MaskGenerationWorker(QThread):
    """Background worker for Gaussian fitting + mask painting + regionprops.

    Processes a Points layer against an image (2D or T,Y,X stack).
    Computes regionprops per frame alongside mask generation.
    """

    progress = Signal(str, int, int)  # (stage, current, total)
    finished = Signal(object, object)  # (mask array, regionprops DataFrame)
    errored = Signal(str)

    def __init__(
        self,
        image: np.ndarray,
        points: np.ndarray,
        patch_radius: int = 4,
        fallback_radius: float = 2.0,
        fit_backend: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._image = image
        self._points = points
        self._patch_radius = patch_radius
        self._fallback_radius = fallback_radius
        self._fit_backend = fit_backend

    def _compute_regionprops(self, mask, image, frame=None):
        """Compute regionprops for a single frame."""
        import pandas as pd
        from skimage.measure import regionprops_table

        properties = [
            "label", "area", "centroid", "mean_intensity", "max_intensity",
            "min_intensity",
        ]
        table = regionprops_table(
            mask, intensity_image=image, properties=properties,
        )
        df = pd.DataFrame(table)
        df.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)
        if frame is not None:
            df.insert(0, "frame", frame)
        return df

    def run(self):
        import pandas as pd

        try:
            if self._image.ndim == 2:
                pts = self._points[:, -2:] if self._points.shape[1] > 2 else self._points
                self.progress.emit("Fitting spots", 0, 1)
                result = fit_and_mask_2d(
                    self._image, pts,
                    patch_radius=self._patch_radius,
                    fallback_radius=self._fallback_radius,
                    backend=self._fit_backend,
                )
                props_df = self._compute_regionprops(result.mask, self._image)
                self.progress.emit("Fitting spots", 1, 1)
                self.finished.emit(result.mask, props_df)

            elif self._image.ndim == 3:
                if self._points.shape[1] != 3:
                    self.errored.emit("Points must have 3 columns (frame, y, x) for stacks.")
                    return

                n_frames = self._image.shape[0]
                all_masks = []
                all_props = []

                for t in range(n_frames):
                    self.progress.emit("Fitting spots", t + 1, n_frames)
                    frame_pts = self._points[self._points[:, 0] == t][:, 1:]
                    result = fit_and_mask_2d(
                        self._image[t], frame_pts,
                        patch_radius=self._patch_radius,
                        fallback_radius=self._fallback_radius,
                        backend=self._fit_backend,
                    )
                    all_masks.append(result.mask)
                    all_props.append(
                        self._compute_regionprops(result.mask, self._image[t], frame=t)
                    )

                combined_mask = np.stack(all_masks, axis=0)
                combined_props = pd.concat(all_props, ignore_index=True)
                self.finished.emit(combined_mask, combined_props)
            else:
                self.errored.emit(f"Unsupported image dimensions: {self._image.ndim}D")
        except Exception:
            self.errored.emit(traceback.format_exc())
