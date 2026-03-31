from __future__ import annotations

import traceback

import numpy as np
from qtpy.QtCore import QThread, Signal

from napari_spotiflow_tracking._fitting import fit_and_mask_2d
from napari_spotiflow_tracking._preprocessing import remove_background_high_pass
from napari_spotiflow_tracking._segmentation import detect_spots, detect_spots_log


class DetectionWorker(QThread):
    """Background worker for spot detection + Gaussian fitting.

    Handles both single 2D images and T,Y,X stacks.
    Supports Spotiflow and LoG detection methods.
    """

    progress = Signal(str, int, int)  # (stage, current, total)
    finished = Signal(object, object)  # (all_points, all_masks)
    errored = Signal(str)

    def __init__(
        self,
        image: np.ndarray,
        model=None,
        prob_thresh: float | None = None,
        min_distance: int = 2,
        generate_mask: bool = True,
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
            points, _ = detect_spots(
                frame_image, self._model,
                prob_thresh=self._prob_thresh,
                min_distance=self._min_distance,
            )
        else:
            points = detect_spots_log(
                frame_image,
                min_sigma=self._log_min_sigma,
                max_sigma=self._log_max_sigma,
                num_sigma=self._log_num_sigma,
                threshold=self._log_threshold,
            )

        if self._generate_mask:
            result = fit_and_mask_2d(
                frame_image, points,
                patch_radius=self._patch_radius,
                fallback_radius=self._fallback_radius,
            )
            return points, result.mask
        return points, None

    def run(self):
        try:
            if self._image.ndim == 2:
                self.progress.emit("Detecting spots", 0, 1)
                points, mask = self._process_frame(self._image)
                self.progress.emit("Detecting spots", 1, 1)
                self.finished.emit(points, mask)

            elif self._image.ndim == 3:
                n_frames = self._image.shape[0]
                all_points = []
                all_masks = []

                for t in range(n_frames):
                    self.progress.emit("Detecting spots", t + 1, n_frames)
                    points, mask = self._process_frame(self._image[t])

                    if len(points) > 0:
                        frame_col = np.full((len(points), 1), t)
                        points_with_frame = np.hstack([frame_col, points])
                        all_points.append(points_with_frame)

                    if mask is not None:
                        all_masks.append(mask)

                if all_points:
                    combined_points = np.vstack(all_points)
                else:
                    combined_points = np.empty((0, 3))
                combined_masks = np.stack(all_masks, axis=0) if all_masks else None

                self.finished.emit(combined_points, combined_masks)
            else:
                self.errored.emit(
                    f"Expected 2D or 3D (T,Y,X) image, got ndim={self._image.ndim}"
                )
        except Exception:
            self.errored.emit(traceback.format_exc())
