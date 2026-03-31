import numpy as np
import pytest


FWHM_CONSTANT = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ~2.355


class TestSpotFit2D:
    def test_fwhm_calculation(self):
        from napari_spotiflow_tracking._fitting import SpotFit2D

        fit = SpotFit2D(
            y0=0.0, x0=0.0,
            sigma_y=2.0, sigma_x=3.0,
            amplitude=100.0, background=10.0,
            success=True,
        )

        assert fit.fwhm_y == pytest.approx(FWHM_CONSTANT * 2.0)
        assert fit.fwhm_x == pytest.approx(FWHM_CONSTANT * 3.0)

    def test_paint_mask_single_spot(self):
        from napari_spotiflow_tracking._fitting import SpotFit2D

        mask = np.zeros((64, 64), dtype=np.uint16)
        fit = SpotFit2D(
            y0=0.0, x0=0.0,
            sigma_y=2.0, sigma_x=2.0,
            amplitude=100.0, background=10.0,
            success=True,
        )

        fit.paint_mask(mask, center_yx=(32, 32), label=1)

        assert mask[32, 32] == 1  # center is painted
        assert mask[0, 0] == 0    # far corner is not painted
        assert np.sum(mask > 0) > 1  # more than just center pixel

    def test_paint_mask_clips_to_bounds(self):
        from napari_spotiflow_tracking._fitting import SpotFit2D

        mask = np.zeros((20, 20), dtype=np.uint16)
        fit = SpotFit2D(
            y0=0.0, x0=0.0,
            sigma_y=5.0, sigma_x=5.0,
            amplitude=100.0, background=10.0,
            success=True,
        )

        # Spot near edge — should not raise
        fit.paint_mask(mask, center_yx=(2, 2), label=1)

        assert mask[2, 2] == 1
        assert mask.shape == (20, 20)  # no out-of-bounds

    def test_paint_mask_elliptical(self):
        from napari_spotiflow_tracking._fitting import SpotFit2D

        mask = np.zeros((64, 64), dtype=np.uint16)
        fit = SpotFit2D(
            y0=0.0, x0=0.0,
            sigma_y=1.0, sigma_x=4.0,  # wider in x
            amplitude=100.0, background=10.0,
            success=True,
        )

        fit.paint_mask(mask, center_yx=(32, 32), label=1)

        # Measure extent in y and x
        painted = np.argwhere(mask > 0)
        y_extent = painted[:, 0].max() - painted[:, 0].min()
        x_extent = painted[:, 1].max() - painted[:, 1].min()

        assert x_extent > y_extent  # ellipse is wider in x

    def test_paint_mask_failed_fit_does_nothing(self):
        from napari_spotiflow_tracking._fitting import SpotFit2D

        mask = np.zeros((64, 64), dtype=np.uint16)
        fit = SpotFit2D(
            y0=0.0, x0=0.0,
            sigma_y=2.0, sigma_x=2.0,
            amplitude=100.0, background=10.0,
            success=False,
        )

        fit.paint_mask(mask, center_yx=(32, 32), label=1)

        assert np.sum(mask > 0) == 0  # nothing painted
