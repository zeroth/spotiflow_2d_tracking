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


def _make_gaussian_image(cy, cx, sigma, size=64, amplitude=200.0, bg=20.0):
    """Create a synthetic image with a single 2D Gaussian spot."""
    yy, xx = np.mgrid[0:size, 0:size]
    image = bg + amplitude * np.exp(
        -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)
    )
    return image.astype(np.float32)


class TestFitAndMask2D:
    def test_single_spot_fit(self):
        from napari_spotiflow_tracking._fitting import fit_and_mask_2d

        image = _make_gaussian_image(32, 32, sigma=2.5)
        points = np.array([[32, 32]])

        result = fit_and_mask_2d(image, points, patch_radius=6)

        assert len(result.fits) == 1
        assert result.fits[0].success is True
        assert result.fits[0].sigma_y == pytest.approx(2.5, abs=0.5)
        assert result.fits[0].sigma_x == pytest.approx(2.5, abs=0.5)
        assert result.mask[32, 32] > 0  # mask painted at center

    def test_multiple_spots(self):
        from napari_spotiflow_tracking._fitting import fit_and_mask_2d

        image = np.zeros((64, 64), dtype=np.float32) + 20.0
        yy, xx = np.mgrid[0:64, 0:64]
        image += 200 * np.exp(-((yy - 15) ** 2 + (xx - 15) ** 2) / (2 * 2.0 ** 2))
        image += 200 * np.exp(-((yy - 45) ** 2 + (xx - 45) ** 2) / (2 * 2.0 ** 2))
        points = np.array([[15, 15], [45, 45]])

        result = fit_and_mask_2d(image, points, patch_radius=6)

        assert len(result.fits) == 2
        assert result.mask[15, 15] == 1
        assert result.mask[45, 45] == 2

    def test_mask_dtype_is_uint16(self):
        from napari_spotiflow_tracking._fitting import fit_and_mask_2d

        image = _make_gaussian_image(32, 32, sigma=2.0)
        points = np.array([[32, 32]])

        result = fit_and_mask_2d(image, points)

        assert result.mask.dtype == np.uint16

    def test_fallback_radius_on_failed_fit(self):
        from napari_spotiflow_tracking._fitting import fit_and_mask_2d

        # Flat image — fitting should fail
        image = np.ones((64, 64), dtype=np.float32) * 50.0
        points = np.array([[32, 32]])

        result = fit_and_mask_2d(image, points, fallback_radius=3)

        assert len(result.fits) == 1
        assert result.fits[0].success is False
        assert result.mask[32, 32] > 0

    def test_empty_points_returns_empty(self):
        from napari_spotiflow_tracking._fitting import fit_and_mask_2d

        image = _make_gaussian_image(32, 32, sigma=2.0)
        points = np.empty((0, 2))

        result = fit_and_mask_2d(image, points)

        assert len(result.fits) == 0
        assert np.sum(result.mask) == 0
