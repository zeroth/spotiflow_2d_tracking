"""Tests for high-pass background removal.

The high-pass subtracts a Gaussian-blurred copy, which centres the result on
zero. These cover the re-baselining that follows: the original mean is added
back and the input dtype restored, so an integer image stays usable rather
than wrapping around.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from napari_spotiflow_tracking._preprocessing import (
    remove_background_high_pass,
    remove_background_high_pass_stack,
)


@pytest.fixture(params=["torch", "scipy"])
def backend(request, monkeypatch):
    """Run each test against both the torch path and the scipy fallback."""
    if request.param == "scipy":
        # Setting the entry to None makes `import torch` raise ImportError,
        # which is what selects the fallback branch.
        monkeypatch.setitem(sys.modules, "torch", None)
    return request.param


def _blob_image(dtype=np.uint16):
    """A dim field with one bright blob — a strong background gradient."""
    img = np.full((64, 64), 100, dtype=np.float64)
    img[28:36, 28:36] = 3000.0
    return img.astype(dtype)


class TestHighPassRebaseline:
    def test_preserves_input_dtype(self, backend):
        img = _blob_image(np.uint16)
        out = remove_background_high_pass(img, sigma=5.0)
        assert out.dtype == np.uint16

    def test_restores_the_original_mean(self, backend):
        # The high-pass removes the DC component; adding the mean back should
        # land the result close to where the input sat.
        img = _blob_image(np.uint16)
        out = remove_background_high_pass(img, sigma=5.0)
        assert out.mean() == pytest.approx(img.mean(), rel=0.15)

    def test_integer_result_does_not_wrap_around(self, backend):
        # Without clipping, a negative residual cast to uint16 becomes ~65535,
        # turning the darkest pixels into the brightest in the frame.
        img = _blob_image(np.uint16)
        out = remove_background_high_pass(img, sigma=5.0)
        ceiling = int(img.max()) + int(round(img.mean())) + 1
        assert out.max() <= ceiling
        assert out.min() >= 0

    def test_constant_image_is_unchanged(self, backend):
        # Blurring a constant returns it unchanged, so the residual is zero and
        # the result is the original level.
        img = np.full((32, 32), 500, dtype=np.uint16)
        out = remove_background_high_pass(img, sigma=4.0)
        assert np.allclose(out, 500, atol=1)

    def test_float_input_is_not_clipped(self, backend):
        # Floats keep their sign: only integer dtypes are clipped.
        img = _blob_image(np.float32)
        out = remove_background_high_pass(img, sigma=5.0)
        assert out.dtype == np.float32
        assert out.min() < 0


class TestHighPassStackRebaseline:
    def test_preserves_input_dtype(self, backend):
        stack = np.stack([_blob_image(np.uint16)] * 3)
        out = remove_background_high_pass_stack(stack, sigma=5.0)
        assert out.dtype == np.uint16

    def test_matches_frame_by_frame_2d(self, backend):
        # The widget sends 2D images to one function and stacks to the other;
        # they must not disagree.
        frames = [_blob_image(np.uint16), (_blob_image(np.uint16) // 2)]
        stack = np.stack(frames)
        out = remove_background_high_pass_stack(stack, sigma=5.0)
        for i, frame in enumerate(frames):
            expected = remove_background_high_pass(frame, sigma=5.0)
            assert np.array_equal(out[i], expected)

    def test_rebaselines_each_frame_on_its_own_mean(self, backend):
        # Frame 2 is half as bright; each frame keeps its own level rather than
        # being pushed to a common one.
        bright = _blob_image(np.uint16)
        dim = (bright // 2).astype(np.uint16)
        out = remove_background_high_pass_stack(np.stack([bright, dim]),
                                                sigma=5.0)
        assert out[0].mean() == pytest.approx(bright.mean(), rel=0.15)
        assert out[1].mean() == pytest.approx(dim.mean(), rel=0.15)

    def test_integer_result_does_not_wrap_around(self, backend):
        stack = np.stack([_blob_image(np.uint16)] * 2)
        out = remove_background_high_pass_stack(stack, sigma=5.0)
        ceiling = int(stack.max()) + int(round(stack.mean())) + 1
        assert out.max() <= ceiling
        assert out.min() >= 0
