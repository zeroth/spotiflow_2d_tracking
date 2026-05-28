import numpy as np

# remove background using morphological reconstruction
import skimage
# from skimage.morphology import reconstruction, disk, erosion
from skimage import morphology
def remove_background(image, disk_size=10):
    seed = morphology.erosion(image, morphology.disk(disk_size))
    background = morphology.reconstruction(seed, image, method='dilation')
    cleaned = image - background
    return cleaned

# find blobs in cleaned image using blob_log
from skimage.feature import blob_log
from skimage.exposure import rescale_intensity, equalize_hist
def find_blobs(image, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.1):
    im_range = (np.min(image), np.max(image))
    image = rescale_intensity(image, in_range=im_range, out_range=(0, 1))
    # image = equalize_hist(image)

    blobs = blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=num_sigma, threshold=threshold, overlap=0.0, exclude_border=True)
    return blobs

# histogram matching to reference image
from skimage.exposure import match_histograms
def histogram_match(image, reference):
    matched = match_histograms(image, reference)
    return matched

# create a mask from blobs
from skimage.draw import disk   
def create_blob_mask(image_shape, blobs):
    mask = np.zeros(image_shape, dtype=np.uint8)
    for blob in blobs:
        y, x, sigma = blob
        r = int(sigma * np.sqrt(2))  # adjust radius for blob_log
        rr, cc = disk((y, x), r, shape=image_shape)
        mask[rr, cc] = 255
    return mask

# find regionprops of mask
_defatul_properties = [
    "label",
    "area",
    "centroid",
    "mean_intensity",
    "max_intensity",
    "min_intensity",
    "bbox",
    "equivalent_diameter",
    "perimeter",
    "solidity",
]
from skimage.measure import label, regionprops_table
def get_blob_regionprops(mask, image):
    labeled_mask = label(mask)
    regions = regionprops_table(labeled_mask, intensity_image=image, properties=_defatul_properties)
    return regions


# find spots using spotiflow pretrained model
def find_spots_spotiflow(image, model, prob_thresh=None, min_distance=1):
    kwargs = {}
    kwargs['verbose'] = False
    if prob_thresh is not None:
        kwargs['prob_thresh'] = prob_thresh
    if min_distance != 1:
        kwargs['min_distance'] = min_distance
    points, details = model.predict(image, **kwargs)
    return points, details

# create a mask from spotiflow point detections
def create_spot_mask(image_shape, points, radius=2):
    mask = np.zeros(image_shape, dtype=np.uint8)
    for blob in points:
        y, x = blob[:2]
        rr, cc = disk((int(y), int(x)), radius, shape=image_shape)
        mask[rr, cc] = 255
    return mask

# estimate spot FWHM via vectorised second-moment (no iterative fitting)
FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2))  # ≈ 2.355

def estimate_spot_fwhm(image, points, patch_radius=8):
    n = len(points)
    pr = patch_radius
    h, w = image.shape
    patch_size = 2 * pr + 1

    iy = np.round(points[:, 0]).astype(int)
    ix = np.round(points[:, 1]).astype(int)
    valid = (iy - pr >= 0) & (iy + pr < h) & (ix - pr >= 0) & (ix + pr < w)
    valid_idx = np.nonzero(valid)[0]

    fwhm = np.full(n, np.nan)
    if len(valid_idx) == 0:
        return fwhm

    # batch-extract all patches into a contiguous 3D array: (N, P, P)
    # use fancy indexing to build row/col grids for all patches at once
    offsets = np.arange(-pr, pr + 1)
    row_grid = iy[valid_idx, None] + offsets[None, :]  # (N, P)
    col_grid = ix[valid_idx, None] + offsets[None, :]   # (N, P)
    # extract Y profiles (column through centre) and X profiles (row through centre)
    centre_col = ix[valid_idx]  # x coord of each spot
    centre_row = iy[valid_idx]  # y coord of each spot
    profiles_y = image[row_grid, centre_col[:, None]].astype(float)  # (N, P)
    profiles_x = image[centre_row[:, None], col_grid].astype(float)  # (N, P)

    # background: 10th percentile per profile
    bg_y = np.percentile(profiles_y, 10, axis=1, keepdims=True)
    bg_x = np.percentile(profiles_x, 10, axis=1, keepdims=True)
    w_y = np.maximum(profiles_y - bg_y, 0)
    w_x = np.maximum(profiles_x - bg_x, 0)

    # weighted second moment: sigma^2 = sum(w * (x - mu)^2) / sum(w)
    coords = np.arange(patch_size, dtype=float)
    sum_wy = w_y.sum(axis=1)
    sum_wx = w_x.sum(axis=1)
    # avoid division by zero
    good = (sum_wy > 0) & (sum_wx > 0)
    mu_y = (w_y[good] * coords).sum(axis=1) / sum_wy[good]
    mu_x = (w_x[good] * coords).sum(axis=1) / sum_wx[good]
    var_y = (w_y[good] * (coords[None, :] - mu_y[:, None]) ** 2).sum(axis=1) / sum_wy[good]
    var_x = (w_x[good] * (coords[None, :] - mu_x[:, None]) ** 2).sum(axis=1) / sum_wx[good]
    sigma = (np.sqrt(var_y) + np.sqrt(var_x)) / 2

    # clamp unreasonable sigmas (< 0.3 px or > patch_radius)
    sigma = np.where((sigma >= 0.3) & (sigma <= pr), sigma, np.nan)

    fwhm[valid_idx[good]] = sigma * FWHM_FACTOR

    # fill NaN entries with the average of successfully fitted values
    valid_fwhm = fwhm[~np.isnan(fwhm)]
    if len(valid_fwhm) > 0:
        fwhm[np.isnan(fwhm)] = valid_fwhm.min()
    return fwhm

# create a mask from spotiflow points using per-spot FWHM
def create_spot_mask_with_fwhm(image_shape, points, fwhm, fallback_radius=2):
    mask = np.zeros(image_shape, dtype=np.uint8)
    for i, (y, x) in enumerate(points):
        if not np.isnan(fwhm[i]):
            sigma = fwhm[i] / FWHM_FACTOR
            r = int(round(sigma * np.sqrt(2)))
        else:
            r = fallback_radius
        r = max(r, 1)
        rr, cc = disk((int(round(y)), int(round(x))), r, shape=image_shape)
        mask[rr, cc] = 255
    return mask

# spotiflow-based segmentation function
def segment_image_spotiflow(image, model, prob_thresh=None, min_distance=1, spot_radius=2, patch_radius=8):
    points, details = find_spots_spotiflow(image, model, prob_thresh=prob_thresh, min_distance=min_distance)
    # fwhm = estimate_spot_fwhm(image, points, patch_radius=patch_radius)
    # mask = create_spot_mask_with_fwhm(image.shape, points, fwhm, fallback_radius=spot_radius)
    mask = create_spot_mask(image.shape, points, radius=spot_radius)
    results = {
        'y': points[:, 0].ravel(),
        'x': points[:, 1].ravel(),
        'intensity': np.asarray(details.intens).ravel(),
    }
    return mask, results, points, details


# main segmentation function
def segment_image(image, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.1, disk_size=10, ref_image=None):
    image_hist = None
    if ref_image is not None:
        print(f"Applying Matching histogram")
        image_hist = histogram_match(image, ref_image)
    if image_hist is not None:
        cleaned = remove_background(image_hist, disk_size=disk_size)
    else:
        cleaned = remove_background(image, disk_size=disk_size)
    # cleaned = image
    blobs = find_blobs(cleaned, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    mask = create_blob_mask(image.shape, blobs)
    regions = get_blob_regionprops(mask, image)
    return mask, regions, cleaned


if __name__ == "__main__":
    import tifffile
    import argparse
    import os
    from tqdm import tqdm
    import pandas as pd
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Segment blobs in a TIFF image.")
    parser.add_argument("input_tiff", type=str, help="Path to input TIFF image.")
    parser.add_argument("output_mask", type=str, help="Path to output mask TIFF image.")
    parser.add_argument("--min_sigma", type=float, default=2, help="Minimum sigma for blob detection.")
    parser.add_argument("--max_sigma", type=float, default=10, help="Maximum sigma for blob detection.")
    parser.add_argument("--num_sigma", type=int, default=10, help="Number of sigma values for blob detection.")
    parser.add_argument("--threshold", type=float, default=0.09, help="Threshold for blob detection.")
    parser.add_argument("--disk_size", type=int, default=10, help="Disk size for background removal.")
    # only remove background
    parser.add_argument("--remove_background_only", action='store_true', help="Only remove background without blob detection.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker threads (default: os.cpu_count()).")
    parser.add_argument("--save_cleaned", action='store_true', help="Save background-removed (cleaned) images (off by default).")
    parser.add_argument("--save_blobs", action='store_true', help="Save blob region properties to CSV (off by default).")
    parser.add_argument("--use_spotiflow", action='store_true', help="Use Spotiflow deep learning model instead of blob_log.")
    parser.add_argument("--spotiflow_model", type=str, default="general", help="Spotiflow pretrained model name (default: general).")
    parser.add_argument("--prob_thresh", type=float, default=None, help="Spotiflow probability threshold (default: model default).")
    parser.add_argument("--spot_radius", type=int, default=1, help="Fallback radius for spot mask when sigma fit fails (default: 1).")
    parser.add_argument("--patch_radius", type=int, default=4, help="Patch radius for Gaussian sigma estimation (default: 8).")
    args = parser.parse_args()

    # Load spotiflow model once if requested
    spotiflow_model = None
    if args.use_spotiflow:
        from spotiflow.model import Spotiflow
        print(f"Loading Spotiflow model '{args.spotiflow_model}'...")
        spotiflow_model = Spotiflow.from_pretrained(args.spotiflow_model, verbose=True)
        print("Spotiflow model loaded.")

    # Load image
    image = tifffile.imread(args.input_tiff)
    print(f"Loaded image from {args.input_tiff} with shape {image.shape}")
    if args.save_blobs:
        pd_blobs = pd.DataFrame()
    if args.save_cleaned:
        cleaned_path = str(Path(args.output_mask).with_suffix('')) + "_cleaned.tif"
    if image.ndim == 3:
        # Create memory-mapped output TIFF files with the known shape.
        # This lets us write each slice as it's processed instead of
        # holding the entire output stack in RAM.
        use_bigtiff = image.nbytes > 2**31
        masks = tifffile.memmap(
            args.output_mask,
            shape=image.shape,
            dtype=image.dtype,
            bigtiff=use_bigtiff,
        )
        if args.save_cleaned:
            cleaned_images = tifffile.memmap(
                cleaned_path,
                shape=image.shape,
                dtype=image.dtype,
                bigtiff=use_bigtiff,
            )

        # Process each slice in a 3D stack
        import concurrent.futures
        num_workers = args.workers or os.cpu_count() or 1

        # get first slice as reference for histogram matching if needed
        ref_image = tifffile.imread(args.input_tiff, key=0)

        def process_slice(i):
            blobs = None
            cleaned = None
            if args.remove_background_only:
                mask = remove_background(image[i], disk_size=args.disk_size)
            elif args.use_spotiflow:
                mask, blobs, points, details = segment_image_spotiflow(
                    image[i], spotiflow_model,
                    prob_thresh=args.prob_thresh,
                    spot_radius=args.spot_radius,
                    patch_radius=args.patch_radius)
            else:
                mask, blobs, cleaned = segment_image(image[i], min_sigma=args.min_sigma,
                                    max_sigma=args.max_sigma,
                                    num_sigma=args.num_sigma,
                                    threshold=args.threshold,
                                    disk_size=args.disk_size,
                                    ref_image=ref_image)
            return i, mask, blobs, cleaned

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_slice, i) for i in range(image.shape[0])]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Segmenting slices"):
                i, mask, blobs, cleaned = f.result()
                # Write directly to the memory-mapped file (random access by page index)
                masks[i] = mask
                if args.save_cleaned:
                    cleaned_images[i] = cleaned
                if args.save_blobs and blobs is not None:
                    slice_blobs = pd.DataFrame(blobs)
                    slice_blobs['time'] = i+1
                    pd_blobs = pd.concat([pd_blobs, slice_blobs], ignore_index=True)

        # Flush memory-mapped files to disk and release
        masks.flush()
        del masks
        if args.save_cleaned:
            cleaned_images.flush()
            del cleaned_images
    else:
        # Process single 2D image
        if args.remove_background_only:
            masks = remove_background(image, disk_size=args.disk_size)
        elif args.use_spotiflow:
            masks, blobs, _, _ = segment_image_spotiflow(
                image, spotiflow_model,
                prob_thresh=args.prob_thresh,
                spot_radius=args.spot_radius,
                patch_radius=args.patch_radius)
        else:
            masks, blobs, cleaned_images = segment_image(image, min_sigma=args.min_sigma,
                                    max_sigma=args.max_sigma,
                                    num_sigma=args.num_sigma,
                                    threshold=args.threshold,
                                    disk_size=args.disk_size)

        tifffile.imwrite(args.output_mask, masks)
        if args.save_cleaned and not args.remove_background_only:
            tifffile.imwrite(cleaned_path, cleaned_images)

    # Save blob data
    if args.save_blobs and pd_blobs.shape[0] > 0:
        blob_csv_path = str(Path(args.output_mask).with_suffix('')) + "_blobs.csv"
        pd_blobs.sort_values(by=['time'], inplace=True)
        pd_blobs.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)
        pd_blobs.to_csv(blob_csv_path, index=False)
        print(f"Saved blobs to {blob_csv_path}")
    print(f"Saved output to {args.output_mask}")
