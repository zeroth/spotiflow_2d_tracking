import torch
import numpy as np
import pandas as pd
import tifffile
from torchimize.functions import lsq_lma_parallel
from skimage.measure import regionprops_table

def gaussian_2d_batched(p, xy):
    amp = p[:, 0:1]; x0 = p[:, 1:2]; y0 = p[:, 2:3]
    sx  = p[:, 3:4]; sy = p[:, 4:5]; bg = p[:, 5:6]
    x = xy[0:1, :]; y = xy[1:2, :]
    dx = (x - x0) / sx
    dy = (y - y0) / sy
    out = amp * torch.exp(-0.5 * (dx*dx + dy*dy)) + bg
    return out.unsqueeze(1)

def residuals(p, data, xy):
    return gaussian_2d_batched(p, xy) - data


def extract_crops_vec(stack, df, win):
    """
    stack: (T, H, W) numpy array — full TIFF stack
    df:    pandas DataFrame with columns 'time', 'y', 'x'
    Returns: crops (N, size, size), valid_mask (len(df),), ordered by df row
    """
    T, H, W = stack.shape
    pts_y = np.round(df['y'].to_numpy()).astype(np.int64)
    pts_x = np.round(df['x'].to_numpy()).astype(np.int64)
    pts_t = df['time'].to_numpy().astype(np.int64)

    valid = ((pts_y >= win) & (pts_y < H - win) &
             (pts_x >= win) & (pts_x < W - win) &
             (pts_t >= 0)   & (pts_t < T))

    ty = pts_y[valid]; tx = pts_x[valid]; tt = pts_t[valid]

    # Build (N, size, size) crops via fancy indexing — one op, no Python loop
    dy, dx = np.meshgrid(np.arange(-win, win+1),
                         np.arange(-win, win+1), indexing='ij')
    ys = ty[:, None, None] + dy[None]                # (N, size, size)
    xs = tx[:, None, None] + dx[None]
    ts = tt[:, None, None]                           # broadcasts
    crops = stack[ts, ys, xs].astype(np.float32)     # (N, size, size)
    return crops, valid


def fit_all_frames(stack, df, win=5, device='cuda', max_iter=20, chunk_size=20000):
    size = 2 * win + 1
    M = size * size
    n_total = len(df)

    crops, valid = extract_crops_vec(stack, df, win)
    N = len(crops)

    amp_out = np.full(n_total, np.nan, dtype=np.float64)
    x_out   = np.full(n_total, np.nan, dtype=np.float64)
    y_out   = np.full(n_total, np.nan, dtype=np.float64)
    sx_out  = np.full(n_total, np.nan, dtype=np.float64)
    sy_out  = np.full(n_total, np.nan, dtype=np.float64)
    bg_out  = np.full(n_total, np.nan, dtype=np.float64)

    if N == 0:
        return df.assign(amp=amp_out, x_fit=x_out, y_fit=y_out,
                         sigma_x=sx_out, sigma_y=sy_out, bg=bg_out)

    yy, xx = torch.meshgrid(
        torch.arange(-win, win+1, dtype=torch.float32),
        torch.arange(-win, win+1, dtype=torch.float32),
        indexing='ij',
    )
    xy = torch.stack([xx.flatten(), yy.flatten()]).to(device)
    wvec = torch.ones(1, device=device)

    final_all = np.empty((N, 6), dtype=np.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = crops[start:end]
        B = len(chunk)

        data = torch.from_numpy(chunk.reshape(B, 1, M)).to(device)

        d2 = data[:, 0, :]
        amp0 = d2.max(1).values - d2.min(1).values
        bg0  = d2.min(1).values
        init = torch.stack([
            amp0,
            torch.zeros(B, device=device),
            torch.zeros(B, device=device),
            torch.full((B,), 1.5, device=device),
            torch.full((B,), 1.5, device=device),
            bg0,
        ], dim=1)

        coeffs_list = lsq_lma_parallel(
            p=init, function=residuals, args=(data, xy), wvec=wvec,
            ftol=1e-6, ptol=1e-6, gtol=1e-6, max_iter=max_iter,
        )
        final_all[start:end] = coeffs_list[-1].cpu().numpy()

    pts_y = np.round(df['y'].to_numpy()).astype(np.int64)
    pts_x = np.round(df['x'].to_numpy()).astype(np.int64)

    final = final_all.astype(np.float64)
    amp_out[valid] = final[:, 0]
    x_out[valid]   = pts_x[valid] + final[:, 1]
    y_out[valid]   = pts_y[valid] + final[:, 2]
    sx_out[valid]  = np.abs(final[:, 3])
    sy_out[valid]  = np.abs(final[:, 4])
    bg_out[valid]  = final[:, 5]

    return df.assign(amp=amp_out, x_fit=x_out, y_fit=y_out,
                     sigma_x=sx_out, sigma_y=sy_out, bg=bg_out)


# # def make_spot_mask(stack, df, label_value=255, dtype=np.uint8,
#                    min_radius=1.0, max_radius=None, label_col=None):
#     """
#     Build a binary/labeled mask stack with circles at each fitted spot.

#     stack_shape: (T, H, W) — shape of the source stack
#     df:          DataFrame with columns 'time', 'x_fit', 'y_fit', 'sigma_x', 'sigma_y'
#     label_value: int — value to write inside circles. Use 255 for binary uint8,
#                  or pass df['label'] later for a labeled mask (see note below).
#     dtype:       output dtype (uint8 default; use uint16/int32 if labeling)
#     min_radius:  clamp tiny radii so single-pixel spots still show
#     max_radius:  optional clamp on the upper end (drops blown-up fits)

#     Returns: mask of shape (T, H, W) and the per-spot radius array.
#     """
#     stack_shape = stack.shape
#     T, H, W = stack_shape
#     n_total = len(df)
#     # Drop rows where the fit failed (NaN sigmas)
#     valid = df[['x', 'y', 'sigma_x', 'sigma_y', 'time']].notna().all(axis=1)
#     sub = df.loc[valid]

#     # Output arrays sized to full df length so they align with the original rows
#     area_out = np.full(n_total, np.nan, dtype=np.float64)
#     mean_out = np.full(n_total, np.nan, dtype=np.float64)
#     max_out  = np.full(n_total, np.nan, dtype=np.float64)
#     radii_out = np.full(n_total, np.nan, dtype=np.float64)

#     if len(sub) == 0:
#         return np.zeros(stack_shape, dtype=dtype), df.assign(
#             spot_area=area_out, spot_mean=mean_out, spot_max=max_out,
#             spot_radius=radii_out,
#         )


#     cx = sub['x'].to_numpy()
#     cy = sub['y'].to_numpy()
#     ct = sub['time'].to_numpy().astype(np.int64) -1
#     sx = sub['sigma_x'].to_numpy()
#     sy = sub['sigma_y'].to_numpy()
#     valid_idx = np.where(valid)[0]   # original df row index for each spot in `sub`

#     # Per-spot label
#     if label_col is not None:
#         labels = sub[label_col].to_numpy().astype(dtype)
#     else:
#         labels = np.arange(1, len(sub) + 1, dtype=dtype)

#     # radius = sqrt(2) * mean(sigma_x, sigma_y), clamped
#     radii = np.sqrt(2.0) * 0.5 * (sx + sy)
#     radii = np.clip(radii, min_radius, max_radius if max_radius else np.inf)
#     radii_out[valid_idx] = radii

#     mask = np.zeros((T, H, W), dtype=dtype)

#     # Group spots by integer radius bucket — all spots with the same
#     # rounded radius share one stamp template, vectorized in one pass.
#     r_int = np.ceil(radii).astype(np.int32)
#     unique_r = np.unique(r_int)

#     for r in unique_r:
#         sel = np.where(r_int == r)[0]
#         # if not sel.any():
#         #     continue

#         # Build the disk stamp once for this radius
#         size = 2 * r + 1
#         yy, xx = np.ogrid[-r:r+1, -r:r+1]
#         # Use the actual (sub-pixel) radius for the boundary test:
#         # spots in this bucket all have radius <= r, but we use the per-spot
#         # radius for the inequality so each spot gets its true footprint.
#         # Stamp template covers the worst case; inner check is per-spot.

#         rr = radii[sel]
#         ccx = cx[sel]
#         ccy = cy[sel]
#         cct = ct[sel]
#         lbls = labels[sel]

#         # Integer top-left corner of each stamp
#         x0 = np.round(ccx).astype(np.int64) - r
#         y0 = np.round(ccy).astype(np.int64) - r

#         # Sub-pixel offsets so the disk is centered on the true (cx, cy)
#         dx = ccx - np.round(ccx)
#         dy = ccy - np.round(ccy)

#         # Loop only over spots in this radius bucket — short Python loop,
#         # numpy does the heavy lifting per iteration.
#         for i in range(len(rr)):
#             t = cct[i]
#             xi, yi = x0[i], y0[i]

#             # Build per-spot circle mask (cheap: size × size booleans)
#             disk = (xx - dx[i])**2 + (yy - dy[i])**2 <= rr[i]**2

#             # Clip to image bounds
#             x_start = max(0, xi); x_end = min(W, xi + size)
#             y_start = max(0, yi); y_end = min(H, yi + size)
#             dx_start = x_start - xi; dx_end = dx_start + (x_end - x_start)
#             dy_start = y_start - yi; dy_end = dy_start + (y_end - y_start)

#             if x_end <= x_start or y_end <= y_start:
#                 continue

#             # # OR into the mask (or use np.maximum for labeled masks)
#             # mask[t, y_start:y_end, x_start:x_end] |= (
#             #     disk[dy_start:dy_end, dx_start:dx_end].astype(dtype) * label_value
#             # )
#             local_disk = disk[dy_start:dy_end, dx_start:dx_end]
#             target = mask[t, y_start:y_end, x_start:x_end]
#             # Write label only where disk is True AND target is still 0.
#             # This makes earlier spots "win" at overlaps. Swap to np.maximum
#             # if you'd rather have the higher label win.
#             np.copyto(target, lbls[i], where=local_disk & (target == 0))
    
#     # props = regionprops_table(mask, intensity_image=stack,
#     #                       properties=("label", "area", "centroid", "mean_intensity", "max_intensity", "min_intensity"))
#     return mask, radii


def make_spot_mask(stack_shape, df, intensity_stack=None, dtype=np.int32,
                   min_radius=1.0, max_radius=None, label_col=None):
    T, H, W = stack_shape
    n_total = len(df)

    valid = df[['x', 'y', 'sigma_x', 'sigma_y', 'time']].notna().all(axis=1)
    sub = df.loc[valid]

    # Output arrays sized to full df length so they align with the original rows
    area_out = np.full(n_total, np.nan, dtype=np.float64)
    mean_out = np.full(n_total, np.nan, dtype=np.float64)
    max_out  = np.full(n_total, np.nan, dtype=np.float64)
    radii_out = np.full(n_total, np.nan, dtype=np.float64)

    if len(sub) == 0:
        return np.zeros(stack_shape, dtype=dtype), df.assign(
            spot_area=area_out, spot_mean=mean_out, spot_max=max_out,
            spot_radius=radii_out,
        )

    cx = sub['x'].to_numpy()
    cy = sub['y'].to_numpy()
    ct = sub['time'].to_numpy().astype(np.int64) -1
    sx = sub['sigma_x'].to_numpy()
    sy = sub['sigma_y'].to_numpy()
    valid_idx = np.where(valid)[0]   # original df row index for each spot in `sub`

    if label_col is not None:
        labels = sub[label_col].to_numpy().astype(dtype)
    else:
        labels = np.arange(1, len(sub) + 1, dtype=dtype)

    radii = np.sqrt(2.0) * 0.5 * (sx + sy)
    radii = np.clip(radii, min_radius, max_radius if max_radius else np.inf)
    radii_out[valid_idx] = radii

    mask = np.zeros((T, H, W), dtype=dtype)
    r_int = np.ceil(radii).astype(np.int32)

    for r in np.unique(r_int):
        sel = np.where(r_int == r)[0]   # positions within `sub` for this bucket
        size = 2 * r + 1
        yy, xx = np.ogrid[-r:r+1, -r:r+1]

        for i in sel:
            t = ct[i]; xi = int(round(cx[i])) - r; yi = int(round(cy[i])) - r
            dx = cx[i] - round(cx[i]); dy = cy[i] - round(cy[i])
            disk = (xx - dx)**2 + (yy - dy)**2 <= radii[i]**2

            x_start = max(0, xi); x_end = min(W, xi + size)
            y_start = max(0, yi); y_end = min(H, yi + size)
            dx_start = x_start - xi; dx_end = dx_start + (x_end - x_start)
            dy_start = y_start - yi; dy_end = dy_start + (y_end - y_start)
            if x_end <= x_start or y_end <= y_start:
                continue

            local_disk = disk[dy_start:dy_end, dx_start:dx_end]
            target = mask[t, y_start:y_end, x_start:x_end]
            np.copyto(target, labels[i], where=local_disk & (target == 0))

            # Intensity stats — only if intensity_stack provided
            i_global = valid_idx[i]
            area_out[i_global] = int(local_disk.sum())
            if intensity_stack is not None and local_disk.any():
                disk_pixels = intensity_stack[t, y_start:y_end, x_start:x_end][local_disk]
                mean_out[i_global] = disk_pixels.mean()
                max_out[i_global]  = disk_pixels.max()

    df_out = df.assign(
        spot_area=area_out, spot_mean=mean_out, spot_max=max_out,
        spot_radius=radii_out,
    )
    return mask, df_out

if __name__ == "__main__":
    import pandas as pd
    import tifffile
    import time
    

    df = pd.read_csv(r"test_blobs.csv")
    img = tifffile.imread(r"test_blotb_img.tif")
    # time_point_one_df = df[df["time"] == 1]
    # points = df[["y", "x"]].values
    print(df.info())
    print("Image shape:", img.shape)
    # Usage
    # stack = tifffile.imread("movie.tif")           # (T, H, W)
    # df = pd.read_csv("spottorch.stack_points.csv")        # columns: time, y, x
    # time it
    start = time.time()
    # result = extract_crops_vec(img, df, win=5)
    result = fit_all_frames(img, df, win=2, device='cuda', max_iter=50, chunk_size=20000)
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")
    result.to_csv("fitted_test_blobs_all.csv", index=False)

    mask, prop_df = make_spot_mask( stack_shape=img.shape, df=result, intensity_stack=img, min_radius=1.0,max_radius=5, label_col=None)
    
    print("Mask shape:", mask.shape)
    # print("Props shape:", props.shape)
    tifffile.imwrite("fitted_test_blobs_mask.tif", mask)
    prop_df.to_csv("fitted_test_blobs_mask_props.csv", index=False)
    # fitted = fit_gaussians(img, points, win=5, device='cuda')

    # fitted_df = pd.DataFrame(fitted, columns=["amplitude", "x", "y", "sigma_x", "sigma_y", "average_sigma", "background"])
    # # # print(fitted_df)
    # # # save to csv if you want: 
    # fitted_df.to_csv("fitted_test_blobs.csv", index=False)
