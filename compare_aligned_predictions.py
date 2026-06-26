"""Compare predictions from original and Procrustes-aligned Tucker pickles.

Edit ORIGINAL_PKL and ALIGNED_PKL, then run:
    python compare_aligned_predictions.py
"""
import os
import pickle

import numpy as np
import jax.numpy as jnp

from pyinn.model import INN_gaussian, INN_linear, INN_nonlinear


ORIGINAL_PKL = "pyinn/model_saved/frac_pull_xz_1000_gaussian2_model.pkl"
ALIGNED_PKL = "pyinn/model_saved/frac_pull_xz_1000_gaussian2_model_aligned.pkl"
TARGET_CSV = "data/frac_pull_xz_1000.csv"
GRID_PLOT_PATH = "pyinn/model_saved/alignment_full_grid_slice.png"

BATCH_SIZE = 50000

# None means use the trained model grid size: nelem + 1.
FULL_GRID_NX = None
FULL_GRID_NZ = None
FULL_GRID_NT = None
TIME_SLICE_INDEX = None  # None means middle time slice.


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_grid(config):
    dim = len(config["DATA_PARAM"]["input_col"])
    nelem = int(config["MODEL_PARAM"]["nelem"])
    nnode = nelem + 1
    return [jnp.linspace(0.0, 1.0, nnode, dtype=jnp.float64) for _ in range(dim)]


def make_model(config):
    grid = make_grid(config)
    interp_method = config["interp_method"]
    if interp_method == "linear":
        return INN_linear(grid, config)
    if interp_method == "nonlinear":
        return INN_nonlinear(grid, config)
    if interp_method in ("gaussian", "gaussian2"):
        return INN_gaussian(grid, config)
    raise ValueError(f"Unsupported interp_method: {interp_method}")


def predict_batched(model_data, x, batch_size=BATCH_SIZE, model=None):
    if model is None:
        model = make_model(model_data["config"])
    preds = []
    for i0 in range(0, len(x), batch_size):
        batch = x[i0:i0 + batch_size]
        preds.append(np.asarray(model.v_forward(model_data["params"], jnp.asarray(batch))))
    return np.concatenate(preds, axis=0)


def get_grid_size(config):
    nnode = int(config["MODEL_PARAM"]["nelem"]) + 1
    nx = FULL_GRID_NX or nnode
    nz = FULL_GRID_NZ or nnode
    nt = FULL_GRID_NT or nnode
    return nx, nz, nt


def make_full_grid(config):
    nx, nz, nt = get_grid_size(config)
    xs = np.linspace(0.0, 1.0, nx)
    zs = np.linspace(0.0, 1.0, nz)
    ts = np.linspace(0.0, 1.0, nt)
    t_grid, x_grid, z_grid = np.meshgrid(ts, xs, zs, indexing="ij")
    x = np.stack([x_grid, z_grid, t_grid], axis=-1).reshape(-1, 3)
    return x, xs, zs, ts


def infer_target_csv(model_data):
    if TARGET_CSV is not None:
        return TARGET_CSV
    data_name = model_data["config"].get("data_name")
    if data_name is None:
        return None
    return os.path.join("data", f"{data_name}.csv")


def load_target_slice(model_data, target_t):
    target_csv = infer_target_csv(model_data)
    if target_csv is None or not os.path.exists(target_csv):
        print(f"Target CSV not found; skipping target plot: {target_csv}")
        return None

    config = model_data["config"]
    data = np.loadtxt(target_csv, delimiter=",", dtype=np.float64, skiprows=1)
    x_cols = config["DATA_PARAM"]["input_col"]
    u_col = config["DATA_PARAM"]["output_col"][0]
    x_data = data[:, x_cols]
    u_data = data[:, u_col]

    if config["DATA_PARAM"].get("bool_normalize", False):
        x_min = np.asarray(model_data["x_data_minmax"]["min"])
        x_max = np.asarray(model_data["x_data_minmax"]["max"])
        u_min = np.asarray(model_data["u_data_minmax"]["min"])[0]
        u_max = np.asarray(model_data["u_data_minmax"]["max"])[0]
        x_data = (x_data - x_min) / (x_max - x_min)
        u_data = (u_data - u_min) / (u_max - u_min)

    t_values = np.unique(x_data[:, 2])
    nearest_t = t_values[np.argmin(np.abs(t_values - target_t))]
    rows = np.isclose(x_data[:, 2], nearest_t)

    xs_target = np.unique(x_data[rows, 0])
    zs_target = np.unique(x_data[rows, 1])
    target = np.full((len(xs_target), len(zs_target)), np.nan)
    x_index = {value: idx for idx, value in enumerate(xs_target)}
    z_index = {value: idx for idx, value in enumerate(zs_target)}

    for xzt, value in zip(x_data[rows], u_data[rows]):
        target[x_index[xzt[0]], z_index[xzt[1]]] = value

    extent = [
        float(zs_target[0]),
        float(zs_target[-1]),
        float(xs_target[0]),
        float(xs_target[-1]),
    ]
    return target, extent, nearest_t


def save_full_grid_slice_plot(pred_original, pred_aligned, nx, nz, nt, xs, zs, ts, path, model_data):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping full-grid slice plot.")
        return

    pred_original = pred_original.reshape(nt, nx, nz)
    pred_aligned = pred_aligned.reshape(nt, nx, nz)
    diff = pred_original - pred_aligned

    it = TIME_SLICE_INDEX if TIME_SLICE_INDEX is not None else nt // 2
    extent = [float(zs[0]), float(zs[-1]), float(xs[0]), float(xs[-1])]
    vmax = max(abs(float(diff[it].min())), abs(float(diff[it].max())))
    target_slice = load_target_slice(model_data, ts[it])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.ravel()
    im0 = axes[0].imshow(pred_original[it], origin="lower", extent=extent, aspect="auto")
    axes[0].set_title("Original")
    im1 = axes[1].imshow(pred_aligned[it], origin="lower", extent=extent, aspect="auto")
    axes[1].set_title("Aligned")
    im2 = axes[2].imshow(diff[it], origin="lower", extent=extent, aspect="auto", cmap="coolwarm",
                         vmin=-vmax if vmax else None, vmax=vmax if vmax else None)
    axes[2].set_title("Original - aligned")
    if target_slice is not None:
        target, target_extent, nearest_t = target_slice
        im3 = axes[3].imshow(target, origin="lower", extent=target_extent, aspect="auto")
        axes[3].set_title(f"Target (nearest t={nearest_t:.4f})")
    else:
        im3 = None
        axes[3].axis("off")
        axes[3].set_title("Target unavailable")

    for ax in axes:
        ax.set_xlabel("z")
        ax.set_ylabel("x")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    fig.colorbar(im2, ax=axes[2], shrink=0.85)
    if im3 is not None:
        fig.colorbar(im3, ax=axes[3], shrink=0.85)
    fig.suptitle(f"Full-grid prediction slice at t={ts[it]:.4f}")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved full-grid slice plot to: {path}")


def main():
    original = load_pkl(ORIGINAL_PKL)
    aligned = load_pkl(ALIGNED_PKL)
    original_model = make_model(original["config"])
    aligned_model = make_model(aligned["config"])

    full_x, xs, zs, ts = make_full_grid(original["config"])
    nx, nz, nt = len(xs), len(zs), len(ts)
    print(f"Evaluating full grid with shape nt x nx x nz = {nt} x {nx} x {nz}")
    full_original = predict_batched(original, full_x, model=original_model).reshape(-1)
    full_aligned = predict_batched(aligned, full_x, model=aligned_model).reshape(-1)
    full_diff = full_original - full_aligned

    print("Full-grid comparison:")
    print(f"max abs diff: {np.max(np.abs(full_diff)):.6e}")
    print(f"mean abs diff: {np.mean(np.abs(full_diff)):.6e}")
    print(f"RMSE diff: {np.sqrt(np.mean(full_diff ** 2)):.6e}")
    save_full_grid_slice_plot(full_original, full_aligned, nx, nz, nt, xs, zs, ts, GRID_PLOT_PATH, original)


if __name__ == "__main__":
    main()
