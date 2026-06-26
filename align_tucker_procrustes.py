"""Align saved INN Tucker factors with orthogonal Procrustes rotations.

Usage:
    Edit INPUT_PKL, REFERENCE_PKL, and OUTPUT_PKL below, then run:
        python align_tucker_procrustes.py

The output pickle has the same dictionary structure as the input pickle. Only
the Tucker core and selected factor matrices are rotated.
"""
import copy
import os
import pickle

import numpy as np


INPUT_PKL = "pyinn/model_saved/frac_pull_xz_1000_gaussian2_model.pkl"
REFERENCE_PKL = "pyinn/model_saved/align_test_ref_gaussian2_model.pkl"
OUTPUT_PKL = "pyinn/model_saved/frac_pull_xz_1000_gaussian2_model_aligned.pkl"

# Spatial dimensions are x,z by default.
ALIGNED_DIMS = [0, 1]


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(path, data):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def get_factor_matrix(factors, dim):
    """Return W^(d) with shape (J_d, R_d)."""
    if isinstance(factors, list):
        factor = np.asarray(factors[dim])
        if factor.ndim == 3:
            return factor[:, 0, :].T
        if factor.ndim == 2:
            return factor.T
        raise ValueError(f"Unsupported factor shape for dim {dim}: {factor.shape}")

    factors = np.asarray(factors)
    if factors.ndim == 4:
        return factors[:, dim, 0, :].T
    if factors.ndim == 3:
        return factors[:, dim, :].T
    raise ValueError(f"Unsupported factors shape: {factors.shape}")


def set_factor_matrix(factors, dim, matrix):
    """Set W^(d), where matrix has shape (J_d, R_d)."""
    updated = copy.deepcopy(factors)
    if isinstance(updated, list):
        factor = np.asarray(updated[dim])
        if factor.ndim == 3:
            factor = factor.copy()
            factor[:, 0, :] = matrix.T
            updated[dim] = factor
            return updated
        if factor.ndim == 2:
            updated[dim] = matrix.T
            return updated
        raise ValueError(f"Unsupported factor shape for dim {dim}: {factor.shape}")

    updated = np.asarray(updated).copy()
    if updated.ndim == 4:
        updated[:, dim, 0, :] = matrix.T
        return updated
    if updated.ndim == 3:
        updated[:, dim, :] = matrix.T
        return updated
    raise ValueError(f"Unsupported factors shape: {updated.shape}")


def mode_product(tensor, matrix, mode):
    """Compute tensor times_mode matrix.

    matrix has shape (new_mode_size, old_mode_size).
    """
    result = np.tensordot(matrix, tensor, axes=([1], [mode]))
    return np.moveaxis(result, 0, mode)


def procrustes_rotation(source, reference):
    """Solve min_Q ||source Q - reference||_F with Q^T Q = I."""
    if source.shape != reference.shape:
        raise ValueError(
            f"Source/reference factor shapes must match, got {source.shape} and {reference.shape}"
        )
    u, _, vt = np.linalg.svd(source.T @ reference, full_matrices=False)
    return u @ vt


def align_model(model_data, reference_data, aligned_dims):
    aligned = copy.deepcopy(model_data)

    params = aligned["params"]
    ref_params = reference_data["params"]
    if not isinstance(params, list) or len(params) != 2:
        raise ValueError("Expected input params to be [core, factors].")
    if not isinstance(ref_params, list) or len(ref_params) != 2:
        raise ValueError("Expected reference params to be [core, factors].")

    core = np.asarray(params[0]).copy()
    factors = params[1]
    ref_factors = ref_params[1]

    for dim in aligned_dims:
        source_w = get_factor_matrix(factors, dim)
        ref_w = get_factor_matrix(ref_factors, dim)
        q = procrustes_rotation(source_w, ref_w)

        factors = set_factor_matrix(factors, dim, source_w @ q)
        # If W is transformed as W Q, transform the core by Q^T on the same mode
        # so the represented function is unchanged.
        core = mode_product(core, q.T, dim)

    aligned["params"] = [core, factors]
    return aligned


def main():
    model_data = load_pkl(INPUT_PKL)
    reference_data = load_pkl(REFERENCE_PKL)
    aligned = align_model(model_data, reference_data, ALIGNED_DIMS)
    save_pkl(OUTPUT_PKL, aligned)

    print(f"Input model: {INPUT_PKL}")
    print(f"Reference model: {REFERENCE_PKL}")
    print(f"Aligned dimensions: {ALIGNED_DIMS}")
    print(f"Saved aligned model to: {OUTPUT_PKL}")


if __name__ == "__main__":
    main()
