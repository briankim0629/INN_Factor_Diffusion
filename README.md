# INN Tucker Training and Alignment

This repo trains Tucker models on fracture-style CSV datasets and saves the trained Tucker core and factor weights to a pickle file. The current workflow assumes Tucker decomposition, so the main things to choose are the dataset config, training hyperparameters, and optional Procrustes alignment paths.

## Setup

Activate the Python environment that has JAX, Optax, NumPy, scikit-learn, PyYAML, and Matplotlib installed.

```bash
conda activate binn-env
cd /Users/gyungmin/Desktop/fracture/pyinn_tucker
```

Run commands from the `pyinn_tucker` directory. Use module execution for training:

```bash
python -m pyinn.main
```

## Dataset and Config

Training is selected by `pyinn/settings.yaml`.

Set the dataset name:

```yaml
DATA:
  data_name : 'frac_pull_xz_1000'
```

The code then loads the matching config file:

```text
config/frac_pull_xz_1000.yaml
```

and the matching data file:

```text
data/frac_pull_xz_1000.csv
```

The CSV should contain input columns followed by the output column. For the fracture runs used here, the expected mapping is:

```text
x, z, t -> fracture
```

with the config using:

```yaml
DATA_PARAM:
  input_col : [0,1,2]
  output_col : [3]
```

## Training

Edit the dataset config in `config/<data_name>.yaml`. The most important training settings are:

```yaml
MODEL_PARAM:
  tucker_rank : [60, 60, 24]
  nelem : 200
  sigma_factor : 0.4

TRAIN_PARAM:
  num_epochs_INN : 80
  batch_size : 512
  learning_rate : 5e-4
```

Then run:

```bash
python -m pyinn.main
```

After training finishes, the model is saved automatically to:

```text
pyinn/model_saved/<data_name>_<interp_method>_model.pkl
```

For example:

```text
pyinn/model_saved/frac_pull_xz_1000_gaussian2_model.pkl
```

The saved pickle contains:

- `config`: the config used for the run, including `data_name` and `interp_method`
- `x_data_minmax`: input normalization min/max values
- `u_data_minmax`: output normalization min/max values
- `params`: Tucker parameters, where `params[0]` is the core tensor and `params[1]` stores the factor weights

## Aligning Tucker Weights

Use `align_tucker_procrustes.py` to align one trained model's spatial factor weights to a reference model. The script aligns the selected factors and transforms the Tucker core so the represented function stays unchanged.

Edit these paths at the top of `align_tucker_procrustes.py`:

```python
INPUT_PKL = "pyinn/model_saved/frac_pull_xz_1000_gaussian2_model.pkl"
REFERENCE_PKL = "pyinn/model_saved/align_test_ref_gaussian2_model.pkl"
OUTPUT_PKL = "pyinn/model_saved/frac_pull_xz_1000_gaussian2_model_aligned.pkl"
```

Choose which dimensions to align:

```python
ALIGNED_DIMS = [0, 1]
```

For the fracture dataset, `[0, 1]` means align the spatial dimensions `x` and `z`, while leaving time unchanged.

Run:

```bash
python align_tucker_procrustes.py
```

The aligned model is saved to `OUTPUT_PKL`.

## Verifying Alignment

Use `compare_aligned_predictions.py` to check that the original and aligned models give the same predictions.

Edit:

```python
ORIGINAL_PKL = "pyinn/model_saved/frac_pull_xz_1000_gaussian2_model.pkl"
ALIGNED_PKL = "pyinn/model_saved/frac_pull_xz_1000_gaussian2_model_aligned.pkl"
TARGET_CSV = "data/frac_pull_xz_1000.csv"
GRID_PLOT_PATH = "pyinn/model_saved/alignment_full_grid_slice.png"
```

Run:

```bash
python compare_aligned_predictions.py
```

The script prints the max absolute difference, mean absolute difference, and RMSE difference between the original and aligned model predictions. It also saves a grid-slice comparison plot to `GRID_PLOT_PATH`.