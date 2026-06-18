# 2D Human Pose Estimation

This project trains a ResNet-50 SimpleBaseline model to predict 17 COCO
keypoint heatmaps. The original exploratory notebook is kept at
`notebooks/PoseEstimation_Baseline.ipynb`; reusable code now lives in `src/`.

## Installation

Create and activate a Python virtual environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The default configuration downloads COCO val2017 images and keypoint
annotations into `coco/` on the first run.

## Training

Run commands from the repository root:

```bash
python scripts/train.py --config configs/coco_simplebaseline.yaml
```

Use `--device auto` or `--device cuda` to train on a CUDA-capable PyTorch
installation. The training config enables mixed precision and channels-last
memory format on CUDA, and keeps CPU training compatible.

Training saves a copy of the config, model weights, loss history, and loss plot
under `outputs/`.
The default settings match the notebook: 256x192 person crops, 64x48 target
heatmaps, MSE loss, 30 epochs, and backbone unfreezing before epoch 6.

## Baseline evaluation and comparison

Evaluate the trivial Average-Pose baseline:

```bash
python scripts/evaluate_trivial_baseline.py --config configs/coco_simplebaseline.yaml
```

This writes:

- `outputs/results/trivial_baseline_template.json`
- `outputs/results/trivial_baseline_pck.json`

Evaluate a trained SimpleBaseline checkpoint:

```bash
python scripts/evaluate_pck.py --config configs/coco_simplebaseline.yaml
```

By default this expects `outputs/simplebaseline_weights.pth` and writes the
metrics to `outputs/pck_results.json`. Use `--weights PATH` when the checkpoint
is stored elsewhere.

After both evaluations exist, create the comparison:

```bash
python scripts/compare_baselines.py --config configs/coco_simplebaseline.yaml
```

The comparison verifies that both results use the same PCK threshold and valid
keypoint counts, then writes:

- `outputs/results/pck_comparison.json`
- `outputs/figures/pck_comparison.png`
- `outputs/figures/per_joint_pck.png`

Missing result files or a missing SimpleBaseline checkpoint are reported with
the command and expected path needed to continue.

Run a small synthetic PCK sanity check:

```bash
python scripts/check_pck_sanity.py
```

Evaluate COCO OKS-based keypoint AP:

```bash
python scripts/evaluate_coco_ap.py --config configs/coco_simplebaseline.yaml
```

This writes:

- `outputs/results/simplebaseline_coco_results.json`
- `outputs/results/simplebaseline_coco_ap.json`

## Visualization

```bash
python scripts/visualize_predictions.py --config configs/coco_simplebaseline.yaml
```

Visualization selects high- and low-PCK validation examples and writes
`outputs/figures/prediction_examples.png`.
