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

Training saves the model weights, loss history, and loss plot under `outputs/`.
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
keypoint counts, then writes
`outputs/results/baseline_comparison.json`. Missing result files or a missing
SimpleBaseline checkpoint are reported with the command and expected path
needed to continue.

## Visualization

```bash
python scripts/visualize_predictions.py --config configs/coco_simplebaseline.yaml
```

Visualization writes a grid of validation predictions to `outputs/`.
