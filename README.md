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

## Evaluation and visualization

```bash
python scripts/evaluate_pck.py --config configs/coco_simplebaseline.yaml
python scripts/visualize_predictions.py --config configs/coco_simplebaseline.yaml
```

Evaluation writes per-joint PCK metrics and a chart to `outputs/`.
Visualization writes a grid of validation predictions to `outputs/`.
