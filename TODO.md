# TODO – 2D Human Pose Estimation Project

This document tracks the next development steps for our university deep learning project on **2D Human Pose Estimation with Heatmap Regression**.

The goal is to turn the current notebook-based prototype into a clean, reproducible project with:
- a clear repository structure,
- a true baseline comparison,
- standardized evaluation,
- qualitative visualizations,
- report-ready figures,
- and well-documented code.

---

## Current Project Status

The repository currently started from a notebook-based implementation:

- `PoseEstimation_Baseline.ipynb`
- basic COCO keypoint loading
- heatmap generation
- ResNet-based heatmap prediction model
- training loop
- PCK evaluation
- qualitative prediction visualizations

The next goal is to refactor this into a maintainable Python project and prepare it for final submission.

---

## Important Development Rules

Before working on a task:

1. Create or switch to a feature branch.
2. Pull the latest changes from `main`.
3. Keep each task focused.
4. Do not mix unrelated changes in one commit.
5. Run basic checks before committing.
6. Keep generated data, checkpoints, and output figures out of Git unless explicitly needed for the report.

Recommended basic checks:

```bash
git status
python -m compileall src scripts
```

Recommended commit style:

```bash
git add .
git commit -m "Short description of completed task"
```

---

## Suggested Repository Structure

Target structure:

```text
intro-to-dl-pose-estimation/
├── README.md
├── TODO.md
├── requirements.txt
├── configs/
│   └── coco_simplebaseline.yaml
├── notebooks/
│   └── PoseEstimation_Baseline.ipynb
├── scripts/
│   ├── train.py
│   ├── evaluate_pck.py
│   ├── evaluate_trivial_baseline.py
│   ├── evaluate_coco_ap.py
│   └── make_report_figures.py
├── src/
│   ├── data/
│   │   ├── coco_dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── simplebaseline.py
│   │   └── trivial_baseline.py
│   ├── training/
│   │   ├── train.py
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── pck.py
│   │   └── coco_eval.py
│   └── visualization/
│       └── visualize.py
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   └── results/
└── report/
    └── figures/
```

---

# Phase 1 – Refactor Notebook into Project Structure

## Task 1.1 – Move notebook code into modules

**Status:** In progress / to be checked

**Goal:**  
Refactor reusable notebook code into Python modules.

Expected files:

```text
src/data/coco_dataset.py
src/data/transforms.py
src/models/simplebaseline.py
src/training/train.py
src/evaluation/pck.py
src/visualization/visualize.py
scripts/train.py
scripts/evaluate_pck.py
scripts/visualize_predictions.py
configs/coco_simplebaseline.yaml
requirements.txt
```

**Acceptance criteria:**

- Notebook is preserved under `notebooks/`.
- Reusable code is moved into `src/`.
- Scripts are runnable from the repository root.
- No important working logic from the notebook is lost.
- Output files are saved under `outputs/`.
- Basic imports work.

**Check commands:**

```bash
python -m compileall src scripts
git status
```

**Suggested AI-agent prompt:**

```text
Review the refactored project structure.

Check:
1. Are all imports correct?
2. Can scripts be run from the repository root?
3. Is the original notebook preserved?
4. Are outputs written only to outputs/? 
5. Does python -m compileall src scripts pass?
6. Are there obvious duplicated code blocks?

Fix issues without adding new features.
```

---

## Task 1.2 – Add `.gitignore`

**Goal:**  
Prevent generated files, datasets, checkpoints, and virtual environments from being committed.

Suggested `.gitignore` entries:

```gitignore
.venv/
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/

data/
coco/
outputs/
*.pth
*.pt
*.ckpt

.DS_Store
.vscode/
.idea/
```

**Acceptance criteria:**

- Dataset folders are ignored.
- Checkpoints are ignored.
- Python cache files are ignored.
- Virtual environments are ignored.

---

# Phase 2 – Setup and Reproducibility

## Task 2.1 – Improve `requirements.txt`

**Goal:**  
Make dependency installation reproducible.

Expected dependencies:

```text
torch
torchvision
numpy
matplotlib
tqdm
Pillow
pycocotools
PyYAML
opencv-python
```

Only add additional dependencies if they are actually used.

**Acceptance criteria:**

- `pip install -r requirements.txt` works.
- README contains setup instructions.
- No unnecessary heavy dependencies are added.

---

## Task 2.2 – Add config-driven training

**Goal:**  
Move training parameters into `configs/coco_simplebaseline.yaml`.

Suggested config fields:

```yaml
dataset:
  img_dir: coco/val2017
  ann_file: coco/annotations/person_keypoints_val2017.json
  train_split: 0.85
  seed: 42

model:
  name: simplebaseline_resnet50
  num_keypoints: 17
  pretrained_backbone: true

training:
  batch_size: 8
  num_epochs: 10
  learning_rate: 0.001
  weight_decay: 0.0001
  freeze_backbone_epochs: 2

heatmaps:
  input_size: [256, 192]
  heatmap_size: [64, 48]
  sigma: 2

outputs:
  output_dir: outputs
```

**Acceptance criteria:**

- Training can be launched using:

```bash
python scripts/train.py --config configs/coco_simplebaseline.yaml
```

- Training saves:
  - checkpoint,
  - config copy,
  - training log,
  - loss curve.

---

# Phase 3 – Baselines

## Task 3.1 – Add true trivial baseline

**Goal:**  
Implement a simple non-neural baseline for comparison.

The baseline should estimate the average normalized keypoint position for each COCO joint from the training annotations and predict this average pose for each validation person.

Expected file:

```text
src/models/trivial_baseline.py
scripts/evaluate_trivial_baseline.py
```

**Acceptance criteria:**

- Invisible or missing keypoints are ignored.
- Average keypoint template can be saved as JSON.
- Trivial baseline can be evaluated with PCK.
- Results are saved to:

```text
outputs/results/trivial_baseline_pck.json
```

**Suggested AI-agent prompt:**

```text
Implement a true trivial pose baseline.

The baseline should:
1. Estimate average normalized keypoint positions from visible training annotations.
2. Predict this fixed average pose for each validation person crop.
3. Evaluate the predictions with the existing PCK evaluator.
4. Save overall and per-joint PCK as JSON.
5. Add a README command for running the baseline.

Do not change the learned model architecture.
```

---

## Task 3.2 – Compare trivial baseline and learned model

**Goal:**  
Create a clear baseline-vs-model comparison.

Expected outputs:

```text
outputs/results/pck_comparison.json
outputs/figures/pck_comparison.png
outputs/figures/per_joint_pck.png
```

**Acceptance criteria:**

- Trivial baseline and SimpleBaseline are evaluated on the same validation split.
- Overall PCK is reported.
- Per-joint PCK is reported.
- A comparison plot is generated.

---

# Phase 4 – Evaluation

## Task 4.1 – Standardize PCK evaluation

**Goal:**  
Make the PCK metric implementation clean, reusable, and consistent.

Expected file:

```text
src/evaluation/pck.py
```

**Acceptance criteria:**

The PCK evaluator should support:

- configurable threshold, default `0.2`,
- overall PCK,
- per-joint PCK,
- ignoring invisible keypoints,
- JSON-compatible result output.

**Suggested AI-agent prompt:**

```text
Refactor the PCK evaluation.

Create one clear function that computes:
- overall PCK
- per-joint PCK
- number of valid keypoints
- threshold used

The function should ignore invisible/missing keypoints and return a JSON-serializable dictionary.
Add a small synthetic sanity check script.
```

---

## Task 4.2 – Add COCO keypoint AP evaluation

**Goal:**  
Add dataset-standard COCO evaluation using OKS-based keypoint AP.

Expected files:

```text
src/evaluation/coco_eval.py
scripts/evaluate_coco_ap.py
```

**Important:**  
The model predicts keypoints in heatmap/crop coordinates. For COCO evaluation, predictions must be mapped back to original image coordinates.

Coordinate mapping chain:

```text
heatmap coordinates
→ network input coordinates
→ person crop coordinates
→ original image coordinates
```

**Acceptance criteria:**

- Uses `pycocotools.COCOeval` with `iouType="keypoints"`.
- Saves prediction JSON:

```text
outputs/results/simplebaseline_coco_results.json
```

- Reports:
  - AP,
  - AP50,
  - AP75,
  - AP medium,
  - AP large.

**Suggested AI-agent prompt:**

```text
Add COCO keypoint AP evaluation using pycocotools.

Implement prediction conversion to COCO keypoint JSON format:
{
  "image_id": int,
  "category_id": 1,
  "keypoints": [x1, y1, v1, ..., x17, y17, v17],
  "score": float
}

Map predicted heatmap keypoints back to original image coordinates using the person bounding box.
Use heatmap maximum values as keypoint confidence.
Print and save AP, AP50, AP75, AP_medium, and AP_large.
```

---

# Phase 5 – Visualization and Report Figures

## Task 5.1 – Qualitative prediction visualization

**Goal:**  
Visualize predicted COCO keypoints and skeleton limbs on real images.

Expected file:

```text
src/visualization/visualize.py
```

Expected output:

```text
outputs/figures/prediction_examples.png
```

**Acceptance criteria:**

- Draws visible keypoints.
- Draws COCO skeleton limbs.
- Shows both good and bad examples.
- Saves figures under `outputs/figures/`.

---

## Task 5.2 – Success and failure cases

**Goal:**  
Automatically select qualitative examples based on PCK score.

Expected outputs:

```text
report/figures/success_cases.png
report/figures/failure_cases.png
```

**Acceptance criteria:**

- Success cases have high PCK.
- Failure cases have low PCK.
- Visualizations are readable and report-ready.

---

## Task 5.3 – Generate report figures automatically

**Goal:**  
Create one script that generates all figures needed for the report and presentation.

Expected file:

```text
scripts/make_report_figures.py
```

Expected outputs:

```text
report/figures/training_loss_curve.png
report/figures/pck_comparison.png
report/figures/per_joint_pck.png
report/figures/success_cases.png
report/figures/failure_cases.png
report/figures/target_heatmap_example.png
```

**Acceptance criteria:**

- Figures can be regenerated from saved results/checkpoints.
- Filenames are stable.
- Figures are readable in the final report.

---

# Phase 6 – Experiments

## Task 6.1 – Run main experiment

**Goal:**  
Train the SimpleBaseline model and save results.

Suggested command:

```bash
python scripts/train.py --config configs/coco_simplebaseline.yaml
```

Expected outputs:

```text
outputs/checkpoints/best_model.pth
outputs/results/training_log.csv
outputs/figures/loss_curve.png
```

Record:

```text
Date:
Commit hash:
Config:
Number of epochs:
Final train loss:
Final validation loss:
Best validation loss:
Overall PCK:
COCO AP:
Notes:
```

---

## Task 6.2 – Optional heatmap sigma ablation

**Goal:**  
Compare different Gaussian sigma values for target heatmaps.

Suggested experiments:

```text
sigma = 1
sigma = 2
sigma = 3
```

**Acceptance criteria:**

- Same train/validation split.
- Same model architecture.
- Results saved separately.
- Short discussion added to report.

---

## Task 6.3 – Optional backbone freeze ablation

**Goal:**  
Compare training with and without frozen ResNet backbone in early epochs.

Suggested experiments:

```text
freeze_backbone_epochs = 0
freeze_backbone_epochs = 2
```

**Acceptance criteria:**

- Same dataset split.
- Same number of epochs.
- Compare validation loss and PCK.
- Add result to report if useful.

---

# Phase 7 – README and Documentation

## Task 7.1 – Improve README

**Goal:**  
Make the repository understandable for reviewers and teammates.

README should include:

1. Project title
2. Short description
3. Dataset description
4. Method overview
5. Repository structure
6. Installation
7. Dataset preparation
8. Training command
9. Evaluation commands
10. Visualization commands
11. Results summary
12. Known limitations

**Important note:**  
If only `COCO val2017` is used and split into train/validation subsets, state this clearly as a subset experiment.

---

## Task 7.2 – Add result table to README

**Goal:**  
Summarize final metrics.

Suggested table:

```markdown
| Method | PCK@0.2 | COCO AP | AP50 | AP75 | Notes |
|---|---:|---:|---:|---:|---|
| Trivial average-pose baseline | TBD | N/A | N/A | N/A | Fixed average pose |
| SimpleBaseline ResNet heatmap model | TBD | TBD | TBD | TBD | ResNet + deconv heatmap head |
```

---

# Phase 8 – Report

## Task 8.1 – Draft report structure

Suggested report outline:

```text
1. Introduction
2. Dataset and preprocessing
3. Methods
   3.1 Trivial baseline
   3.2 Heatmap regression model
   3.3 Loss function
4. Experimental setup
5. Results
   5.1 Quantitative results
   5.2 Qualitative results
6. Discussion
7. Conclusion
8. References
```

---

## Task 8.2 – Write method section

Include:

- COCO keypoints
- person crops
- input resolution
- heatmap resolution
- Gaussian heatmap generation
- ResNet/SimpleBaseline architecture
- MSE heatmap loss
- coordinate decoding from heatmaps

---

## Task 8.3 – Write evaluation section

Include:

- PCK@0.2
- per-joint PCK
- COCO AP if implemented
- success/failure case analysis

---

# Phase 9 – Presentation

## Task 9.1 – Prepare presentation structure

Suggested slide outline:

```text
1. Title and team
2. Problem: 2D human pose estimation
3. Dataset: COCO keypoints subset
4. Method overview
5. Heatmap target generation
6. Model architecture
7. Baselines
8. Quantitative results
9. Qualitative success cases
10. Failure cases and limitations
11. Conclusion
```

---

# Open Questions

Use this section to track unresolved decisions.

- [ ] Are we using only COCO `val2017`, or also `train2017`?
- [ ] Which machine/GPU will run final training?
- [ ] How many epochs are feasible?
- [ ] Do we include COCO AP in final results?
- [ ] Which team member is responsible for report writing?
- [ ] Which team member is responsible for presentation slides?

---

# Team Task Assignment

| Task Area | Responsible Person | Status | Notes |
|---|---|---|---|
| Refactor notebook | TBD | In progress |  |
| Dataset/data pipeline | TBD | Open |  |
| SimpleBaseline model | TBD | Open |  |
| Trivial baseline | TBD | Open |  |
| PCK evaluation | TBD | Open |  |
| COCO AP evaluation | TBD | Open |  |
| Visualizations | TBD | Open |  |
| README | TBD | Open |  |
| Report | TBD | Open |  |
| Presentation | TBD | Open |  |

---

# Suggested Next Immediate Steps

1. Finish and review the notebook refactor.
2. Run:

```bash
python -m compileall src scripts
```

3. Commit the refactor.
4. Add or verify `.gitignore`.
5. Implement the trivial average-pose baseline.
6. Standardize PCK evaluation.
7. Add COCO AP evaluation if time allows.
8. Generate final figures for report and presentation.

---

# Definition of Done

The project is ready for submission when:

- [ ] Code is organized in `src/` and `scripts/`.
- [ ] README explains how to run the project.
- [ ] Dataset paths are documented.
- [ ] At least one true baseline is implemented.
- [ ] SimpleBaseline model can be trained.
- [ ] PCK evaluation works.
- [ ] COCO AP evaluation works or its absence is clearly justified.
- [ ] Qualitative predictions are visualized.
- [ ] Report figures are generated.
- [ ] Final metrics are documented.
- [ ] Report is complete.
- [ ] Presentation is complete.
