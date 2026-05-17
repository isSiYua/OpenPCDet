# Resource-Aware Optimization of PointPillars for KITTI 3D Object Detection

This repository contains the code used for a group project on LiDAR-based 3D object detection with PointPillars on the KITTI dataset.

The project is based on OpenPCDet and investigates resource-aware optimization of the original PointPillars pipeline. The main focus is not only to improve validation AP, but also to understand why some modifications help, why some fail, and how accuracy, reproducibility, and inference efficiency interact in a realistic GPU-limited setting.

---

## 1. Project Overview

The project studies PointPillars for KITTI-format 3D object detection and evaluates a sequence of architecture-level, loss-level, augmentation-level, and analysis-level modifications.

The main experimental line is:

```text
V0  Baseline PointPillars
V1  SiLU activation
V2  SiLU + PillarSE
V3  SiLU + CBAM
V4  SiLU + CBAM + PillarSE
V5  Naive DIoU auxiliary loss
V6  DIoU / direction loss weight tuning
V7  Class-decoupled DIoU, best stable model
V8  MobileBEVBackbone
V9  MobileBEVBackbone + augmentation, best observed run
V10 Augmentation / batch-size / repeatability experiments
```

The final report distinguishes two important models:

- **V7**: best stable and interpretable model
- **V9**: best observed single run, but less reproducible due to augmentation and training variance

---

## 2. Main Contributions

The main code-level contributions are:

1. SiLU activation replacement in PointPillars modules
2. PillarSE channel recalibration in PillarVFE / PFN
3. CBAM attention ablation in the BEV backbone
4. 3D DIoU auxiliary box regression loss
5. Class-decoupled DIoU weighting for Car, Pedestrian, and Cyclist
6. MobileBEVBackbone for lightweight BEV feature extraction
7. Augmentation ablation configurations
8. Parameter counting and computation analysis utilities
9. Qualitative frame-selection and BEV visualization tools

Detailed file-level changes are described in `Changes.md`.

---

## 3. Recommended Submission Contents

For WISEFlow submission, the ZIP archive should contain source code, configuration files, and documentation only.

Recommended structure:

```text
OpenPCDet/
  pcdet/
  tools/
  README.md
  Changes.md
```

Important files and folders:

```text
pcdet/models/backbones_3d/vfe/pillar_vfe.py
pcdet/models/backbones_2d/base_bev_backbone.py
pcdet/models/backbones_2d/mobile_bev_backbone.py
pcdet/models/backbones_2d/mobile_bev_backbone_v2.py
pcdet/models/backbones_2d/__init__.py
pcdet/models/dense_heads/anchor_head_template.py
tools/cfgs/kitti_models/
tools/count_params.py
tools/cal_flops.py
tools/find_vis_candidates.py
tools/draw_bev_compare.py
parse_log.py
```

Do **not** submit:

```text
data/
output/
checkpoints/
*.pth
*.pkl
*.bin
*.log
__pycache__/
*.pyc
TensorBoard logs
large generated evaluation outputs
```

The course instruction says trained models do not need to be submitted.

---

## 4. Environment

The experiments were conducted with an OpenPCDet-based environment.

Main environment information:

```text
Python: 3.8
PyTorch: 2.1.0 + CUDA 12.1
OpenPCDet: 0.6.0-based modified code
spconv-cu118: 2.3.6
numpy: 1.24.4
scipy: 1.10.1
numba: 0.58.1
```

The exact environment may vary depending on the GPU cluster setup.

---

## 5. Dataset Preparation

This project uses the KITTI 3D object detection dataset.

Expected KITTI structure:

```text
OpenPCDet/
  data/
    kitti/
      ImageSets/
      training/
        calib/
        image_2/
        label_2/
        velodyne/
      testing/
        calib/
        image_2/
        velodyne/
```

After preparing the dataset, generate KITTI info files using the standard OpenPCDet procedure.

Example:

```bash
cd OpenPCDet
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

If your OpenPCDet version expects the command to be run from `tools/`, follow the original OpenPCDet instructions for dataset preparation.

---

## 6. Important Configurations

The main experiment configs are located under:

```text
tools/cfgs/kitti_models/
```

Important configuration variants include:

```text
pointpillar.yaml
pointpillar_baseline_aug.yaml
pointpillar_v7_aug.yaml
pointpillar_mobile_diou_aug_v1.yaml
pointpillar_v7_mobile_v2.yaml
```

Depending on the local branch, the exact filenames may differ slightly. The core differences are described in `Changes.md`.

---

## 7. Training

Most main experiments were trained for 80 epochs.

Example two-GPU training command:

```bash
cd OpenPCDet/tools

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --nproc_per_node=2 \
  train.py \
  --launcher pytorch \
  --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --extra_tag baseline_real_gpu2_batch4
```

For a V7-style config:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --nproc_per_node=2 \
  train.py \
  --launcher pytorch \
  --cfg_file cfgs/kitti_models/pointpillar_v7_aug.yaml \
  --extra_tag v7_experiment
```

For MobileBEVBackbone experiments, use the corresponding MobileBEV config.

---

## 8. Evaluation

Example evaluation command:

```bash
cd OpenPCDet/tools

CUDA_VISIBLE_DEVICES=0 python test.py \
  --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../output/kitti_models/pointpillar/EXPERIMENT_TAG/ckpt/checkpoint_epoch_80.pth \
  --batch_size 1
```

To measure inference time:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../output/kitti_models/pointpillar/EXPERIMENT_TAG/ckpt/checkpoint_epoch_80.pth \
  --batch_size 1 \
  --infer_time
```

Reported examples from the project:

```text
Baseline: 4.837M params, 37.36 ms/frame, 23.87 it/s
V7:       4.837M params, 38.55 ms/frame, 23.12 it/s
Mobile:   2.166M params, 43.22 ms/frame, 21.56 it/s
```

---

## 9. Parameter Counting

Use `tools/count_params.py` to count model parameters.

Example:

```bash
cd OpenPCDet/tools

python count_params.py --cfg_file cfgs/kitti_models/pointpillars_v7.yaml
```

This script reports:

- total parameters
- trainable parameters
- module-level parameters:
  - VFE
  - BEV backbone
  - dense head

This was used to compare the original BEV backbone and the MobileBEVBackbone.

---

## 10. FLOPs / Computation Analysis

Use `tools/cal_flops.py` to estimate or inspect model computation cost.

Example:

```bash
cd OpenPCDet/tools

python cal_flops.py --cfg_file cfgs/kitti_models/pointpillar.yaml
```

This script was mainly used for the resource-aware comparison between the original `BaseBEVBackbone` and the proposed `MobileBEVBackbone`.

---

## 11. Qualitative Visualization

The project includes tools for selecting and drawing qualitative examples.

### 11.1 Candidate Frame Selection

Use:

```bash
python find_vis_candidates.py
```

Purpose:

- compare KITTI ground truth with baseline and improved predictions
- count objects by class
- find frames where the improved model behaves differently from the baseline
- select success and limitation cases for the report

Frames used in the report:

```text
004074: relatively clean success case
003114: complex crowded-scene / limitation case
```

---

### 11.2 BEV Comparison Drawing

Use:

```bash
python draw_bev_compare.py \
  --gt_dir /path/to/KITTI/training/label_2 \
  --base_dir /path/to/baseline/final_result/data \
  --v7_dir /path/to/v7/final_result/data \
  --aug_dir /path/to/best_observed/final_result/data \
  --frames 004074 003114 \
  --score_th 0.5
```

Purpose:

- draw BEV comparison panels
- compare baseline, V7, and V9 / best observed predictions
- visualize ground-truth and predicted boxes
- generate report-ready qualitative figures

The generated images can be used directly in the final report.

---

## 12. Log Parsing

Use `parse_log.py` to extract and summarize AP values from evaluation logs.

Purpose:

- avoid manually copying many AP numbers from log files
- standardize result comparison across ablation runs
- help build tables for the final report

Example:

```bash
python parse_log.py path/to/evaluation_log.txt
```

If the local script interface differs, check the script header or argument parser.

---

## 13. Main Results Summary

The main report conclusion is:

- **V7** is the best stable model.
- **V9** is the best observed run.
- MobileBEVBackbone reduces parameter count but does not improve measured inference latency.
- CBAM and naive DIoU are useful negative ablations.
- Data augmentation can improve some runs but also introduces reproducibility concerns.

Key efficiency numbers:

```text
Baseline parameters: 4.837M
MobileBEV parameters: 2.166M
Parameter reduction: approximately 55.2%
```

---

## 14. Reproducibility Notes

Not every experiment used a fixed random seed. Some augmentation experiments were repeated and showed run-to-run variation.

Therefore:

- V7 is reported as the stable and interpretable improved model.
- V9 is reported as the best observed run, not the most reproducible final architecture.

For a stricter future study, each major variant should be trained with multiple fixed random seeds and reported as mean ± standard deviation.

---

## 15. Authors and Contributions
- **Zhenwei Zhang**: led the overall OpenPCDet-based PointPillars experimental framework, later-stage model optimization, core loss-function improvement, final analysis, visualization, and final report revision. Zhenwei supported the early V1--V4 PointPillars ablations and then mainly led the later optimization stages, including naive DIoU, DIoU weight tuning, class-decoupled DIoU, MobileBEVBackbone, augmentation experiments, parameter and inference-time analysis, qualitative visualization, result verification, and final report formatting and polishing.

- **Qi Fan Andrea Pan**: contributed to advanced-model benchmarking and early-stage PointPillars ablation experiments. Qi Fan worked on both TANet and CenterPoint training/evaluation, which provided important comparison baselines for model selection. Qi Fan also contributed to the implementation and testing of the early PointPillars variants from V1 to V4, including the initial SiLU, PillarSE, and CBAM-related ablation experiments.

- **Shuveccha Barua**: assisted with the early TANet exploration together with Qi Fan and contributed an initial data augmentation idea for the project. Shuveccha also contributed to the early report drafting process, including background writing, report organization, and formatting support.

---

## 16. Notes for Evaluators

This repository is intended to show the implementation and experimental workflow. It does not include trained model weights or the KITTI dataset.

Please refer to:

- `Changes.md` for file-level modifications
- this `README.md` for setup, training, evaluation, and visualization usage
- the final project report for detailed results and interpretation
