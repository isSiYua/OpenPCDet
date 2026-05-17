# PointPillars Optimization Changelog

This document summarizes the code-level modifications made to the original OpenPCDet PointPillars pipeline during the project.

The changes include activation replacement, attention-based feature recalibration, DIoU-based geometric loss integration, backbone replacement experiments, augmentation-based training ablations, model analysis utilities, and qualitative visualization tools.

No trained model checkpoints are included in this submission.

---

## 1. Baseline Reference

The project started from the original OpenPCDet PointPillars implementation for KITTI.

### Reference configuration

- `tools/cfgs/kitti_models/pointpillar.yaml`

### Purpose

- Used as the initial OpenPCDet PointPillars reference.
- Used to reproduce the baseline before adding architecture-level, loss-level, and augmentation-level modifications.
- The final fair baseline was trained for 80 epochs using two GPUs, batch size 4 per GPU, and total batch size 8.

---

## 2. Activation Function Modification: ReLU to SiLU

### Goal

To reduce the hard truncation effect of ReLU and preserve weak geometric features in sparse 3D point cloud representations.

### Modified files

- `pcdet/models/backbones_3d/vfe/pillar_vfe.py`
- `pcdet/models/backbones_2d/base_bev_backbone.py`

### Details

- Replaced `nn.ReLU()` with `nn.SiLU()` in the pillar feature extraction pipeline.
- Replaced `nn.ReLU()` with `nn.SiLU()` in the BEV backbone blocks and deblocks.
- Kept the original module interfaces compatible with OpenPCDet configuration files.

### Related experiments

- V1: SiLU
- V2: SiLU + PillarSE
- V3/V4: SiLU + CBAM variants
- V7: SiLU + PillarSE + class-decoupled DIoU

---

## 3. Attention-Based Feature Recalibration

### 3.1 PillarSE Channel Recalibration

### Goal

To enhance informative pillar-level feature channels while avoiding aggressive spatial suppression.

### Modified file

- `pcdet/models/backbones_3d/vfe/pillar_vfe.py`

### Details

- Added a lightweight SE-style channel recalibration module in the pillar feature extraction stage.
- Applied channel-wise feature recalibration after the linear transformation and normalization steps in the PFN pipeline.
- Used a reduction ratio of 4.
- Kept the original PillarVFE output interface unchanged.

### Notes

- This experiment was retained as one of the useful feature enhancement directions.
- It was primarily intended to improve small-object and weak-feature representation.

### Related experiments

- V2: SiLU + PillarSE
- V4: SiLU + CBAM + PillarSE
- V7: SiLU + PillarSE + class-decoupled DIoU

---

### 3.2 CBAM in BEV Backbone

### Goal

To test whether BEV features benefit from both channel and spatial attention.

### Modified file

- `pcdet/models/backbones_2d/base_bev_backbone.py`

### Details

- Added attention-related modules such as:
  - `ChannelAttention`
  - `SpatialAttention`
  - `CBAMBlock`
- Inserted CBAM into the BEV backbone for ablation testing.
- Evaluated CBAM both without and with PillarSE.

### Notes

- This experiment was used for analysis only.
- Spatial attention did not consistently help on sparse BEV pseudo-images and was not retained as the final design.
- The negative result was kept as an ablation because it helped motivate the final design choice.

### Related experiments

- V3: SiLU + CBAM
- V4: SiLU + CBAM + PillarSE

---

## 4. DIoU-Based Geometric Loss Integration

### Goal

To improve box regression by introducing a geometric auxiliary loss that explicitly considers center distance, instead of relying only on the original residual regression loss.

### Core modified file

- `pcdet/models/dense_heads/anchor_head_template.py`

### Related config file

- `tools/cfgs/kitti_models/pointpillar.yaml`

### Details

- Extended the box regression loss computation in `get_box_reg_layer_loss(...)`.
- Decoded predicted box residuals and target residuals into 3D boxes using:
  - `self.box_coder.decode_torch(...)`
- Introduced DIoU-based auxiliary loss computation for positive anchors.
- Added the DIoU term into the final box regression loss.
- Logged the DIoU term into `tb_dict` for monitoring.

### Config-side changes

The PointPillars loss configuration was extended with:

```yaml
USE_DIOU: True
diou_weight: ...
```

### Related experiments

- V5: naive DIoU
- V6: DIoU and direction-loss weight tuning
- V7: class-decoupled DIoU

---

## 5. Class-Decoupled DIoU Weighting

### Goal

To avoid using a single geometric loss weight for all classes, since Car, Pedestrian, and Cyclist have different box geometries and different sensitivity to center alignment versus orientation.

### Core modified file

- `pcdet/models/dense_heads/anchor_head_template.py`

### Related config file

- `tools/cfgs/kitti_models/pointpillar.yaml`

### Details

- Extracted positive-sample class labels during regression loss computation.
- Built class-aware weighting masks for the DIoU loss.
- Changed DIoU weighting from a single scalar to class-specific weights.
- Applied the class-specific DIoU weight only to positive anchors.

### Final class-decoupled DIoU setting

```yaml
diou_weight: [0.05, 0.2, 0.05]
dir_weight: 0.2
```

The three DIoU weights correspond to:

- Car: 0.05
- Pedestrian: 0.20
- Cyclist: 0.05

### Notes

- This was the main retained loss-level contribution in the final V7 system.
- It was introduced to reduce class interference and improve stability of geometric supervision across categories.
- The DIoU term is used only during training, so it does not add inference-time layers or parameters.

### Related experiment

- V7: best stable model

---

## 6. Direction Loss Weight Ablations

### Goal

To study the interaction between direction classification loss and DIoU-based geometric loss.

### Modified files

- `pcdet/models/dense_heads/anchor_head_template.py`
- `tools/cfgs/kitti_models/pointpillar.yaml`

### Details

- Tested multiple `dir_weight` settings, including:
  - `0.2`
  - `0.3`
  - `0.5`
- Explored whether class-sensitive loss balancing was needed when DIoU was enabled.
- Compared globally shared direction-loss weighting against class-sensitive geometric loss weighting.

### Notes

- These were ablation experiments used to reach the final V7 design.
- The final retained version kept:
  - shared `dir_weight`
  - class-decoupled `diou_weight`

---

## 7. MobileNet-Style BEV Backbone Experiment

### Goal

To test whether a lighter BEV backbone could reduce computation cost and improve deployment efficiency.

### New file

- `pcdet/models/backbones_2d/mobile_bev_backbone.py`

### Registration file

- `pcdet/models/backbones_2d/__init__.py`

### Related config files

- `tools/cfgs/kitti_models/pointpillar.yaml`
- `tools/cfgs/kitti_models/pointpillar_mobile_diou_aug_v1.yaml`

### Details

- Implemented `MobileBEVBackbone` as a MobileNet-style lightweight alternative to `BaseBEVBackbone`.
- Used depthwise separable convolution and inverted-residual-style design ideas.
- Registered the backbone in the OpenPCDet backbone registry.
- Switched the PointPillars pipeline to use the new mobile backbone through config selection.
- Preserved compatibility with the original PointPillars dense head and BEV feature interface.

### Notes

- This was an architecture exploration for speed and efficiency.
- Although parameter count was reduced, measured GPU runtime did not consistently improve.
- This experiment was retained as an ablation and resource-aware analysis, not as the final stable model.

### Related experiments

- V8: MobileBEVBackbone
- V9: MobileBEVBackbone + augmentation

---

## 8. Conservative Mobile Backbone Retry

### Goal

To re-test MobileNet-style backbone replacement with a more conservative design after observing performance instability in the initial mobile backbone.

### New file

- `pcdet/models/backbones_2d/mobile_bev_backbone_v2.py`

### Registration file

- `pcdet/models/backbones_2d/__init__.py`

### Related config file

- `tools/cfgs/kitti_models/pointpillar_v7_mobile_v2.yaml`

### Details

- Implemented a revised, more conservative mobile-style BEV backbone.
- Kept output channels and multi-scale fusion dimensions unchanged for fair comparison.
- Introduced a cleaner MobileNet-style replacement attempt based on the V7 pipeline.

### Notes

- This experiment served as a follow-up architecture retry.
- It was evaluated as an experimental branch and was not retained as the final model.

---

## 9. Data Augmentation Experiments

### Goal

To test whether additional global scene perturbation could improve generalization beyond the default PointPillars augmentation pipeline.

### Modified / new config files

- `tools/cfgs/kitti_models/pointpillar_baseline_aug.yaml`
- `tools/cfgs/kitti_models/pointpillar_v7_aug.yaml`
- `tools/cfgs/kitti_models/pointpillar_mobile_diou_aug_v1.yaml`

### Details

The standard PointPillars training augmentations include:

- `gt_sampling`
- `random_world_flip`
- `random_world_rotation`
- `random_world_scaling`

Additional augmentation was also tested:

```yaml
- NAME: random_world_translation
  NOISE_TRANSLATE_STD: [0.2, 0.2, 0.0]
```

### Notes

- The augmentation was tested on:
  - original baseline
  - V7
  - mobile backbone version
- Its effectiveness was configuration-dependent rather than universally beneficial.
- The strongest single run was reported as V9 best observed, but repeated runs showed variability.

---

## 10. Baseline + Augmentation Configuration

### Goal

To test the added augmentation on the original OpenPCDet PointPillars baseline without DIoU or mobile backbone changes.

### Config file

- `tools/cfgs/kitti_models/pointpillar_baseline_aug.yaml`

### Details

- Created from the original PointPillars baseline configuration.
- Kept:
  - `BaseBEVBackbone`
  - original loss settings
  - original direction loss setup
- Added the same augmentation setting used in later augmentation experiments.

---

## 11. V7 + Augmentation Configuration

### Goal

To test whether the same augmentation remains beneficial after introducing class-decoupled DIoU weighting.

### Config file

- `tools/cfgs/kitti_models/pointpillar_v7_aug.yaml`

### Details

- Based on the V7 pipeline.
- Kept:
  - `BaseBEVBackbone`
  - DIoU enabled
  - class-decoupled `diou_weight`
- Added the same augmentation pipeline used in the baseline augmentation experiment.

---

## 12. Utility / Analysis Scripts

The following scripts were added to support experiment tracking, model analysis, and qualitative visualization.

### 12.1 Training Log Parser

### Utility file

- `parse_log.py`

### Details

- Extracts final AP values from training / evaluation logs.
- Helps standardize result comparison across multiple ablation runs.
- Used to summarize experiment outputs more efficiently.

---

### 12.2 Parameter Counting Script

### Utility file

- `tools/count_params.py`

### Details

- Builds the model from a given YAML config file.
- Reports total parameters and trainable parameters.
- Reports module-level parameter counts, including:
  - VFE
  - BEV backbone
  - dense head

### Example usage

```bash
python count_params.py --cfg_file cfgs/kitti_models/pointpillars_v7.yaml
```

### Used for

- Comparing the original BaseBEVBackbone model with MobileBEVBackbone.
- Reporting parameter reduction in the final report.

---

### 12.3 FLOPs / Computation Analysis Script

### Utility file

- `tools/cal_flops.py`

### Details

- Estimates or inspects computational cost for different PointPillars variants.
- Mainly used to compare the original BEV backbone and the MobileBEVBackbone.
- Supports the resource-aware discussion in the report.

---

### 12.4 Visualization Candidate Selection Script

### Utility file

- `tools/find_vis_candidates.py`

### Goal

To automatically search for KITTI validation frames that are suitable for qualitative visualization.

### Details

- Reads KITTI ground-truth label files.
- Reads prediction files from baseline and improved models.
- Applies a confidence-score threshold to filter predictions.
- Counts objects by class for each frame.
- Compares ground truth, baseline predictions, and improved-model predictions.
- Prints candidate frames where the improved model shows more useful behavior than the baseline.

### Used for

- Selecting qualitative visualization examples for the report.
- Finding both:
  - relatively clean success cases
  - complex / crowded limitation cases

### Example selected frames

- `004074`: relatively clean success case
- `003114`: complex crowded-scene / limitation case

---

### 12.5 BEV Comparison Drawing Script

### Utility file

- `tools/draw_bev_compare.py`

### Goal

To generate BEV qualitative comparison figures for selected KITTI frames.

### Details

- Reads KITTI ground-truth labels from `label_2`.
- Reads prediction results from baseline, V7, and V9 / best observed output folders.
- Converts KITTI-format 3D boxes into BEV rectangles.
- Draws top-down BEV comparison panels.
- Uses different line styles / colors for ground truth and predictions.
- Supports score-threshold filtering through command-line arguments.
- Saves visualization images for direct use in the final report.

### Example usage

```bash
python draw_bev_compare.py \
  --gt_dir /path/to/KITTI/training/label_2 \
  --base_dir /path/to/baseline/final_result/data \
  --v7_dir /path/to/v7/final_result/data \
  --aug_dir /path/to/best_observed/final_result/data \
  --frames 004074 003114 \
  --score_th 0.5
```

### Used for

- Generating qualitative report figures.
- Comparing:
  - baseline predictions
  - V7 stable model predictions
  - V9 best observed model predictions

---

### 12.6 Camera + BEV Qualitative Figure Generation

### Utility file

- `tools/draw_bev_compare.py`

or the corresponding visualization script used in the repository if the final local filename differs.

### Goal

To create clearer report-ready visualizations that combine camera image context and BEV prediction comparison.

### Details

- Uses KITTI frame IDs to load corresponding camera images and label files.
- Shows the original camera image for scene context.
- Shows BEV panels for baseline and improved models.
- Highlights qualitative differences between ground truth and predictions.
- Designed to make the report figures easier to understand than raw prediction tables.

### Notes

- If your final local script has a different name, replace this entry with the actual filename before submission.
- This script was used only for visualization and does not affect training or evaluation metrics.

---

## 13. Evaluation and Inference-Time Testing

The original OpenPCDet `test.py` script was used with `--infer_time` to measure inference speed.

### Example command

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --cfg_file path/to/pointpillar.yaml \
  --ckpt path/to/checkpoint_epoch_80.pth \
  --batch_size 1 \
  --infer_time
```

### Measured results used in the report

```text
Baseline: 4.837M params, 37.36 ms/frame, 23.87 it/s
V7:       4.837M params, 38.55 ms/frame, 23.12 it/s
Mobile:   2.166M params, 43.22 ms/frame, 21.56 it/s
```

### Main observation

- MobileBEVBackbone reduces parameter count but does not improve measured GPU inference latency.

---

## 14. Files Not Included in the Submission

The following files and folders are intentionally excluded from the submitted ZIP archive:

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

### Reason

- The course instruction says trained models do not need to be submitted.
- KITTI data and generated output folders are too large and are not part of the modified source code.
- The submitted archive should contain only code, configuration files, and documentation needed to understand and reproduce the experiments.

---

## 15. Main Final Models

### 15.1 V7: Best Stable Model

V7 combines:

- SiLU activation
- PillarSE feature recalibration
- Class-decoupled 3D DIoU auxiliary loss

It is treated as the most stable and interpretable improved model.

---

### 15.2 V9: Best Observed Run

V9 combines:

- V7-style DIoU loss
- MobileBEVBackbone
- data augmentation

It achieved the highest mean moderate 3D AP in our experiments, but repeated augmentation experiments showed stronger run-to-run variability. Therefore, it is reported as the best observed run rather than the most stable final architecture.

---

## 16. Main Modified Files Overview

### Core implementation files

- `pcdet/models/backbones_3d/vfe/pillar_vfe.py`
- `pcdet/models/backbones_2d/base_bev_backbone.py`
- `pcdet/models/dense_heads/anchor_head_template.py`
- `pcdet/models/backbones_2d/mobile_bev_backbone.py`
- `pcdet/models/backbones_2d/mobile_bev_backbone_v2.py`
- `pcdet/models/backbones_2d/__init__.py`

### Experiment config files

- `tools/cfgs/kitti_models/pointpillar.yaml`
- `tools/cfgs/kitti_models/pointpillar_baseline_aug.yaml`
- `tools/cfgs/kitti_models/pointpillar_v7_aug.yaml`
- `tools/cfgs/kitti_models/pointpillar_mobile_diou_aug_v1.yaml`
- `tools/cfgs/kitti_models/pointpillar_v7_mobile_v2.yaml`

### Utility and analysis files

- `parse_log.py`
- `tools/count_params.py`
- `tools/cal_flops.py`
- `tools/find_vis_candidates.py`
- `tools/draw_bev_compare.py`

---

## 17. Summary

Compared with the original OpenPCDet PointPillars codebase, this project adds:

- SiLU activation variants
- PillarSE channel recalibration
- CBAM attention ablation
- 3D DIoU auxiliary regression loss
- Class-decoupled DIoU weighting
- Direction loss weight ablations
- MobileBEVBackbone
- Revised MobileBEVBackbone retry
- Augmentation ablation configs
- Parameter counting utilities
- FLOPs / computation analysis utilities
- Qualitative frame selection tools
- BEV and camera-BEV visualization scripts

The main goal of these changes is to study resource-aware PointPillars optimization on KITTI and to analyze both successful and unsuccessful modification attempts.
