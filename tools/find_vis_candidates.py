from pathlib import Path
from collections import Counter

GT_DIR = Path("/home/cv10f26/Group10/Datasets/data/training/label_2")
BASE_DIR = Path("/home/cv10f26/Group10/PointPillars/OpenPCDet/output/output/kitti_models/pointpillar/default/pointpillar/default/eval/epoch_80/val/default/final_result/data")
IMP_DIR = Path("/home/cv10f26/Group10/PointPillars/OpenPCDet/output/output/kitti_models/pointpillar/diou_loss_6/pointpillar/v7_diou_eval/eval/epoch_80/val/default/final_result/data")

CLASSES = ["Car", "Pedestrian", "Cyclist"]
SCORE_TH = 0.3


def read_gt(path):
    counts = Counter()
    if not path.exists():
        return counts
    for line in path.read_text().strip().splitlines():
        if not line:
            continue
        cls = line.split()[0]
        if cls in CLASSES:
            counts[cls] += 1
    return counts


def read_pred(path, score_th=0.3):
    counts = Counter()
    if not path.exists():
        return counts
    for line in path.read_text().strip().splitlines():
        if not line:
            continue
        parts = line.split()
        cls = parts[0]
        if cls not in CLASSES:
            continue
        try:
            score = float(parts[-1])
        except Exception:
            score = 1.0
        if score >= score_th:
            counts[cls] += 1
    return counts


rows = []

for gt_file in sorted(GT_DIR.glob("*.txt")):
    frame = gt_file.stem
    gt = read_gt(gt_file)
    base = read_pred(BASE_DIR / f"{frame}.txt", SCORE_TH)
    imp = read_pred(IMP_DIR / f"{frame}.txt", SCORE_TH)

    # Focus on Pedestrian and Cyclist because they are harder and more informative.
    score = 0
    for cls in ["Pedestrian", "Cyclist"]:
        gt_n = gt[cls]
        if gt_n == 0:
            continue

        base_err = abs(base[cls] - gt_n)
        imp_err = abs(imp[cls] - gt_n)

        if imp_err < base_err:
            score += 3
        if imp[cls] > base[cls]:
            score += 1

    if score > 0:
        rows.append((score, frame, gt, base, imp))

rows = sorted(rows, reverse=True)

print("Top candidate frames:")
for score, frame, gt, base, imp in rows[:30]:
    print(
        frame,
        "score=", score,
        "| GT:", dict(gt),
        "| Baseline:", dict(base),
        "| Improved:", dict(imp),
    )