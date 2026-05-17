from pathlib import Path
import math
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


CLASSES = {"Car", "Pedestrian", "Cyclist"}


def read_kitti_label(path, is_pred=False, score_th=0.5, max_per_class=30):
    """
    Read KITTI label/prediction txt.

    KITTI format:
    type trunc occluded alpha bbox_left bbox_top bbox_right bbox_bottom h w l x y z ry [score]
    """
    boxes = []

    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return boxes

    lines = path.read_text().strip().splitlines()

    for line in lines:
        if not line:
            continue

        p = line.split()
        cls = p[0]

        if cls not in CLASSES:
            continue

        if len(p) < 15:
            continue

        h, w, l = map(float, p[8:11])
        x, y, z = map(float, p[11:14])
        ry = float(p[14])

        score = 1.0
        if is_pred:
            if len(p) > 15:
                score = float(p[15])
            if score < score_th:
                continue

        boxes.append({
            "cls": cls,
            "h": h,
            "w": w,
            "l": l,
            "x": x,
            "y": y,
            "z": z,
            "ry": ry,
            "score": score,
        })

    # 防止某些 prediction 文件里低质量框太多，把图画爆
    if is_pred and max_per_class is not None:
        grouped = defaultdict(list)
        for b in boxes:
            grouped[b["cls"]].append(b)

        filtered = []
        for cls, cls_boxes in grouped.items():
            cls_boxes = sorted(cls_boxes, key=lambda b: b["score"], reverse=True)
            filtered.extend(cls_boxes[:max_per_class])

        boxes = filtered

    return boxes


def bev_corners(box):
    """
    Draw BEV in KITTI camera coordinates:
    x = lateral direction
    z = forward direction

    We use box width w and length l.
    """
    x, z = box["x"], box["z"]
    l, w = box["l"], box["w"]
    ry = box["ry"]

    corners = np.array([
        [ w / 2,  l / 2],
        [ w / 2, -l / 2],
        [-w / 2, -l / 2],
        [-w / 2,  l / 2],
    ])

    c, s = math.cos(ry), math.sin(ry)
    R = np.array([[c, -s], [s, c]])

    rotated = corners @ R.T
    rotated[:, 0] += x
    rotated[:, 1] += z

    return rotated


def draw_boxes(ax, boxes, is_gt=False):
    for b in boxes:
        pts = bev_corners(b)
        pts = np.vstack([pts, pts[0]])

        if is_gt:
            ax.plot(pts[:, 0], pts[:, 1], linewidth=1.8, linestyle="-")
        else:
            ax.plot(pts[:, 0], pts[:, 1], linewidth=1.1, linestyle="--")

        label = b["cls"][0]
        if not is_gt:
            label += f"{b['score']:.2f}"

        ax.text(b["x"], b["z"], label, fontsize=5)


def setup_axis(ax, title):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x lateral")
    ax.set_ylabel("z forward")

    # KITTI 常用显示范围
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 70)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.2)


def plot_frame(frame_id, gt_dir, base_dir, v7_dir, aug_dir, out_dir, score_th):
    gt = read_kitti_label(gt_dir / f"{frame_id}.txt", is_pred=False)
    base = read_kitti_label(base_dir / f"{frame_id}.txt", is_pred=True, score_th=score_th)
    v7 = read_kitti_label(v7_dir / f"{frame_id}.txt", is_pred=True, score_th=score_th)
    aug = read_kitti_label(aug_dir / f"{frame_id}.txt", is_pred=True, score_th=score_th)

    print(f"\nFrame {frame_id}")
    print(f"GT boxes: {len(gt)}")
    print(f"Baseline boxes after threshold: {len(base)}")
    print(f"V7 boxes after threshold: {len(v7)}")
    print(f"V7+Mobile+Aug boxes after threshold: {len(aug)}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    setup_axis(axes[0], f"(a) Baseline | {frame_id}")
    draw_boxes(axes[0], gt, is_gt=True)
    draw_boxes(axes[0], base, is_gt=False)

    setup_axis(axes[1], "(b) V7")
    draw_boxes(axes[1], gt, is_gt=True)
    draw_boxes(axes[1], v7, is_gt=False)

    setup_axis(axes[2], "(c) V7 + Mobile + Aug")
    draw_boxes(axes[2], gt, is_gt=True)
    draw_boxes(axes[2], aug, is_gt=False)

    fig.suptitle(
        "Solid boxes: ground truth | Dashed boxes: predictions",
        fontsize=10
    )

    plt.tight_layout()

    out_path = out_dir / f"bev_compare_{frame_id}_th{score_th}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--v7_dir", type=str, required=True)
    parser.add_argument("--aug_dir", type=str, required=True)
    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--score_th", type=float, default=0.5)
    parser.add_argument("--out_dir", type=str, default="bev_figures")

    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    base_dir = Path(args.base_dir)
    v7_dir = Path(args.v7_dir)
    aug_dir = Path(args.aug_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    for frame_id in args.frames:
        plot_frame(
            frame_id=frame_id,
            gt_dir=gt_dir,
            base_dir=base_dir,
            v7_dir=v7_dir,
            aug_dir=aug_dir,
            out_dir=out_dir,
            score_th=args.score_th,
        )


if __name__ == "__main__":
    main()