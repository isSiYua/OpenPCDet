from pathlib import Path
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from PIL import Image


CLASSES = {"Car", "Pedestrian", "Cyclist"}


def read_calib(calib_path: Path):
    data = {}
    for line in calib_path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        vals = np.array([float(x) for x in value.strip().split()])
        data[key] = vals

    Tr = data["Tr_velo_to_cam"].reshape(3, 4)
    Tr = np.vstack([Tr, [0, 0, 0, 1]])

    R0 = data.get("R0_rect", np.eye(3).reshape(-1)).reshape(3, 3)
    R0_4 = np.eye(4)
    R0_4[:3, :3] = R0

    return Tr, R0_4


def load_points_in_camera(lidar_path: Path, calib_path: Path):
    points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)

    Tr, R0 = read_calib(calib_path)
    cam = (R0 @ Tr @ xyz_h.T).T[:, :3]

    # KITTI camera coordinate:
    # x = left/right, y = vertical, z = forward
    mask = (
        (cam[:, 2] > 0) &
        (cam[:, 2] < 70) &
        (cam[:, 0] > -40) &
        (cam[:, 0] < 40)
    )
    return cam[mask]


def read_kitti_boxes(path: Path, is_pred=False, score_th=0.5, max_per_class=40):
    boxes = []

    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return boxes

    text = path.read_text().strip()
    if not text:
        return boxes

    for line in text.splitlines():
        p = line.split()
        if len(p) < 15:
            continue

        cls = p[0]
        if cls not in CLASSES:
            continue

        # KITTI label format:
        # type trunc occluded alpha bbox_left bbox_top bbox_right bbox_bottom h w l x y z ry [score]
        bbox2d = list(map(float, p[4:8]))
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
            "bbox2d": bbox2d,
            "h": h,
            "w": w,
            "l": l,
            "x": x,
            "y": y,
            "z": z,
            "ry": ry,
            "score": score,
        })

    # Avoid overly cluttered figures
    if is_pred and max_per_class is not None:
        filtered = []
        for cls in CLASSES:
            cls_boxes = [b for b in boxes if b["cls"] == cls]
            cls_boxes = sorted(cls_boxes, key=lambda b: b["score"], reverse=True)
            filtered.extend(cls_boxes[:max_per_class])
        boxes = filtered

    return boxes


def bev_corners_kitti_camera(box):
    """
    BEV in KITTI camera coordinates, using x-z plane.
    x: lateral direction
    z: forward direction
    ry: rotation around camera y-axis
    """
    x, z = box["x"], box["z"]
    w, l = box["w"], box["l"]
    ry = box["ry"]

    # local corners in x-z plane
    # Use width along x, length along z
    corners = np.array([
        [ w / 2,  l / 2],
        [ w / 2, -l / 2],
        [-w / 2, -l / 2],
        [-w / 2,  l / 2],
    ])

    c, s = math.cos(ry), math.sin(ry)
    R = np.array([
        [c, -s],
        [s,  c],
    ])

    pts = corners @ R.T
    pts[:, 0] += x
    pts[:, 1] += z
    return pts


def draw_camera(ax, image_path: Path, gt_boxes, title: str, show_camera_labels=False):
    img = Image.open(image_path)
    ax.imshow(img)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    for b in gt_boxes:
        x1, y1, x2, y2 = b["bbox2d"]
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1.6,
            edgecolor="lime",
            facecolor="none"
        )
        ax.add_patch(rect)

        if show_camera_labels:
            label = {
                "Car": "Car",
                "Pedestrian": "Ped",
                "Cyclist": "Cyc",
            }.get(b["cls"], b["cls"])
            ax.text(
                x1,
                max(0, y1 - 3),
                label,
                color="lime",
                fontsize=7,
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1)
            )


def draw_bev(
    ax,
    points_cam,
    gt_boxes,
    pred_boxes,
    title: str,
    show_bev_labels=False,
    xlim=(-30, 30),
    ylim=(0, 60),
):
    # LiDAR point cloud background
    ax.scatter(
        points_cam[:, 0],
        points_cam[:, 2],
        s=0.12,
        c="lightgray",
        alpha=0.45,
        linewidths=0
    )

    # Ground-truth boxes: green solid
    for b in gt_boxes:
        pts = bev_corners_kitti_camera(b)
        pts = np.vstack([pts, pts[0]])
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color="limegreen",
            linewidth=1.8,
            linestyle="-"
        )

        if show_bev_labels:
            ax.text(
                b["x"],
                b["z"],
                b["cls"][0],
                color="green",
                fontsize=7,
                fontweight="bold"
            )

    # Prediction boxes: red dashed
    for b in pred_boxes:
        pts = bev_corners_kitti_camera(b)
        pts = np.vstack([pts, pts[0]])
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color="red",
            linewidth=1.4,
            linestyle="--"
        )

        if show_bev_labels:
            ax.text(
                b["x"],
                b["z"],
                b["cls"][0],
                color="red",
                fontsize=7,
                fontweight="bold"
            )

    ax.set_title(title, fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.25, alpha=0.45)
    ax.set_xlabel("x lateral", fontsize=9)
    ax.set_ylabel("z forward", fontsize=9)


def plot_frame(frame: str, args):
    image_path = Path(args.image_dir) / f"{frame}.png"
    lidar_path = Path(args.lidar_dir) / f"{frame}.bin"
    calib_path = Path(args.calib_dir) / f"{frame}.txt"
    gt_path = Path(args.gt_dir) / f"{frame}.txt"

    base_path = Path(args.base_dir) / f"{frame}.txt"
    v7_path = Path(args.v7_dir) / f"{frame}.txt"
    best_path = Path(args.best_dir) / f"{frame}.txt"

    gt_boxes = read_kitti_boxes(gt_path, is_pred=False)
    base_boxes = read_kitti_boxes(
        base_path,
        is_pred=True,
        score_th=args.score_th,
        max_per_class=args.max_per_class
    )
    v7_boxes = read_kitti_boxes(
        v7_path,
        is_pred=True,
        score_th=args.score_th,
        max_per_class=args.max_per_class
    )
    best_boxes = read_kitti_boxes(
        best_path,
        is_pred=True,
        score_th=args.score_th,
        max_per_class=args.max_per_class
    )

    points_cam = load_points_in_camera(lidar_path, calib_path)

    print(f"\nFrame {frame}")
    print(f"GT boxes: {len(gt_boxes)}")
    print(f"Baseline boxes: {len(base_boxes)}")
    print(f"V7 boxes: {len(v7_boxes)}")
    print(f"Best observed boxes: {len(best_boxes)}")

    # Top: camera image spans all columns
    # Bottom: three BEV panels
    fig = plt.figure(figsize=(15.5, 8.2))
    gs = GridSpec(
        2,
        3,
        height_ratios=[1.05, 2.2],
        width_ratios=[1, 1, 1],
        hspace=0.22,
        wspace=0.16
    )

    ax_cam = fig.add_subplot(gs[0, :])
    ax_base = fig.add_subplot(gs[1, 0])
    ax_v7 = fig.add_subplot(gs[1, 1])
    ax_best = fig.add_subplot(gs[1, 2])

    draw_camera(
        ax_cam,
        image_path,
        gt_boxes,
        title=f"(a) Camera image with ground-truth 2D boxes | Frame {frame}",
        show_camera_labels=args.show_camera_labels
    )

    xlim = (args.x_min, args.x_max)
    ylim = (args.z_min, args.z_max)

    draw_bev(
        ax_base,
        points_cam,
        gt_boxes,
        base_boxes,
        "(b) Baseline",
        show_bev_labels=args.show_bev_labels,
        xlim=xlim,
        ylim=ylim
    )
    draw_bev(
        ax_v7,
        points_cam,
        gt_boxes,
        v7_boxes,
        "(c) V7",
        show_bev_labels=args.show_bev_labels,
        xlim=xlim,
        ylim=ylim
    )
    draw_bev(
        ax_best,
        points_cam,
        gt_boxes,
        best_boxes,
        "(d) Best observed",
        show_bev_labels=args.show_bev_labels,
        xlim=xlim,
        ylim=ylim
    )

    # Create a clean figure-level legend
    legend_handles = [
        patches.Patch(facecolor="none", edgecolor="limegreen", linewidth=1.8, label="Ground truth"),
        patches.Patch(facecolor="none", edgecolor="red", linewidth=1.4, linestyle="--", label="Prediction"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.985)
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"paper_style_v2_{frame}_th{args.score_th}.png"
    pdf_path = out_dir / f"paper_style_v2_{frame}_th{args.score_th}.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--lidar_dir", required=True)
    parser.add_argument("--calib_dir", required=True)
    parser.add_argument("--gt_dir", required=True)

    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--v7_dir", required=True)
    parser.add_argument("--best_dir", required=True)

    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--score_th", type=float, default=0.5)
    parser.add_argument("--max_per_class", type=int, default=40)

    parser.add_argument("--out_dir", default="paper_style_figures_v2")

    # Display controls
    parser.add_argument("--show_camera_labels", action="store_true")
    parser.add_argument("--show_bev_labels", action="store_true")

    # BEV display range
    parser.add_argument("--x_min", type=float, default=-30)
    parser.add_argument("--x_max", type=float, default=30)
    parser.add_argument("--z_min", type=float, default=0)
    parser.add_argument("--z_max", type=float, default=60)

    args = parser.parse_args()

    for frame in args.frames:
        plot_frame(frame, args)


if __name__ == "__main__":
    main()