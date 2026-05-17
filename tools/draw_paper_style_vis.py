from pathlib import Path
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


CLASSES = {"Car", "Pedestrian", "Cyclist"}


def read_calib(calib_path):
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


def load_points_in_camera(lidar_path, calib_path):
    pts = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)
    xyz = pts[:, :3]
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)

    Tr, R0 = read_calib(calib_path)
    cam = (R0 @ Tr @ xyz_h.T).T[:, :3]

    # camera coordinate: x lateral, y vertical, z forward
    mask = (cam[:, 2] > 0) & (cam[:, 2] < 70) & (cam[:, 0] > -40) & (cam[:, 0] < 40)
    return cam[mask]


def read_kitti_boxes(path, is_pred=False, score_th=0.5, max_per_class=40):
    boxes = []
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return boxes

    for line in path.read_text().strip().splitlines():
        if not line:
            continue
        p = line.split()
        cls = p[0]
        if cls not in CLASSES or len(p) < 15:
            continue

        bbox2d = list(map(float, p[4:8]))
        h, w, l = map(float, p[8:11])
        x, y, z = map(float, p[11:14])
        ry = float(p[14])

        score = 1.0
        if is_pred and len(p) > 15:
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
            "score": score
        })

    if is_pred:
        filtered = []
        for cls in CLASSES:
            cls_boxes = [b for b in boxes if b["cls"] == cls]
            cls_boxes = sorted(cls_boxes, key=lambda b: b["score"], reverse=True)
            filtered.extend(cls_boxes[:max_per_class])
        boxes = filtered

    return boxes


def bev_corners_kitti_camera(box):
    """
    KITTI camera coordinate BEV using x-z plane.
    Dimensions are h, w, l. Rotation is ry around camera y-axis.
    """
    x, z = box["x"], box["z"]
    w, l = box["w"], box["l"]
    ry = box["ry"]

    # local object corners in x-z plane
    corners = np.array([
        [ l / 2,  w / 2],
        [ l / 2, -w / 2],
        [-l / 2, -w / 2],
        [-l / 2,  w / 2],
    ])

    c, s = math.cos(ry), math.sin(ry)
    R = np.array([[c, s], [-s, c]])
    pts = corners @ R.T
    pts[:, 0] += x
    pts[:, 1] += z
    return pts


def draw_camera(ax, image_path, gt_boxes, title):
    img = Image.open(image_path)
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
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
        ax.text(x1, y1 - 3, b["cls"], color="lime", fontsize=7)


def draw_bev(ax, points_cam, gt_boxes, pred_boxes, title):
    # point cloud background
    ax.scatter(points_cam[:, 0], points_cam[:, 2], s=0.15, c="lightgray", alpha=0.45)

    # GT boxes: green solid
    for b in gt_boxes:
        pts = bev_corners_kitti_camera(b)
        pts = np.vstack([pts, pts[0]])
        ax.plot(pts[:, 0], pts[:, 1], color="limegreen", linewidth=1.6, linestyle="-")
        ax.text(b["x"], b["z"], b["cls"][0], color="green", fontsize=6)

    # prediction boxes: red dashed
    for b in pred_boxes:
        pts = bev_corners_kitti_camera(b)
        pts = np.vstack([pts, pts[0]])
        ax.plot(pts[:, 0], pts[:, 1], color="red", linewidth=1.2, linestyle="--")
        ax.text(b["x"], b["z"], b["cls"][0], color="red", fontsize=6)

    ax.set_title(title, fontsize=10)
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, 60)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.2)
    ax.set_xlabel("x lateral")
    ax.set_ylabel("z forward")


def plot_frame(frame, args):
    image_path = Path(args.image_dir) / f"{frame}.png"
    lidar_path = Path(args.lidar_dir) / f"{frame}.bin"
    calib_path = Path(args.calib_dir) / f"{frame}.txt"
    gt_path = Path(args.gt_dir) / f"{frame}.txt"

    base_path = Path(args.base_dir) / f"{frame}.txt"
    v7_path = Path(args.v7_dir) / f"{frame}.txt"
    best_path = Path(args.best_dir) / f"{frame}.txt"

    gt_boxes = read_kitti_boxes(gt_path, is_pred=False)
    base_boxes = read_kitti_boxes(base_path, is_pred=True, score_th=args.score_th)
    v7_boxes = read_kitti_boxes(v7_path, is_pred=True, score_th=args.score_th)
    best_boxes = read_kitti_boxes(best_path, is_pred=True, score_th=args.score_th)

    points_cam = load_points_in_camera(lidar_path, calib_path)

    print(f"\nFrame {frame}")
    print(f"GT: {len(gt_boxes)} | Baseline: {len(base_boxes)} | V7: {len(v7_boxes)} | Best: {len(best_boxes)}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))

    draw_camera(axes[0], image_path, gt_boxes, f"(a) Camera + GT | {frame}")
    draw_bev(axes[1], points_cam, gt_boxes, base_boxes, "(b) Baseline")
    draw_bev(axes[2], points_cam, gt_boxes, v7_boxes, "(c) V7")
    draw_bev(axes[3], points_cam, gt_boxes, best_boxes, "(d) Best observed")

    fig.suptitle("Green solid boxes: ground truth | Red dashed boxes: predictions", fontsize=12)
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"paper_style_{frame}_th{args.score_th}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


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
    parser.add_argument("--out_dir", default="paper_style_figures")
    args = parser.parse_args()

    for frame in args.frames:
        plot_frame(frame, args)


if __name__ == "__main__":
    main()