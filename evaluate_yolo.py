import os
import glob
import math
import yaml
from collections import defaultdict
from typing import List, Tuple

import cv2
import numpy as np


# ===================== 設定區 =====================

# YOLO 訓練時用的 data 設定檔
DATA_YAML = "aortic_valve_colab.yaml"

# YOLO val 命令輸出的預測 label 目錄 (save_txt 產生的)
PRED_LABEL_DIR = os.path.join("runs", "detect", "val", "labels")

# 輸出評估結果
OUTPUT_MD = "evaluation_report.md"
VIS_OUTPUT_DIR = "vis_eval"

# IoU 門檻，用來判定 TP/FP/FN
IOU_THRESHOLD = 0.5

# 圖像顯示設定
GT_COLOR = (0, 255, 0)       # 綠色：Ground Truth
PRED_COLOR = (0, 0, 255)     # 紅色：Prediction
LINE_THICKNESS = 1           # 框線粗細
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_THICKNESS = 1


# ===================== 工具函式 =====================

def load_data_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    # names 可能是 dict{id: name} 或 list
    if isinstance(names, dict):
        # 轉成 list，根據 id 排序
        max_id = max(int(k) for k in names.keys())
        names_list = [None] * (max_id + 1)
        for k, v in names.items():
            names_list[int(k)] = v
        names = names_list

    # 取得 test 影像位置
    # 常見寫法：path + test / 或直接 test 為絕對/相對路徑
    root_path = data.get("path", "")
    test_path = data.get("test") or data.get("val") or data.get("train")

    if root_path and not os.path.isabs(test_path):
        test_path = os.path.join(root_path, test_path)

        # 根據 test_path 的最後一層名稱，判斷是否已經是 images 資料夾
    last_part = os.path.basename(os.path.normpath(test_path)).lower()
    if last_part == "images":
        # yaml 已經指到 images 目錄
        test_images_dir = test_path
        test_labels_dir = os.path.join(os.path.dirname(test_path), "labels")
    else:
        # 預設: test 下有 images / labels 子資料夾
        test_images_dir = os.path.join(test_path, "images")
        test_labels_dir = os.path.join(test_path, "labels")

    return names, test_images_dir, test_labels_dir


def yolo_to_xyxy(box: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """YOLO 格式 [xc, yc, w, h] (normalize) 轉成絕對座標 [x1, y1, x2, y2]"""
    xc, yc, w, h = box
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return x1, y1, x2, y2


def compute_iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def load_label_file(path: str, img_w: int, img_h: int, is_pred: bool):
    """
    讀取 YOLO label:
    GT:  class xc yc w h
    Pred: class xc yc w h [conf]
    回傳: list of dict:
      { 'cls': int, 'bbox': [x1,y1,x2,y2], 'conf': float or None }
    """
    if not os.path.exists(path):
        return []

    objs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            conf = float(parts[5]) if is_pred and len(parts) >= 6 else None

            x1, y1, x2, y2 = yolo_to_xyxy([xc, yc, w, h], img_w, img_h)
            objs.append({
                "cls": cls_id,
                "bbox": [x1, y1, x2, y2],
                "conf": conf
            })
    return objs


def match_predictions(gt_objs, pred_objs, iou_thr: float):
    """
    單張圖片做匹配:
    - gt_objs: list of dict
    - pred_objs: list of dict
    回傳:
      matches: list of (gt_idx, pred_idx, iou)
      gt_unmatched: set of gt_idx
      pred_unmatched: set of pred_idx
      duplicate_matches: list of (gt_idx, [pred_idx...])  # 同一 GT 被多個 pred 命中
    """
    if not gt_objs and not pred_objs:
        return [], set(), set(), []

    iou_matrix = np.zeros((len(gt_objs), len(pred_objs)), dtype=np.float32)
    for i, gt in enumerate(gt_objs):
        for j, pr in enumerate(pred_objs):
            if gt["cls"] != pr["cls"]:
                iou_matrix[i, j] = 0.0
            else:
                iou_matrix[i, j] = compute_iou(gt["bbox"], pr["bbox"])

    # 貪婪匹配: 一次取 matrix 中的最大 IoU，若 >= threshold 就當作匹配
    gt_matched = set()
    pred_matched = set()
    matches = []

    while True:
        if iou_matrix.size == 0:
            break
        max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        max_iou = float(iou_matrix[max_idx])
        if max_iou < iou_thr:
            break
        gi, pj = max_idx
        if gi in gt_matched or pj in pred_matched:
            iou_matrix[gi, pj] = -1.0
            continue
        gt_matched.add(gi)
        pred_matched.add(pj)
        matches.append((gi, pj, max_iou))
        iou_matrix[gi, :] = -1.0
        iou_matrix[:, pj] = -1.0

    gt_unmatched = set(range(len(gt_objs))) - gt_matched
    pred_unmatched = set(range(len(pred_objs))) - pred_matched

    # 檢查重複匹配: 若有多個 pred 的 IoU >= threshold 指向同一 gt（即使沒被選為主匹配），視為 duplicate
    duplicate_matches = []
    for gi, gt in enumerate(gt_objs):
        pred_indices_for_gt = []
        for pj, pr in enumerate(pred_objs):
            if gt["cls"] != pr["cls"]:
                continue
            iou_ = compute_iou(gt["bbox"], pr["bbox"])
            if iou_ >= iou_thr:
                pred_indices_for_gt.append(pj)
        if len(pred_indices_for_gt) > 1:
            duplicate_matches.append((gi, pred_indices_for_gt))

    return matches, gt_unmatched, pred_unmatched, duplicate_matches


def draw_boxes(image, gt_objs, pred_objs, names, save_path: str):
    img = image.copy()

    # 畫 GT
    for obj in gt_objs:
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        cls_id = obj["cls"]
        cls_name = names[cls_id] if names and 0 <= cls_id < len(names) else str(cls_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, LINE_THICKNESS)
        cv2.putText(img, f"GT:{cls_name}",
                    (x1, max(0, y1 - 5)),
                    FONT, FONT_SCALE, GT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    # 畫 Pred
    for obj in pred_objs:
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        cls_id = obj["cls"]
        conf = obj["conf"]
        cls_name = names[cls_id] if names and 0 <= cls_id < len(names) else str(cls_id)
        label = f"PD:{cls_name}"
        if conf is not None:
            label += f" {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), PRED_COLOR, LINE_THICKNESS)
        cv2.putText(img, label,
                    (x1, min(img.shape[0] - 5, y2 + 12)),
                    FONT, FONT_SCALE, PRED_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


# ===================== 主流程 =====================

def main():
    names, test_images_dir, test_labels_dir = load_data_yaml(DATA_YAML)

    if not os.path.isdir(test_images_dir):
        raise FileNotFoundError(f"Test images dir not found: {test_images_dir}")
    if not os.path.isdir(test_labels_dir):
        raise FileNotFoundError(f"Test labels dir not found: {test_labels_dir}")
    if not os.path.isdir(PRED_LABEL_DIR):
        raise FileNotFoundError(f"Prediction labels dir not found: {PRED_LABEL_DIR}")

    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

    # 取得所有 test 影像
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        img_paths.extend(glob.glob(os.path.join(test_images_dir, ext)))
    img_paths = sorted(img_paths)

    # 統計資料
    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou_sum = 0.0
    total_iou_count = 0
    total_duplicates = 0

    per_class_stats = defaultdict(lambda: {
        "gt": 0,
        "pred": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "iou_sum": 0.0,
        "iou_count": 0,
        "duplicates": 0
    })

    print(f"Found {len(img_paths)} test images.")
    for idx, img_path in enumerate(img_paths):
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]

        print(f"[{idx + 1}/{len(img_paths)}] Processing {img_name}...")

        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: cannot read image {img_path}, skip.")
            continue
        h, w = img.shape[:2]

        gt_label_path = os.path.join(test_labels_dir, base_name + ".txt")
        pred_label_path = os.path.join(PRED_LABEL_DIR, base_name + ".txt")

        gt_objs = load_label_file(gt_label_path, w, h, is_pred=False)
        pred_objs = load_label_file(pred_label_path, w, h, is_pred=True)

        total_gt += len(gt_objs)
        total_pred += len(pred_objs)

        for o in gt_objs:
            per_class_stats[o["cls"]]["gt"] += 1
        for o in pred_objs:
            per_class_stats[o["cls"]]["pred"] += 1

        matches, gt_unmatched, pred_unmatched, duplicate_matches = match_predictions(
            gt_objs, pred_objs, IOU_THRESHOLD
        )

        total_tp += len(matches)
        total_fn += len(gt_unmatched)
        total_fp += len(pred_unmatched)
        total_duplicates += len(duplicate_matches)

        for gi, pj, iou in matches:
            cls_id = gt_objs[gi]["cls"]
            per_class_stats[cls_id]["tp"] += 1
            per_class_stats[cls_id]["iou_sum"] += iou
            per_class_stats[cls_id]["iou_count"] += 1
            total_iou_sum += iou
            total_iou_count += 1

        for gi in gt_unmatched:
            cls_id = gt_objs[gi]["cls"]
            per_class_stats[cls_id]["fn"] += 1

        for pj in pred_unmatched:
            cls_id = pred_objs[pj]["cls"]
            per_class_stats[cls_id]["fp"] += 1

        for gi, pred_idx_list in duplicate_matches:
            cls_id = gt_objs[gi]["cls"]
            per_class_stats[cls_id]["duplicates"] += (len(pred_idx_list) - 1)
            total_duplicates += (len(pred_idx_list) - 1)

        # 產生可視化影像
        vis_save_path = os.path.join(VIS_OUTPUT_DIR, base_name + "_vis.png")
        draw_boxes(img, gt_objs, pred_objs, names, vis_save_path)

    # ========= 統計與輸出 MD 報告 =========
    if total_iou_count > 0:
        mean_iou = total_iou_sum / total_iou_count
    else:
        mean_iou = 0.0

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    lines = []
    # 標題
    lines.append("# YOLO 評估摘要\n")

    # 總覽區塊（對齊你提供的範例格式）
    lines.append("## YOLO 評估總覽\n")
    lines.append(f"IoU 門檻 : {IOU_THRESHOLD}")
    lines.append(f"圖片數量 : {len(img_paths)}")
    lines.append(f"GT 數量 : {total_gt}")
    lines.append(f"預測框數量 : {total_pred}")
    lines.append(f"True Positive (TP) : {total_tp} ；TP 平均 IoU : {mean_iou:.4f}")
    lines.append(f"False Positive (FP) : {total_fp}")
    lines.append(f"False Negative (FN) : {total_fn}")
    lines.append(f"多重檢出數 : {total_duplicates}\n")

    # 補充：整體 Precision / Recall / F1
    lines.append("## 整體指標\n")
    lines.append(f"Precision : {precision:.4f}")
    lines.append(f"Recall    : {recall:.4f}")
    lines.append(f"F1-score  : {f1:.4f}\n")

    # 圖像輸出路徑摘要（與目前輸出邏輯對應）
    lines.append("## 圖像輸出總結\n")
    lines.append(f"全部影像 : `{VIS_OUTPUT_DIR}`")
    lines.append("說明 : 圖像中 綠色細框為 Ground Truth，紅色細框為模型預測（含類別與信心分數）。")

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n=== 評估完成 ===")
    print(f"Markdown 報告輸出：{OUTPUT_MD}")
    print(f"可視化影像輸出資料夾：{VIS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()