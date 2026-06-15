"""HA-LoRA evaluation script: mIoU, per-class IoU, visualizations, confusion matrix."""

import argparse
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import LoveDADataset
from model import get_model

# Distinct colors for 7 LoveDA classes + ignore
PALETTE = np.array(
    [
        [0, 0, 0],         # 0 Background
        [128, 0, 0],       # 1 Building
        [128, 128, 0],     # 2 Road
        [0, 0, 128],       # 3 Water
        [128, 128, 128],   # 4 Barren
        [0, 128, 0],       # 5 Forest
        [0, 128, 128],     # 6 Agriculture
        [255, 255, 255],   # 7/255 Ignore
    ],
    dtype=np.uint8,
)


@torch.no_grad()
def evaluate(cfg, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.get("fp16", True) and device.type == "cuda"

    # ---- Model ----
    model = get_model(OmegaConf.to_container(cfg, resolve=True))
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg.output_dir, "best_model.pth")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Remap legacy dw_conv/pw_conv keys to spatial_conv
    for k in list(ckpt["model"].keys()):
        new_k = k
        if ".dw_conv." in k:
            new_k = k.replace(".dw_conv.", ".spatial_conv.0.")
        elif ".pw_conv." in k:
            new_k = k.replace(".pw_conv.", ".spatial_conv.1.")
        if new_k != k:
            ckpt["model"][new_k] = ckpt["model"].pop(k)

    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    if "best_miou" in ckpt:
        print(f"Training best mIoU: {ckpt['best_miou']:.4f}")

    # ---- Data ----
    val_dataset = LoveDADataset(
        root=cfg.data_root, split="Val", transform=False, image_size=cfg.get("image_size", 512)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # ---- Eval ----
    num_classes = cfg.get("num_classes", 7)
    confusion = torch.zeros(num_classes, num_classes)
    vis_dir = cfg.get("vis_dir", "vis")
    os.makedirs(vis_dir, exist_ok=True)
    vis_count = 0

    for idx, (images, masks) in enumerate(val_loader):
        images = images.to(device)
        with autocast(enabled=use_amp):
            logits = model(images)
        preds = logits.argmax(dim=1).cpu()

        valid = masks != 255
        preds_flat = preds[valid].numpy()
        targets_flat = masks[valid].numpy()
        for t, p in zip(targets_flat, preds_flat):
            confusion[t, p] += 1

        # Save visualizations for first few samples
        if vis_count < 16:
            for i in range(images.shape[0]):
                if vis_count >= 16:
                    break
                pred_np = preds[i].numpy().clip(0, 7)
                mask_np = masks[i].numpy().clip(0, 7)
                # Replace 255 (ignore) with 7 for color mapping
                mask_vis = mask_np.copy()
                mask_vis[mask_vis == 255] = 7

                pred_color = PALETTE[pred_np]
                mask_color = PALETTE[mask_vis]

                cv2.imwrite(
                    os.path.join(vis_dir, f"pred_{vis_count:03d}.png"),
                    cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    os.path.join(vis_dir, f"gt_{vis_count:03d}.png"),
                    cv2.cvtColor(mask_color, cv2.COLOR_RGB2BGR),
                )
                vis_count += 1

    # ---- Metrics ----
    iou_per_class = []
    for cls in range(num_classes):
        tp = confusion[cls, cls]
        union = confusion[cls].sum() + confusion[:, cls].sum() - tp
        iou_per_class.append((tp / union).item() if union > 0 else 0.0)

    miou = float(np.mean(iou_per_class))

    # Overall accuracy
    total = confusion.sum()
    correct = confusion.diag().sum()
    oa = (correct / total).item() if total > 0 else 0.0

    # F1 per class
    f1_per_class = []
    for cls in range(num_classes):
        tp = confusion[cls, cls]
        precision = tp / confusion[:, cls].sum() if confusion[:, cls].sum() > 0 else 0
        recall = tp / confusion[cls, :].sum() if confusion[cls, :].sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_per_class.append(f1.item())
    mf1 = float(np.mean(f1_per_class))

    # ---- Print ----
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"{'Class':<15} {'IoU':>8} {'F1':>8}")
    print("-" * 35)
    for i in range(num_classes):
        name = LoveDADataset.CLASS_NAMES[i] if i < len(LoveDADataset.CLASS_NAMES) else f"Class{i}"
        print(f"{name:<15} {iou_per_class[i]:>8.4f} {f1_per_class[i]:>8.4f}")
    print("-" * 35)
    print(f"{'mIoU':<15} {miou:>8.4f}")
    print(f"{'mF1':<15} {mf1:>8.4f}")
    print(f"{'OA':<15} {oa:>8.4f}")
    print("=" * 60)

    # ---- Confusion matrix ----
    confusion_np = confusion.numpy()
    confusion_norm = confusion_np / (confusion_np.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(confusion_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(LoveDADataset.CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(LoveDADataset.CLASS_NAMES, fontsize=9)
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Prediction")
    ax.set_title(f"Confusion Matrix (mIoU={miou:.4f})")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f"{confusion_norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if confusion_norm[i, j] > 0.5 else "black")
    plt.colorbar(im, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ---- Save JSON ----
    results = {
        "miou": miou,
        "mf1": mf1,
        "oa": oa,
        "iou_per_class": {
            LoveDADataset.CLASS_NAMES[i]: iou_per_class[i] for i in range(num_classes)
        },
        "f1_per_class": {
            LoveDADataset.CLASS_NAMES[i]: f1_per_class[i] for i in range(num_classes)
        },
        "confusion_matrix": confusion_np.tolist(),
    }
    with open(os.path.join(vis_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {vis_dir}/eval_results.json")

    return miou


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HA-LoRA Evaluation")
    parser.add_argument("--config", type=str, default="src/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    evaluate(cfg, args.checkpoint)
