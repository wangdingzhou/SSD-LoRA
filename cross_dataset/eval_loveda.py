#!/usr/bin/env python3
"""LoveDA: Sliding Window + TTA Evaluation.

Usage:
    python eval_loveda.py --checkpoint PATH --config PATH
"""

import sys, os, argparse, torch, numpy as np, cv2
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from model import get_model
from dataset import LoveDADataset
from torchvision import transforms
from omegaconf import OmegaConf

CLASS_NAMES = LoveDADataset.CLASS_NAMES
NUM_CLASSES = LoveDADataset.NUM_CLASSES
IGNORE_IDX = 7
FINAL_EVAL_STRIDE = 256
FINAL_EVAL_SCALES = "0.75,1.0,1.25,1.5"

_NORM_MAP = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "sat493m": ([0.430, 0.411, 0.296], [0.213, 0.156, 0.143]),
}


def _m2f_inference_tile(model, tile, device):
    B = tile.shape[0]
    features_raw = model.backbone.get_intermediate_layers(
        tile, n=model.layers_to_extract_0idx, reshape=False, norm=True
    )
    _, _, H, W = tile.shape
    h_p, w_p = H // model.patch_size, W // model.patch_size
    feature_maps = []
    for feat in features_raw:
        feat = feat.reshape(B, h_p, w_p, -1).permute(0, 3, 1, 2).contiguous()
        feature_maps.append(feat)
    return model.decoder.predict(feature_maps, img_size=(H, W))


def _inference_tile(model, tile, device):
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'predict'):
        return _m2f_inference_tile(model, tile, device)
    return model(tile)


def load_model(checkpoint_path, config_path, device):
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    model = get_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "ema" in ckpt:
        state = ckpt["ema"]
        print("Loaded EMA weights from checkpoint")
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    for k in list(state.keys()):
        if "pfeb" in k or "freq_plugin" in k or "sru" in k or "feat_plugin" in k:
            continue
        new_k = k
        if ".dw_conv." in k:
            new_k = k.replace(".dw_conv.", ".spatial_conv.0.")
        elif ".pw_conv." in k:
            new_k = k.replace(".pw_conv.", ".spatial_conv.1.")
        if new_k != k:
            state[new_k] = state.pop(k)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def sliding_window_inference(model, image_tensor, tile_size, stride, device):
    C, H, W = image_tensor.shape
    if H <= tile_size and W <= tile_size:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            tile = image_tensor.unsqueeze(0).to(device)
            logits = _inference_tile(model, tile, device)
        return logits.squeeze(0).cpu().float()

    logits_sum = torch.zeros(NUM_CLASSES, H, W)
    count_map = torch.zeros(1, H, W)

    pad_h = max(0, tile_size - H)
    pad_w = max(0, tile_size - W)
    if pad_h > 0 or pad_w > 0:
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode="reflect")
        _, pH, pW = image_tensor.shape
    else:
        pH, pW = H, W

    ys = list(range(0, pH - tile_size + 1, stride))
    xs = list(range(0, pW - tile_size + 1, stride))
    if not ys or ys[-1] + tile_size < pH:
        ys.append(pH - tile_size)
    if not xs or xs[-1] + tile_size < pW:
        xs.append(pW - tile_size)

    for y in ys:
        for x in xs:
            crop = image_tensor[:, y:y+tile_size, x:x+tile_size]
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                crop_logits = _inference_tile(model, crop.unsqueeze(0).to(device), device)
            crop_logits = crop_logits.float().cpu()

            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            ch, cw = y_end - y, x_end - x
            # Crop (not interpolate) the valid region; reflect-padded rows/cols
            # at the bottom/right must be discarded, not squashed into the valid area.
            logits_sum[:, y:y_end, x:x_end] += crop_logits[0, :, :ch, :cw]
            count_map[:, y:y_end, x:x_end] += 1

    return logits_sum / count_map.clamp(min=1)


def inference_with_tta(model, image_tensor, tile_size, stride, device, tta_mode):
    logits = sliding_window_inference(model, image_tensor, tile_size, stride, device)
    count = 1

    if "hflip" in tta_mode:
        flipped = torch.flip(image_tensor, [-1])
        logits_flip = sliding_window_inference(model, flipped, tile_size, stride, device)
        logits += torch.flip(logits_flip, [-1])
        count += 1

    if "vflip" in tta_mode:
        flipped = torch.flip(image_tensor, [-2])
        logits_flip = sliding_window_inference(model, flipped, tile_size, stride, device)
        logits += torch.flip(logits_flip, [-2])
        count += 1

    return logits / count


def compute_confusion(pred, target, num_classes, ignore_idx=7):
    valid = (target >= 0) & (target < num_classes) & (target != ignore_idx)
    p = pred[valid].astype(np.int32)
    t = target[valid].astype(np.int32)
    idx = t * num_classes + p
    counts = np.bincount(idx, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def inference_with_mst(model, image_tensor, tile_size, stride, device, scales, tta_mode):
    C, H, W = image_tensor.shape
    logits_sum = None

    for si, scale in enumerate(scales):
        print(f"    [scale {si+1}/{len(scales)}] {scale:.2f}x")
        if scale != 1.0:
            new_H, new_W = int(H * scale), int(W * scale)
            scaled = F.interpolate(
                image_tensor.unsqueeze(0), size=(new_H, new_W),
                mode="bilinear", align_corners=False
            ).squeeze(0)
        else:
            scaled = image_tensor

        if tta_mode:
            logits = inference_with_tta(model, scaled, tile_size, stride, device, tta_mode)
        else:
            logits = sliding_window_inference(model, scaled, tile_size, stride, device)

        if logits.shape[1:] != (H, W):
            logits = F.interpolate(
                logits.unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False
            ).squeeze(0)

        if logits_sum is None:
            logits_sum = logits
        else:
            logits_sum += logits

    return logits_sum / len(scales)


def evaluate(model, data_root, tile_size, stride, device, tta_mode, scales=None, input_norm="imagenet"):
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    val_ds = LoveDADataset(root=data_root, split="Val", transform=False)
    mean, std = _NORM_MAP.get(input_norm, _NORM_MAP["imagenet"])
    normalize = transforms.Normalize(mean=mean, std=std)
    scale_str = f", scales={[f'{s:.2f}' for s in scales]}" if scales else ""
    print(f"Eval: {len(val_ds)} tiles, tile={tile_size}, stride={stride}, tta={tta_mode}{scale_str}, norm={input_norm}")

    for i in range(len(val_ds)):
        img_path = val_ds.images[i]
        mask_path = val_ds.masks[i]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Map ignore index 7 to ignore_idx for eval
        mask = mask.astype(np.int64)
        H, W = image.shape[:2]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = normalize(image_tensor)

        if scales and len(scales) > 1:
            logits = inference_with_mst(model, image_tensor, tile_size, stride, device, scales, tta_mode)
        elif tta_mode:
            logits = inference_with_tta(model, image_tensor, tile_size, stride, device, tta_mode)
        else:
            logits = sliding_window_inference(model, image_tensor, tile_size, stride, device)

        if logits.shape[1:] != (H, W):
            logits = F.interpolate(logits.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)

        pred = logits.argmax(dim=0).numpy()
        confusion += compute_confusion(pred, mask, NUM_CLASSES, IGNORE_IDX)

        print(f"  [{i+1}/{len(val_ds)}] {os.path.basename(img_path)}")

    ious = []
    for cls in range(NUM_CLASSES):
        tp = confusion[cls, cls]
        union = confusion[cls].sum() + confusion[:, cls].sum() - tp
        ious.append(tp / union if union > 0 else 0.0)

    f1s = []
    for cls in range(NUM_CLASSES):
        tp = confusion[cls, cls]
        precision = tp / confusion[:, cls].sum() if confusion[:, cls].sum() > 0 else 0
        recall = tp / confusion[cls, :].sum() if confusion[cls, :].sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1s.append(float(f1))

    total = confusion.sum()
    oa = float(confusion.diagonal().sum() / total) if total > 0 else 0.0

    return float(np.mean(ious)), ious, float(np.mean(f1s)), f1s, oa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=FINAL_EVAL_STRIDE)
    parser.add_argument("--tta", default="hflip+vflip", choices=["hflip", "vflip", "hflip+vflip", "none"])
    parser.add_argument(
        "--scales",
        default=FINAL_EVAL_SCALES,
        help="Multi-scale testing, e.g. 0.75,1.0,1.25; use 'none' to disable",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_root", default=None)
    args = parser.parse_args()

    cfg_for_path = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if args.data_root is None:
        args.data_root = cfg_for_path.get("data_root", "./datasets/LoveDA")

    if args.tta == "none":
        args.tta = ""

    scales = None
    if args.scales and args.scales.lower() != "none":
        scales = [float(s) for s in args.scales.split(",")]
        print(f"MST scales: {scales}")

    device = torch.device("cuda")
    print(f"Tile={args.tile_size}, Stride={args.stride}, TTA={args.tta or 'none'}")

    model = load_model(args.checkpoint, args.config, device)
    print("Model loaded")

    cfg_dict = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    input_norm = cfg_dict.get("input_norm", "imagenet")

    miou, ious, mf1, f1s, oa = evaluate(
        model, args.data_root, args.tile_size, args.stride, device, args.tta, scales, input_norm
    )

    tags = ["Sliding Window"]
    if args.tta:
        tags.append(f"TTA({args.tta})")
    if scales:
        tags.append(f"MST({args.scales})")
    tag_str = " + ".join(tags)

    print(f"\n{tag_str}")
    print(f"  mIoU: {miou:.2%}")
    print(f"  mF1:  {mf1:.2%}")
    print(f"  OA:   {oa:.2%}")
    print(f"  {'Class':<15} {'IoU':>8} {'F1':>8}")
    print(f"  {'-'*35}")
    for name, iou, f1 in zip(CLASS_NAMES, ious, f1s):
        print(f"  {name:<15} {iou:>8.2%} {f1:>8.2%}")


if __name__ == "__main__":
    main()
