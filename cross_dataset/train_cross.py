"""Cross-dataset training script for HA-LoRA.
Supports LoveDA and Potsdam datasets via config.dataset_type.
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.amp import GradScaler
import torch.amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# M2F support
from m2f_loss import HungarianMatcher, M2FSetCriterion, compute_m2f_loss

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from dataset import LoveDADataset
from dataset_potsdam import PotsdamDataset
from model import count_trainable_parameters, get_model


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def get_dataset(cfg, split: str):
    dataset_type = cfg.get("dataset_type", "loveda")
    transform = (split == "train")
    split_name = "Train" if split == "train" else "Val"
    image_size = cfg.get("image_size", 512)
    input_norm = cfg.get("input_norm", "imagenet")

    if dataset_type == "potsdam":
        return PotsdamDataset(
            samples_per_tile=cfg.get("samples_per_tile", 1),
            root=cfg.data_root, split=split, transform=transform,
            image_size=image_size, input_norm=input_norm
        )
    else:
        return LoveDADataset(
            root=cfg.data_root, split=split_name, transform=transform,
            image_size=image_size, input_norm=input_norm
        )


def get_dataset_class(cfg):
    dataset_type = cfg.get("dataset_type", "loveda")
    return PotsdamDataset if dataset_type == "potsdam" else LoveDADataset


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class ModelEMA:
    """Exponential moving average of model parameters.

    Maintains a shadow copy whose params follow:
        ema_p = decay * ema_p + (1 - decay) * model_p
    Only trainable parameters are tracked; buffers (e.g., BN running stats)
    are also copied for consistency at eval time.

    Validate / save using ema.module instead of model — typically +0.3~0.7%
    mIoU for segmentation, free except for one extra FP32 model in memory.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9995, device=None):
        import copy
        self.module = copy.deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.device = device

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


class DiceLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float()
        if self.ignore_index >= 0:
            valid = (targets != self.ignore_index).unsqueeze(1).float()
        else:
            valid = torch.ones_like(targets, dtype=torch.float32).unsqueeze(1)
        probs = F.softmax(logits, dim=1)
        intersection = (probs * targets_one_hot * valid).sum(dim=(2, 3))
        union = (probs * valid).sum(dim=(2, 3)) + (targets_one_hot * valid).sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()




class OHEMCELoss(nn.Module):
    """Online Hard Example Mining for Cross-Entropy loss."""
    def __init__(self, ignore_index=255, ohem_ratio=0.25):
        super().__init__()
        self.ignore_index = ignore_index
        self.ohem_ratio = ohem_ratio

    def forward(self, logits, targets):
        per_pixel = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction="none")
        loss_flat = per_pixel.view(-1)
        nonzero = loss_flat[loss_flat > 0]
        if len(nonzero) == 0:
            return per_pixel.mean()
        num_hard = max(1, int(len(nonzero) * self.ohem_ratio))
        topk_val, _ = torch.topk(nonzero, num_hard)
        return topk_val.mean()


# ---------------------------------------------------------------------------
# Boundary Loss utilities
# ---------------------------------------------------------------------------


def extract_boundary_mask_gpu(masks_tensor, num_classes, boundary_width=3):
    """GPU-accelerated boundary extraction using vectorized max/min pooling."""
    # One-hot: (B, num_classes, H, W) -> dilation/erosion on all channels at once
    one_hot = F.one_hot(masks_tensor.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float()
    k = boundary_width * 2 + 1
    pad = boundary_width
    dilated = F.max_pool2d(one_hot, kernel_size=k, stride=1, padding=pad)
    eroded = -F.max_pool2d(-one_hot, kernel_size=k, stride=1, padding=pad)
    boundary_gt = (dilated - eroded).max(dim=1, keepdim=True)[0].clamp(0, 1)
    return boundary_gt


class BoundaryLoss(nn.Module):
    """Boundary-aware loss: BCE on boundary pixels + Dice on boundary."""

    def __init__(self, boundary_width=3):
        super().__init__()
        self.boundary_width = boundary_width

    def forward(self, boundary_logits, boundary_gt):
        """
        Args:
            boundary_logits: (B, 1, H, W) raw logits for boundary prediction
            boundary_gt: (B, 1, H, W) float32 boundary ground truth
        """
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(boundary_logits, boundary_gt, reduction="none")
        # Dice loss
        pred_prob = torch.sigmoid(boundary_logits)
        intersection = (pred_prob * boundary_gt).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + boundary_gt.sum(dim=(2, 3))
        dice = (2 * intersection + 1.0) / (union + 1.0)
        dice_loss = 1.0 - dice.mean()

        return bce.mean() + dice_loss

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_confusion(preds: np.ndarray, targets: np.ndarray, num_classes: int, ignore_idx: int = -1) -> np.ndarray:
    if ignore_idx >= 0:
        valid = (targets >= 0) & (targets < num_classes) & (targets != ignore_idx)
    else:
        valid = (targets >= 0) & (targets < num_classes)
    confusion = np.bincount(
        num_classes * targets[valid].astype(int) + preds[valid].astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return confusion


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _m2f_predict(model, images):
    """Get semantic logits from M2F model via predict()."""
    B, _, H, W = images.shape
    features_raw = model.backbone.get_intermediate_layers(
        images, n=model.layers_to_extract_0idx, reshape=False, norm=True
    )
    h_p, w_p = H // model.patch_size, W // model.patch_size
    feature_maps = []
    for feat in features_raw:
        feat = feat.reshape(B, h_p, w_p, -1).permute(0, 3, 1, 2).contiguous()
        feature_maps.append(feat)
    return model.decoder.predict(feature_maps, img_size=(H, W))


def train(cfg):
    dataset_cls = get_dataset_class(cfg)
    num_classes = cfg.get("num_classes", 7)
    decoder_type = cfg.get("decoder", {}).get("type", "mlp")

    # ---- Data ----
    train_dataset = get_dataset(cfg, "train")
    val_dataset = get_dataset(cfg, "val")

    nw = cfg.get("num_workers", 4)
    bs = cfg.get("batch_size", 4)
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=nw,
        pin_memory=True, drop_last=True, persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True,
    )

    # ---- Model ----
    model = get_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    # ---- Optimizer ----
    lora_cfg = cfg.get("lora", {})
    lora_plus_lambda = lora_cfg.get("lora_plus_lambda", None)

    if lora_plus_lambda and lora_plus_lambda > 1:
        # LoRA+: separate A and B with different learning rates (Hayou 2024)
        lora_A_params = []
        lora_B_params = []
        alpha_params = []
        lora_other_params = []  # spatial_conv etc.
        decoder_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_sem_A" in name or "lora_str_A" in name:
                lora_A_params.append(param)
            elif "lora_sem_B" in name or "lora_str_B" in name:
                lora_B_params.append(param)
            elif "alpha_sem" in name or "alpha_str" in name:
                alpha_params.append(param)
            elif "spatial_conv" in name:
                lora_other_params.append(param)
            else:
                decoder_params.append(param)
        lr = cfg.get("lr", 1.5e-4)
        wd = cfg.get("weight_decay", 0.05)
        optimizer = torch.optim.AdamW([
            {"params": lora_A_params, "lr": lr, "weight_decay": 0.0},
            {"params": lora_B_params, "lr": lr * lora_plus_lambda, "weight_decay": 0.0},
            {"params": alpha_params, "lr": lr, "weight_decay": 0.0},
            {"params": lora_other_params, "lr": lr, "weight_decay": 0.0},
            {"params": decoder_params, "lr": lr, "weight_decay": wd},
        ], lr=lr)
        print(f"LoRA+ enabled: λ={lora_plus_lambda}, A_lr={lr}, B_lr={lr * lora_plus_lambda}")
    else:
        lora_params = []
        decoder_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ("lora", "ssd", "alpha_sem", "alpha_str", "spatial_conv")):
                lora_params.append(param)
            else:
                decoder_params.append(param)

        lr = cfg.get("lr", 1.5e-4)
        wd = cfg.get("weight_decay", 0.05)
        optimizer = torch.optim.AdamW([
            {"params": lora_params, "weight_decay": 0.0},
            {"params": decoder_params, "weight_decay": wd},
        ], lr=lr)

    # ---- Loss ----
    cw = cfg.get("class_weights", None)
    if cw is not None:
        cw = torch.tensor(cw, dtype=torch.float32).to(device)
        print(f"Class weights: {cw.tolist()}")
    # Dataset-specific ignore_index: LoveDA masks are already mapped 7->255, so always use 255
    ignore_index = 255  # LoveDA maps 7->255 in __getitem__, Potsdam has no ignore
    ohem_ratio = cfg.get("ohem_ratio", None)
    if ohem_ratio is not None and ohem_ratio > 0:
        if cw is not None:
            print("WARNING: OHEM and class_weights both set; class_weights ignored.")
        ce_loss_fn = OHEMCELoss(ignore_index=ignore_index, ohem_ratio=ohem_ratio)
        print(f"OHEM enabled: ratio={ohem_ratio}")
    else:
        ce_loss_fn = nn.CrossEntropyLoss(weight=cw, ignore_index=ignore_index)
    dice_loss_fn = DiceLoss(ignore_index=ignore_index)
    dice_weight = cfg.get("dice_weight", 1.0)

    # ---- M2F Loss (for mask2former decoder) ----
    is_m2f = (decoder_type == "mask2former")
    m2f_criterion = None
    if is_m2f:
        m2f_cfg = cfg.get("decoder", {}).get("mask2former", {})
        matcher = HungarianMatcher(
            cost_class=m2f_cfg.get("cost_class", 2.0),
            cost_mask=m2f_cfg.get("cost_mask", 5.0),
            cost_dice=m2f_cfg.get("cost_dice", 5.0),
            num_points=m2f_cfg.get("num_points", 12544),
        )
        m2f_criterion = M2FSetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_ce=m2f_cfg.get("weight_ce", 2.0),
            weight_mask=m2f_cfg.get("weight_mask", 5.0),
            weight_dice=m2f_cfg.get("weight_dice", 5.0),
            eos_coef=m2f_cfg.get("eos_coef", 0.1),
            num_points=m2f_cfg.get("num_points", 12544),
        ).to(device)
        print("M2F set prediction loss enabled")

    # ---- Boundary Loss ----
    boundary_weight = cfg.get("boundary_weight", 0.0)
    boundary_loss_fn = None
    boundary_head = None
    if boundary_weight > 0:
        boundary_loss_fn = BoundaryLoss(boundary_width=cfg.get("boundary_width", 3))
        boundary_head = nn.Sequential(
            nn.Conv2d(num_classes, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        ).to(device)
        print(f"Boundary loss enabled: weight={boundary_weight}, width={cfg.get('boundary_width', 3)}")

    # ---- Scheduler ----
    epochs = cfg.get("epochs", 50)
    grad_accum_steps = cfg.get("grad_accum_steps", 1)
    grad_clip_norm = cfg.get("grad_clip_norm", 1.0)
    warmup = cfg.get("warmup_iters", 1000)
    steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = steps_per_epoch * epochs

    def _lr_lambda(step: int) -> float:
        if step >= total_steps:
            return 0.0
        warmup_factor = min(1.0, step / max(1, warmup))
        decay_factor = max(0.0, (1.0 - step / total_steps)) ** 0.9
        return warmup_factor * decay_factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # ---- AMP ----
    use_amp = cfg.get("fp16", True)
    scaler = GradScaler("cuda", enabled=use_amp)

    # ---- EMA ----
    ema_cfg = cfg.get("ema", {})
    ema_enabled = ema_cfg.get("enabled", False)
    ema_decay = ema_cfg.get("decay", 0.9995)
    ema_start_step = ema_cfg.get("start_step", 0)
    ema = ModelEMA(model, decay=ema_decay, device=device) if ema_enabled else None
    if ema_enabled:
        print(f"EMA enabled: decay={ema_decay}, start_step={ema_start_step}")

    # ---- TensorBoard ----
    log_dir = cfg.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    output_dir = cfg.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting training: {epochs} epochs, {len(train_loader)} batches/epoch, grad_accum={grad_accum_steps}")
    print(f"Effective steps/epoch: {steps_per_epoch}, total_steps: {total_steps}")
    print(f"AMP: {use_amp}, Warmup: {warmup} iters, Decoder: {decoder_type}")
    print(f"Batch size: {bs}, Effective batch: {bs * grad_accum_steps}")

    best_miou = 0.0
    global_step = 0
    start_epoch = 0
    optimizer.zero_grad(set_to_none=True)

    # ---- Resume from checkpoint ----
    resume_path = cfg.get("resume", None)
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou = ckpt.get("best_miou", 0.0)
        if boundary_head is not None and "boundary_head" in ckpt:
            boundary_head.load_state_dict(ckpt["boundary_head"])
        if ema is not None and "ema" in ckpt:
            ema.module.load_state_dict(ckpt["ema"])
            print("Resumed EMA shadow weights from checkpoint")
        # Advance scheduler to correct step
        for _ in range(start_epoch * steps_per_epoch):
            scheduler.step()
        print(f"Resumed from {resume_path}, epoch {start_epoch}, best_miou {best_miou:.4f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_dice = 0.0
        t0 = time.time()

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(images)

                if is_m2f and m2f_criterion is not None:
                    # Mask2Former: use set prediction loss
                    loss, m2f_loss_dict = compute_m2f_loss(m2f_criterion, output, masks, num_classes)
                    l_ce = m2f_loss_dict.get("loss_ce", torch.tensor(0.0))
                    l_dice = m2f_loss_dict.get("loss_dice", torch.tensor(0.0))
                else:
                    # Standard decoders (PFU, MLP, UPerNet, SC-CMRD-LAR)
                    logits = output
                    aux_outputs = None
                    if isinstance(logits, dict):
                        aux_outputs = logits
                        logits = logits["logits"]
                    l_ce = ce_loss_fn(logits, masks)
                    l_dice = dice_loss_fn(logits, masks)
                    loss = l_ce + dice_weight * l_dice

                    # Native-scale aux supervision (SC-CMRD-LAR)
                    if aux_outputs is not None and "aux_logits8" in aux_outputs:
                        aux_weight = cfg.get("decoder", {}).get(
                            "sc_cmrd_lar", {}
                        ).get("aux_weight", 0.4)
                        for key in ("aux_logits8", "aux_logits4", "aux_logits16"):
                            aux_logits = aux_outputs[key]
                            aux_logits_up = torch.nn.functional.interpolate(
                                aux_logits, size=masks.shape[-2:],
                                mode="bilinear", align_corners=False,
                            )
                            l_aux_ce = ce_loss_fn(aux_logits_up, masks)
                            l_aux_dice = dice_loss_fn(aux_logits_up, masks)
                            loss = loss + aux_weight * (l_aux_ce + dice_weight * l_aux_dice)

                    # Affinity preservation loss (SC-CMRD-LAR)
                    if (
                        aux_outputs is not None
                        and "affinity_model" in aux_outputs
                        and "affinity_anchor" in aux_outputs
                    ):
                        aff_cfg = cfg.get("decoder", {}).get(
                            "sc_cmrd_lar", {}
                        ).get("affinity_loss", {})
                        aff_enabled = aff_cfg.get("enabled", False)
                        aff_warmup_start = aff_cfg.get("warmup_start_epoch", 5)
                        aff_warmup_end = aff_cfg.get("warmup_end_epoch", 15)
                        aff_target_weight = aff_cfg.get("weight", 0.01)
                        if aff_enabled and epoch >= aff_warmup_start:
                            if epoch < aff_warmup_end:
                                progress = (epoch - aff_warmup_start) / max(1, aff_warmup_end - aff_warmup_start)
                                aff_weight = aff_target_weight * progress
                            else:
                                aff_weight = aff_target_weight
                            if aff_weight > 0:
                                if not hasattr(model, "_affinity_loss_fn"):
                                    from model import AffinityPreservationLoss
                                    model._affinity_loss_fn = AffinityPreservationLoss(
                                        n_anchor_tokens=aff_cfg.get("n_anchor_tokens", 196),
                                        tau=aff_cfg.get("tau", 0.2),
                                    )
                                l_aff = model._affinity_loss_fn(
                                    aux_outputs["affinity_model"],
                                    aux_outputs["affinity_anchor"],
                                )
                                loss = loss + aff_weight * l_aff

                    # Boundary loss (gradients flow back to main model)
                    if boundary_loss_fn is not None:
                        boundary_logits = boundary_head(logits)
                        boundary_gt = extract_boundary_mask_gpu(masks, num_classes,
                            boundary_width=cfg.get("boundary_width", 3))
                        l_boundary = boundary_loss_fn(boundary_logits, boundary_gt)
                        loss = loss + boundary_weight * l_boundary

                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                if batch_idx % 50 == 0:
                    print(f"  [grad_norm: {total_norm:.4f}]")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if ema is not None and global_step >= ema_start_step:
                    ema.update(model)

            epoch_loss += loss.item() * grad_accum_steps
            epoch_ce += l_ce.item()
            epoch_dice += l_dice.item()

            if batch_idx % 50 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                bnd_str = f" Bnd: {l_boundary.item():.4f}" if boundary_loss_fn is not None else ""
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item() * grad_accum_steps:.4f} (CE: {l_ce.item():.4f} Dice: {l_dice.item():.4f}{bnd_str}) LR: {lr_now:.6f}")

        dt = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)

        # ---- Validation ----
        # Validate using EMA model when available (typically +0.3~0.7% mIoU)
        eval_model = ema.module if ema is not None and global_step >= ema_start_step else model
        eval_model.eval()
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    output = eval_model(images)
                    if is_m2f:
                        logits = _m2f_predict(eval_model, images)
                    else:
                        logits = output
                        if isinstance(logits, dict):
                            logits = logits["logits"]
                preds = logits.argmax(dim=1).cpu().numpy()
                targets = masks.numpy()
                confusion += compute_confusion(preds, targets, num_classes)

        iou_per_class = []
        for c in range(num_classes):
            tp = confusion[c, c]
            fp = confusion[:, c].sum() - tp
            fn = confusion[c, :].sum() - tp
            if tp + fp + fn > 0:
                iou_per_class.append(tp / (tp + fp + fn))
            else:
                iou_per_class.append(0.0)
        miou = np.mean(iou_per_class)

        print(f"\nEpoch [{epoch+1}/{epochs}] ({dt:.0f}s) Loss: {avg_loss:.4f} | Val mIoU: {miou:.4f}")
        for i, iou in enumerate(iou_per_class):
            name = dataset_cls.CLASS_NAMES[i] if i < len(dataset_cls.CLASS_NAMES) else f"Class{i}"
            print(f"  {name}: {iou:.4f}")

        writer.add_scalar("val/mIoU", miou, epoch)
        writer.add_scalar("train/loss", avg_loss, epoch)

        # Save best
        if miou > best_miou:
            best_miou = miou
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_miou": best_miou,
            }
            if ema is not None:
                ckpt["ema"] = ema.module.state_dict()
            if boundary_head is not None:
                ckpt["boundary_head"] = boundary_head.state_dict()
            torch.save(ckpt, os.path.join(output_dir, "best_model.pth"))
            print(f"  ** New best mIoU: {best_miou:.4f}")

        # Save latest
        ckpt_latest = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_miou": best_miou,
        }
        if ema is not None:
            ckpt_latest["ema"] = ema.module.state_dict()
        if boundary_head is not None:
            ckpt_latest["boundary_head"] = boundary_head.state_dict()
        torch.save(ckpt_latest, os.path.join(output_dir, "latest_model.pth"))

    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)
