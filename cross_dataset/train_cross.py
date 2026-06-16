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


def build_optimizer_with_groups(model, lr_groups_cfg, weight_decay):
    """Build AdamW with 4 LR groups for SCHRR v2.

    Groups: decoder_new, hr_encoder, router_gate_lar (all 1.5e-4),
            lora_tcam (3.0e-5). Backbone base frozen (excluded by requires_grad=False).
    """
    PEFT_KEYWORDS = ("lora_", "alpha_sem", "alpha_str", "spatial_conv", "tcam")
    groups = {"decoder_new": [], "hr_encoder": [], "router_gate_lar": [], "lora_tcam": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # backbone base frozen
        if any(k in name for k in PEFT_KEYWORDS):
            groups["lora_tcam"].append(param)
        elif "hr_encoder" in name:
            groups["hr_encoder"].append(param)
        elif any(x in name for x in ["router_conv", "attn_convs", "gate", "coeff_conv", "raw_lambda"]):
            groups["router_gate_lar"].append(param)
        else:
            # decoder.layer_projs, schrr_8/4 (lar v_proj/q_proj/out_proj + fuse),
            # refine, head, aux_head_conv
            groups["decoder_new"].append(param)

    param_groups = [
        {"params": groups["decoder_new"], "lr": lr_groups_cfg["decoder_new"], "weight_decay": weight_decay},
        {"params": groups["hr_encoder"], "lr": lr_groups_cfg["hr_encoder"], "weight_decay": weight_decay},
        {"params": groups["router_gate_lar"], "lr": lr_groups_cfg["router_gate_lar"], "weight_decay": weight_decay},
        {"params": groups["lora_tcam"], "lr": lr_groups_cfg["lora_tcam"], "weight_decay": 0.0},
    ]
    for gname in ["decoder_new", "hr_encoder", "router_gate_lar", "lora_tcam"]:
        n_params = sum(p.numel() for p in groups[gname])
        print(f"  LR group {gname}: {len(groups[gname])} tensors, {n_params/1e6:.2f}M params, lr={lr_groups_cfg[gname]}")
    return torch.optim.AdamW(param_groups)


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

    # ---- Warm-start from R0 (v2: before optimizer/EMA, after model.to(device)) ----
    # Loads backbone + LoRA/TCAM from R0 80.95% checkpoint. Decoder from scratch.
    # Must be BEFORE optimizer/EMA creation so EMA shadow = warm-started model.
    warm_start_path = cfg.get("warm_start", None)
    if warm_start_path and os.path.isfile(warm_start_path):
        ckpt = torch.load(warm_start_path, map_location="cpu", weights_only=False)
        pretrained_state = ckpt["model"]
        model_state = model.state_dict()

        # Load strategy: load backbone.* (includes internal LoRA/TCAM).
        # Skip: decoder.* (old R0 decoder, structurally different).
        # Train strategy: base DINO frozen, LoRA/TCAM trainable (see freeze below).
        ALLOWED_PREFIXES = ("backbone.",)
        BLOCKED_PREFIXES = ("decoder.",)

        matched = {}
        skipped_decoder = []
        skipped_other = []
        for k, v in pretrained_state.items():
            if any(k.startswith(p) for p in BLOCKED_PREFIXES):
                skipped_decoder.append(k)
                continue
            if not any(k.startswith(p) for p in ALLOWED_PREFIXES):
                skipped_other.append(k)
                continue
            if k in model_state and v.shape == model_state[k].shape:
                matched[k] = v
            else:
                skipped_other.append(f"{k} (shape mismatch)")
        model_state.update(matched)
        model.load_state_dict(model_state)
        missing = [k for k in model_state if k not in matched]
        print(f"Warm-started {len(matched)} keys (backbone+LoRA/TCAM) from {warm_start_path}")
        print(f"  Skipped decoder: {len(skipped_decoder)} keys (old R0 decoder)")
        print(f"  Skipped other: {len(skipped_other)} keys")
        print(f"  Missing (new model, random init): {len(missing)} keys (new SCHRR decoder.*)")
    elif warm_start_path:
        print(f"WARNING: warm_start path not found: {warm_start_path}")

    # ---- Backbone base freeze (v2: after warm_start, before optimizer) ----
    # Freeze base DINO weights, keep LoRA/TCAM trainable.
    # LoRA/TCAM params are under backbone.blocks.*.{qkv,fc1,fc2}.* with names
    # containing lora_/alpha_sem/alpha_str/spatial_conv/tcam — must except these.
    lr_groups_cfg = cfg.get("lr_groups", None)
    if lr_groups_cfg:
        PEFT_KEYWORDS = ("lora_", "alpha_sem", "alpha_str", "spatial_conv", "tcam")
        n_frozen = 0
        n_peft_kept = 0
        for name, param in model.named_parameters():
            if name.startswith("backbone.") and not any(k in name for k in PEFT_KEYWORDS):
                param.requires_grad = False
                n_frozen += 1
            elif name.startswith("backbone.") and any(k in name for k in PEFT_KEYWORDS):
                n_peft_kept += 1
        print(f"Backbone freeze: {n_frozen} base params frozen, {n_peft_kept} PEFT params kept trainable")

    # ---- Optimizer ----
    if lr_groups_cfg:
        # v2: 4-group LR (decoder/hr_encoder/router_gate=1.5e-4, lora_tcam=3.0e-5)
        optimizer = build_optimizer_with_groups(model, lr_groups_cfg, cfg.get("weight_decay", 0.05))
    else:
        # Legacy: single-LR or LoRA+ (for non-v2 configs)
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
        # v2: router temperature annealing (τ: 3.0→1.0 over anneal_epochs, then 1.0)
        router_cfg = cfg.get("decoder", {}).get("sc_cmrd_lar", {}).get("router", {})
        temp_cfg = router_cfg.get("temperature", {})
        if temp_cfg.get("enabled", False) and hasattr(model, "decoder") and hasattr(model.decoder, "set_router_temperature"):
            temp_start = temp_cfg.get("start", 3.0)
            temp_end = temp_cfg.get("end", 1.0)
            anneal_epochs = temp_cfg.get("anneal_epochs", 10)
            if epoch < anneal_epochs:
                tau = temp_start + (temp_end - temp_start) * (epoch / anneal_epochs)
            else:
                tau = temp_end
            model.decoder.set_router_temperature(tau)
            if ema is not None:
                ema.module.decoder.set_router_temperature(tau)
            print(f"Epoch {epoch}: router temperature τ={tau:.4f}")

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

                # Per-step diagnostics (populated for SC-CMRD-LAR; empty for others)
                step_diag = {}

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

                    # Native-scale aux supervision (SC-CMRD-LAR v2)
                    # Label is nearest-downsampled to aux resolution (not bilinear-up
                    # of logits). Aux heads learn native-scale discrimination.
                    if aux_outputs is not None and "aux_logits8" in aux_outputs:
                        aux_weight = cfg.get("decoder", {}).get(
                            "sc_cmrd_lar", {}
                        ).get("aux_weight", 0.4)
                        for key in ("aux_logits8", "aux_logits4", "aux_logits16"):
                            aux_logits = aux_outputs[key]
                            aux_h, aux_w = aux_logits.shape[-2:]
                            label_down = torch.nn.functional.interpolate(
                                masks.float().unsqueeze(1),  # (B,1,H,W) float
                                size=(aux_h, aux_w),
                                mode="nearest",
                            ).squeeze(1).long()  # (B,aux_h,aux_w) int
                            l_aux_ce = ce_loss_fn(aux_logits, label_down)
                            l_aux_dice = dice_loss_fn(aux_logits, label_down)
                            loss = loss + aux_weight * (l_aux_ce + dice_weight * l_aux_dice)
                            scale_tag = key.replace("aux_logits", "")
                            step_diag[f"aux_ce_{scale_tag}"] = l_aux_ce.item()
                            step_diag[f"aux_dice_{scale_tag}"] = l_aux_dice.item()

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
                        aff_weight = 0.0
                        l_aff_val = 0.0
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
                                l_aff_val = l_aff.item()
                        step_diag["aff_w"] = aff_weight
                        step_diag["aff_l"] = l_aff_val

                    # Router entropy regularization (v2: anti-collapse, from step 0)
                    # H = 0.5 * (H8 + H4), loss -= beta * H (maximize entropy)
                    router_cfg = cfg.get("decoder", {}).get("sc_cmrd_lar", {}).get("router", {})
                    ent_cfg = router_cfg.get("entropy_reg", {})
                    if ent_cfg.get("enabled", False) and aux_outputs is not None and "basis_probs_for_loss" in aux_outputs:
                        beta_start = ent_cfg.get("beta_start", 0.02)
                        beta_end = ent_cfg.get("beta_end", 0.005)
                        decay_epochs = ent_cfg.get("decay_epochs", 20)
                        # Linear decay beta_start -> beta_end over decay_epochs, then beta_end
                        if epoch < decay_epochs:
                            beta = beta_start + (beta_end - beta_start) * (epoch / decay_epochs)
                        else:
                            beta = beta_end
                        bp8 = aux_outputs["basis_probs_for_loss"].get("schrr_8")
                        bp4 = aux_outputs["basis_probs_for_loss"].get("schrr_4")
                        if bp8 is not None and bp4 is not None:
                            # fp32 + clamp_min: AMP softmax can produce exact 0 in fp16,
                            # causing NaN in 0*log(0). Cast to fp32 and clamp >= 1e-6.
                            bp8f = bp8.float().clamp_min(1e-6)
                            bp4f = bp4.float().clamp_min(1e-6)
                            H8 = -(bp8f * torch.log(bp8f)).sum(dim=1).mean()
                            H4 = -(bp4f * torch.log(bp4f)).sum(dim=1).mean()
                            H = 0.5 * (H8 + H4)  # average over scales, don't double beta
                            loss = loss - beta * H  # maximize entropy
                            step_diag["ent_beta"] = beta
                            step_diag["ent_H"] = H.item()

                    # Extract per-block model diagnostics (gate, coeff, lambda,
                    # basis usage, router entropy) from aux_outputs["diag"].
                    if aux_outputs is not None and "diag" in aux_outputs:
                        model_diag = aux_outputs["diag"]
                        for scale_key, scale_tag in [("schrr_8", "8"), ("schrr_4", "4")]:
                            d = model_diag.get(scale_key, {})
                            step_diag[f"gate_{scale_tag}"] = d.get(
                                "gate_mean", torch.tensor(float("nan"))
                            ).item()
                            step_diag[f"csem_{scale_tag}"] = d.get(
                                "coeff_sem_mean", torch.tensor(float("nan"))
                            ).item()
                            step_diag[f"cstr_{scale_tag}"] = d.get(
                                "coeff_str_mean", torch.tensor(float("nan"))
                            ).item()
                            step_diag[f"lam_{scale_tag}"] = d.get(
                                "lambda", torch.tensor(float("nan"))
                            ).item()
                            bp = d.get("basis_probs")
                            if bp is not None:
                                usage = bp.mean(dim=(0, 2, 3)).tolist()
                                for b_idx, u in enumerate(usage):
                                    step_diag[f"b{b_idx}_{scale_tag}"] = u
                                # fp32 + clamp_min: same numerical safety as entropy reg above.
                                bpf = bp.float().clamp_min(1e-6)
                                ent = -(
                                    bpf * torch.log(bpf)
                                ).sum(dim=1).mean().item()
                                step_diag[f"ent_{scale_tag}"] = ent

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
                # Compact SC-CMRD-LAR diagnostics: per-scale gate/coeff/lambda/basis/aux/aff
                if step_diag:
                    parts = []
                    for scale_tag in ("8", "4"):
                        if f"gate_{scale_tag}" in step_diag:
                            parts.append(
                                f"S{scale_tag}[g={step_diag[f'gate_{scale_tag}']:.3f},"
                                f"cs={step_diag[f'csem_{scale_tag}']:.3f},"
                                f"ct={step_diag[f'cstr_{scale_tag}']:.3f},"
                                f"l={step_diag[f'lam_{scale_tag}']:.3f},"
                                f"b=({step_diag.get(f'b0_{scale_tag}', 0):.2f},"
                                f"{step_diag.get(f'b1_{scale_tag}', 0):.2f},"
                                f"{step_diag.get(f'b2_{scale_tag}', 0):.2f}),"
                                f"e={step_diag.get(f'ent_{scale_tag}', 0):.3f},"
                                f"ax={step_diag.get(f'aux_ce_{scale_tag}', 0):.3f}]"
                            )
                    if "16" in str(step_diag.get("aux_ce_16", "")) or "aux_ce_16" in step_diag:
                        parts.append(f"Ax16={step_diag.get('aux_ce_16', 0):.3f}")
                    if "aff_w" in step_diag:
                        parts.append(f"Aff[w={step_diag['aff_w']:.4f},l={step_diag['aff_l']:.4f}]")
                    if "ent_beta" in step_diag:
                        parts.append(f"Ent[β={step_diag['ent_beta']:.4f},H={step_diag['ent_H']:.3f}]")
                    print(f"  DIAG: " + " ".join(parts))

            # TensorBoard per-step diagnostics (SC-CMRD-LAR only)
            if step_diag and writer is not None:
                for k, v in step_diag.items():
                    if isinstance(v, float) and (v != v):  # NaN check
                        continue
                    writer.add_scalar(f"diag/{k}", v, global_step)

        dt = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)

        # ---- Validation ----
        # Validate using EMA model when available (typically +0.3~0.7% mIoU)
        # v2: reset router temperature to 1.0 for validation (both model and EMA)
        if hasattr(model, "decoder") and hasattr(model.decoder, "set_router_temperature"):
            model.decoder.set_router_temperature(1.0)
            if ema is not None:
                ema.module.decoder.set_router_temperature(1.0)
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
