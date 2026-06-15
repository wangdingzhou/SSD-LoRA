"""HA-LoRA training script with AMP, TensorBoard, and checkpoint management.

Supports:
  - MLP / UPerNet decoder: CE + Dice loss
  - Mask2Former decoder: set prediction loss (Hungarian matching + CE + Dice + mask BCE)
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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import LoveDADataset
from model import count_trainable_parameters, get_model


# ---------------------------------------------------------------------------
# Losses -- shared
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        valid_mask = (targets != self.ignore_index).unsqueeze(1).float()
        targets_clamped = targets.clamp(0, num_classes - 1)
        targets_one_hot = F.one_hot(targets_clamped, num_classes).permute(0, 3, 1, 2).float()

        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


# ---------------------------------------------------------------------------
# Mask2Former Set Prediction Loss (Hungarian matching)
# ---------------------------------------------------------------------------

def dice_loss_masks(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute DICE loss for binary masks.

    Args:
        inputs: (N, H, W) predicted logits (before sigmoid).
        targets: (N, H, W) binary ground-truth masks.
    """
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1).float()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid focal cross-entropy loss for class prediction.

    Args:
        inputs: (B, Q, num_classes+1) predicted class logits.
        targets: (B, Q) target class indices (integer).
        num_classes: number of valid classes (excluding "no object").
        alpha: focal loss alpha weighting.
        gamma: focal loss gamma (focusing parameter).

    Returns:
        Scalar loss value.
    """
    ce_loss = F.cross_entropy(
        inputs.reshape(-1, inputs.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


class SetCriterion(nn.Module):
    """Mask2Former set prediction loss with Hungarian matching.

    Computes:
      - CE loss on predicted class labels
      - Dice loss on predicted masks
      - BCE loss on predicted masks
    with Hungarian matching between predictions and ground-truth segments.
    """

    def __init__(
        self,
        num_classes: int,
        weight_ce: float = 2.0,
        weight_dice: float = 5.0,
        weight_mask: float = 5.0,
        no_object_coef: float = 0.1,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_mask = weight_mask
        self.no_object_coef = no_object_coef
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        # Empty weight: down-weight "no object" class in CE loss
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = no_object_coef
        self.register_buffer("empty_weight", empty_weight)

    def _sample_point_coords(
        self, masks: torch.Tensor, gt_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample point coordinates for efficient mask loss computation.

        Uses importance sampling: sample more points near mask boundaries.

        Args:
            masks: (B, Q, H, W) predicted mask logits.
            gt_masks: (B, Q, H, W) ground-truth binary masks.

        Returns:
            point_coords: (B, Q, num_points, 2) sampled coordinates in [0, 1].
            point_labels: (B, Q, num_points) binary labels at sampled points.
        """
        B, Q, H, W = masks.shape
        num_points = self.num_points

        # Strategy: oversample, then subsample with importance
        num_oversample = int(num_points * self.oversample_ratio)
        num_important = int(num_points * self.importance_sample_ratio)
        num_random = num_points - num_important

        # Random coordinates in [0, 1]
        point_coords = torch.rand(B, Q, num_oversample, 2, device=masks.device)

        # Get predictions at sampled points
        # point_coords: (B, Q, num_oversample, 2)
        # Flatten for grid_sample: (B*Q, num_oversample, 2)
        flat_coords = point_coords.reshape(B * Q, num_oversample, 2)
        flat_masks = masks.reshape(B * Q, 1, H, W)
        # grid_sample expects (N, H_out, W_out, 2) -> need (B*Q, 1, 1, num_oversample)
        sampled_logits = F.grid_sample(
            flat_masks,
            flat_coords.unsqueeze(2),  # (B*Q, num_oversample, 1, 2)
            mode="bilinear",
            align_corners=False,
        )
        sampled_logits = sampled_logits.squeeze(2).reshape(B, Q, num_oversample)  # (B, Q, num_oversample)

        # Get GT at sampled points
        flat_gt = gt_masks.reshape(B * Q, 1, H, W).float()
        sampled_gt = F.grid_sample(
            flat_gt,
            flat_coords.unsqueeze(2),
            mode="nearest",
            align_corners=False,
        )
        sampled_gt = sampled_gt.squeeze(2).reshape(B, Q, num_oversample)

        # Sort by prediction uncertainty (close to 0.5 sigmoid = uncertain)
        # High absolute logit = certain, low = uncertain
        uncertainty = -sampled_logits.abs()  # higher = more uncertain
        _, topk_idx = uncertainty.topk(num_important, dim=-1)  # (B, Q, num_important)

        # Gather important points
        topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, 2)
        important_coords = point_coords.gather(2, topk_idx_expanded)  # (B, Q, num_important, 2)
        important_labels = sampled_gt.gather(2, topk_idx)  # (B, Q, num_important)

        # Random points
        random_coords = torch.rand(B, Q, num_random, 2, device=masks.device)
        flat_random = random_coords.reshape(B * Q, num_random, 2)
        random_labels = F.grid_sample(
            flat_gt,
            flat_random.unsqueeze(2),
            mode="nearest",
            align_corners=False,
        ).squeeze(2).reshape(B, Q, num_random)

        # Combine
        final_coords = torch.cat([important_coords, random_coords], dim=2)
        final_labels = torch.cat([important_labels, random_labels], dim=2)

        return final_coords, final_labels

    def _get_src_permutation_idx(self, indices: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Get source (prediction) indices from Hungarian matching results."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Get target (GT) indices from Hungarian matching results."""
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(
        self, outputs: dict, targets: list[dict]
    ) -> dict[str, torch.Tensor]:
        """Compute set prediction loss.

        Args:
            outputs: dict with keys:
                - pred_logits: (B, Q, num_classes+1) class predictions
                - pred_masks: (B, Q, H, W) mask predictions
                - aux_outputs: list of dicts with same keys (intermediate predictions)
            targets: list of B dicts with keys:
                - labels: (N_i,) class labels per instance
                - masks: (N_i, H, W) binary masks per instance

        Returns:
            dict of scalar losses: {loss_ce, loss_mask, loss_dice, ...}
        """
        # Compute matching cost and perform Hungarian matching
        indices = self._hungarian_match(outputs, targets)

        # Class loss (CE)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        # Create full target: for matched queries use GT label, for unmatched use "no object"
        num_queries = outputs["pred_logits"].shape[1]
        target_classes = torch.full(
            (len(targets), num_queries),
            self.num_classes,  # "no object" class index
            dtype=torch.int64,
            device=outputs["pred_logits"].device,
        )
        idx = self._get_src_permutation_idx(indices)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            outputs["pred_logits"].transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
        ) * self.weight_ce

        # Mask loss (point-sampled BCE + Dice) on matched pairs
        target_masks = torch.cat(
            [t["masks"][J] for t, (_, J) in zip(targets, indices)]
        )
        src_idx = self._get_src_permutation_idx(indices)
        pred_masks_matched = outputs["pred_masks"][src_idx]

        # Importance-sampled points (oversample + topk by uncertainty)
        B = len(targets)
        num_matched = pred_masks_matched.shape[0]
        if num_matched > 0:
            # Reshape to (B, Q, H, W) for _sample_point_coords
            H, W = pred_masks_matched.shape[-2:]
            matched_per_batch = [len(J) for (_, J) in indices]
            max_q = max(matched_per_batch) if matched_per_batch else 1
            pred_4d = torch.zeros(B, max_q, H, W, device=pred_masks_matched.device)
            gt_4d = torch.zeros(B, max_q, H, W, device=target_masks.device)
            offset = 0
            for b, (_, J) in enumerate(indices):
                nq = len(J)
                if nq > 0:
                    pred_4d[b, :nq] = pred_masks_matched[offset:offset + nq]
                    gt_4d[b, :nq] = target_masks[offset:offset + nq]
                    offset += nq
            point_coords, point_labels = self._sample_point_coords(pred_4d, gt_4d)
            # Flatten back to (num_matched, num_points, 2) / (num_matched, num_points)
            pc_flat = point_coords.reshape(-1, self.num_points, 2)[:num_matched]
            pl_flat = point_labels.reshape(-1, self.num_points)[:num_matched]

            # Sample logits at the same points for BCE
            point_logits = F.grid_sample(
                pred_masks_matched.unsqueeze(1).float(),
                pc_flat.unsqueeze(2),
                mode="bilinear", align_corners=False,
            ).squeeze(-1).squeeze(1)  # (N, 1, K, 1) -> (N, K)

            # BCE on point samples
            loss_bce = F.binary_cross_entropy_with_logits(point_logits, pl_flat.float())
            # Dice on point samples (shared points, following original M2F)
            point_probs = point_logits.sigmoid()
            numerator = 2 * (point_probs * pl_flat.float()).sum(-1)
            denominator = point_probs.sum(-1) + pl_flat.float().sum(-1)
            loss_dice = (1 - (numerator + 1) / (denominator + 1)).mean()
        else:
            loss_bce = pred_masks_matched.sum() * 0
            loss_dice = pred_masks_matched.sum() * 0
        loss_mask = self.weight_mask * loss_bce + self.weight_dice * loss_dice

        # Auxiliary losses
        aux_losses = []
        if "aux_outputs" in outputs:
            for i, aux_output in enumerate(outputs["aux_outputs"]):
                aux_indices = self._hungarian_match(aux_output, targets)

                # CE
                aux_target_classes_o = torch.cat(
                    [t["labels"][J] for t, (_, J) in zip(targets, aux_indices)]
                )
                aux_target_classes = torch.full(
                    (len(targets), num_queries),
                    self.num_classes,
                    dtype=torch.int64,
                    device=aux_output["pred_logits"].device,
                )
                aux_idx = self._get_src_permutation_idx(aux_indices)
                aux_target_classes[aux_idx] = aux_target_classes_o
                aux_ce = F.cross_entropy(
                    aux_output["pred_logits"].transpose(1, 2),
                    aux_target_classes,
                    weight=self.empty_weight,
                ) * self.weight_ce

                # Point-sampled mask loss for auxiliary outputs
                aux_target_masks = torch.cat(
                    [t["masks"][J] for t, (_, J) in zip(targets, aux_indices)]
                )
                aux_src_idx = self._get_src_permutation_idx(aux_indices)
                aux_pred_masks = aux_output["pred_masks"][aux_src_idx]
                aux_num_matched = aux_pred_masks.shape[0]
                if aux_num_matched > 0:
                    H, W = aux_pred_masks.shape[-2:]
                    aux_matched_per_batch = [len(J) for (_, J) in aux_indices]
                    aux_max_q = max(aux_matched_per_batch) if aux_matched_per_batch else 1
                    aux_pred_4d = torch.zeros(B, aux_max_q, H, W, device=aux_pred_masks.device)
                    aux_gt_4d = torch.zeros(B, aux_max_q, H, W, device=aux_target_masks.device)
                    offset = 0
                    for b, (_, J) in enumerate(aux_indices):
                        nq = len(J)
                        if nq > 0:
                            aux_pred_4d[b, :nq] = aux_pred_masks[offset:offset + nq]
                            aux_gt_4d[b, :nq] = aux_target_masks[offset:offset + nq]
                            offset += nq
                    aux_pc, aux_pl = self._sample_point_coords(aux_pred_4d, aux_gt_4d)
                    aux_pc_flat = aux_pc.reshape(-1, self.num_points, 2)[:aux_num_matched]
                    aux_pl_flat = aux_pl.reshape(-1, self.num_points)[:aux_num_matched]
                    aux_point_logits = F.grid_sample(
                        aux_pred_masks.unsqueeze(1).float(),
                        aux_pc_flat.unsqueeze(2),
                        mode="bilinear", align_corners=False,
                    ).squeeze(-1).squeeze(1)  # (N, 1, K, 1) -> (N, K)
                    aux_loss_bce = F.binary_cross_entropy_with_logits(
                        aux_point_logits, aux_pl_flat.float()
                    )
                    aux_point_probs = aux_point_logits.sigmoid()
                    aux_num = 2 * (aux_point_probs * aux_pl_flat.float()).sum(-1)
                    aux_den = aux_point_probs.sum(-1) + aux_pl_flat.float().sum(-1)
                    aux_loss_dice = (1 - (aux_num + 1) / (aux_den + 1)).mean()
                else:
                    aux_loss_bce = aux_pred_masks.sum() * 0
                    aux_loss_dice = aux_pred_masks.sum() * 0
                aux_mask = (
                    self.weight_mask * aux_loss_bce
                    + self.weight_dice * aux_loss_dice
                )

                aux_losses.append(aux_ce + aux_mask)

        total_loss = loss_ce + loss_mask
        for aux_l in aux_losses:
            total_loss = total_loss + aux_l

        return {
            "loss": total_loss,
            "loss_ce": loss_ce.detach(),
            "loss_mask": loss_mask.detach(),
        }

    def _hungarian_match(self, outputs: dict, targets: list[dict]) -> list[tuple]:
        """Perform Hungarian matching between predictions and targets.

        Args:
            outputs: dict with pred_logits and pred_masks.
            targets: list of B dicts with labels and masks.

        Returns:
            List of (src_indices, tgt_indices) tuples for each batch element.
        """
        from scipy.optimize import linear_sum_assignment

        B, Q = outputs["pred_logits"].shape[:2]
        num_classes_plus1 = outputs["pred_logits"].shape[2]
        H, W = outputs["pred_masks"].shape[2:]

        # Class probabilities: (B, Q, C+1)
        pred_probs = outputs["pred_logits"].softmax(-1)

        indices = []
        for b in range(B):
            # Number of GT segments in this image
            num_gt = targets[b]["masks"].shape[0]
            if num_gt == 0:
                # No GT objects: match nothing
                indices.append((
                    torch.tensor([], dtype=torch.long, device=outputs["pred_logits"].device),
                    torch.tensor([], dtype=torch.long, device=outputs["pred_logits"].device),
                ))
                continue

            # Cost matrix: (Q, num_gt)
            # 1. Class cost: -prob of target class
            gt_labels = targets[b]["labels"]  # (num_gt,)
            cost_class = -pred_probs[b, :, gt_labels]  # (Q, num_gt)

            # 2. Mask cost: dice cost + BCE cost
            pred_masks_b = outputs["pred_masks"][b]  # (Q, H, W)
            gt_masks_b = targets[b]["masks"]  # (num_gt, H, W)

            # Dice cost
            pred_probs_sigmoid = pred_masks_b.sigmoid().flatten(1)  # (Q, H*W)
            gt_masks_flat = gt_masks_b.flatten(1).float()  # (num_gt, H*W)
            numerator = 2 * (pred_probs_sigmoid @ gt_masks_flat.T)  # (Q, num_gt)
            denominator = pred_probs_sigmoid.sum(-1).unsqueeze(1) + gt_masks_flat.sum(-1).unsqueeze(0)
            cost_dice = 1 - (numerator + 1) / (denominator + 1)

            # BCE cost
            # Use point sampling for efficiency if masks are large
            if H * W > 10000:
                num_sample_points = min(12544, H * W)
                sample_idx = torch.randperm(H * W, device=pred_masks_b.device)[:num_sample_points]
                pred_sampled = pred_masks_b.flatten(1)[:, sample_idx]  # (Q, P)
                gt_sampled = gt_masks_b.flatten(1)[:, sample_idx].float()  # (num_gt, P)
                cost_bce = F.binary_cross_entropy_with_logits(
                    pred_sampled.unsqueeze(1).expand(-1, num_gt, -1),
                    gt_sampled.unsqueeze(0).expand(Q, -1, -1),
                    reduction="none",
                ).mean(-1)  # (Q, num_gt)
            else:
                cost_bce = F.binary_cross_entropy_with_logits(
                    pred_masks_b.unsqueeze(1).expand(-1, num_gt, -1, -1),
                    gt_masks_b.unsqueeze(0).expand(Q, -1, -1, -1).float(),
                    reduction="none",
                ).mean(dim=(2, 3))  # (Q, num_gt)

            # Total cost
            cost = (
                self.weight_ce * cost_class
                + self.weight_dice * cost_dice
                + self.weight_mask * cost_bce
            )
            cost = cost.detach().cpu().numpy()
            # Replace NaN/inf with large finite values (happens early in training)
            cost = np.nan_to_num(cost, nan=1e8, posinf=1e8, neginf=-1e8)

            # Hungarian matching
            row_ind, col_ind = linear_sum_assignment(cost)
            indices.append((
                torch.tensor(row_ind, dtype=torch.long, device=outputs["pred_logits"].device),
                torch.tensor(col_ind, dtype=torch.long, device=outputs["pred_logits"].device),
            ))

        return indices


def prepare_m2f_targets(masks: torch.Tensor, num_classes: int) -> list[dict]:
    """Convert dense segmentation masks to M2F instance-format targets.

    Args:
        masks: (B, H, W) integer class labels. 255 = ignore.
        num_classes: number of valid classes.

    Returns:
        List of B dicts with 'labels' and 'masks' tensors.
    """
    B = masks.shape[0]
    targets = []
    for b in range(B):
        mask_b = masks[b]  # (H, W)
        labels_list = []
        masks_list = []
        for cls_id in range(num_classes):
            binary_mask = (mask_b == cls_id)
            if binary_mask.any():
                labels_list.append(cls_id)
                masks_list.append(binary_mask)
        if len(labels_list) == 0:
            # No objects: create empty target
            labels_list = []
            masks_list = []
        labels_t = torch.tensor(labels_list, dtype=torch.long, device=masks.device)
        if len(masks_list) > 0:
            masks_t = torch.stack(masks_list)  # (N, H, W)
        else:
            masks_t = torch.zeros(0, mask_b.shape[0], mask_b.shape[1], dtype=torch.bool, device=masks.device)
        targets.append({"labels": labels_t, "masks": masks_t})
    return targets


def compute_m2f_loss(
    criterion: SetCriterion,
    outputs: dict,
    masks: torch.Tensor,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    """Compute M2F set prediction loss.

    Args:
        criterion: SetCriterion instance.
        outputs: model output dict {pred_logits, pred_masks, aux_outputs}.
        masks: (B, H, W) ground-truth labels.
        num_classes: number of classes.

    Returns:
        dict of losses.
    """
    targets = prepare_m2f_targets(masks, num_classes)
    # Downsample GT masks to match pred_masks feature resolution (e.g. 32x32)
    mask_h, mask_w = outputs["pred_masks"].shape[2:]
    for t in targets:
        if t["masks"].shape[0] > 0:
            t["masks"] = F.interpolate(
                t["masks"].unsqueeze(0).float(),
                size=(mask_h, mask_w),
                mode="nearest",
            ).squeeze(0).bool()
        else:
            t["masks"] = torch.zeros(
                0, mask_h, mask_w, dtype=torch.bool, device=t["masks"].device
            )
    return criterion(outputs, targets)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def get_polynomial_scheduler(optimizer, warmup_iters, max_iters, power=0.9):
    def lr_lambda(current_step):
        if current_step < warmup_iters:
            return float(current_step) / float(max(1, warmup_iters))
        progress = float(current_step - warmup_iters) / float(max(1, max_iters - warmup_iters))
        return max(0.0, (1.0 - progress) ** power)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, device, num_classes=7, decoder_type="mlp"):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes)

    for images, masks in dataloader:
        images = images.to(device)
        with torch.no_grad():
            if decoder_type == "mask2former":
                # M2F: use full precision for inference to avoid type mismatches
                with torch.amp.autocast("cuda", enabled=False):
                    m2f_out = model(images)
                if isinstance(m2f_out, dict):
                    logits = _m2f_output_to_logits(
                        m2f_out, num_classes, target_size=tuple(masks.shape[1:])
                    )
                else:
                    logits = m2f_out
            else:
                with autocast(enabled=True):
                    logits = model(images)

        preds = logits.argmax(dim=1).cpu()

        valid = masks != 255
        preds_flat = preds[valid].numpy()
        targets_flat = masks[valid].numpy()

        idx = targets_flat * num_classes + preds_flat
        counts = np.bincount(idx, minlength=num_classes * num_classes)
        confusion += torch.from_numpy(counts.reshape(num_classes, num_classes))

    iou_per_class = []
    for cls in range(num_classes):
        tp = confusion[cls, cls]
        union = confusion[cls].sum() + confusion[:, cls].sum() - tp
        iou_per_class.append((tp / union).item() if union > 0 else 0.0)

    miou = float(np.mean(iou_per_class))
    return miou, iou_per_class, confusion


def _m2f_output_to_logits(
    outputs: dict, num_classes: int, target_size: tuple = None
) -> torch.Tensor:
    """Convert M2F set-prediction output to per-pixel class logits.

    Args:
        outputs: dict with pred_logits (B, Q, C+1) and pred_masks (B, Q, H, W).
        num_classes: number of valid classes.
        target_size: (H, W) to upsample to. If None, return at native resolution.

    Returns:
        (B, num_classes, H, W) per-pixel logits.
    """
    pred_logits = outputs["pred_logits"].float()  # (B, Q, C+1)
    pred_masks = outputs["pred_masks"].sigmoid().float()  # (B, Q, H, W)

    # CRITICAL: softmax over ALL C+1 classes (including no-object), then take real classes
    # This ensures no-object queries contribute near-zero to real-class predictions
    class_probs = pred_logits.softmax(-1)[:, :, :num_classes]  # (B, Q, C)
    seg_logits = torch.einsum("bqc,bqhw->bchw", class_probs, pred_masks)

    if target_size is not None and seg_logits.shape[2:] != target_size:
        seg_logits = F.interpolate(
            seg_logits, size=target_size, mode="bilinear", align_corners=False
        )
    return seg_logits


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg):
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # ---- Detect decoder type ----
    decoder_type = cfg.get("decoder", {}).get("type", "mlp")
    is_m2f = decoder_type == "mask2former"
    num_classes = cfg.get("num_classes", 7)

    # ---- Data ----
    train_dataset = LoveDADataset(
        root=cfg.data_root, split="Train", transform=True, image_size=cfg.get("image_size", 512)
    )
    val_dataset = LoveDADataset(
        root=cfg.data_root, split="Val", transform=False, image_size=cfg.get("image_size", 512)
    )

    nw = cfg.get("num_workers", 4)
    bs = cfg.get("batch_size", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )

    # ---- Model ----
    model = get_model(OmegaConf.to_container(cfg, resolve=True))
    model = model.to(device)

    trainable = count_trainable_parameters(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,} ({100 * trainable / total:.2f}%)")

    # ---- Loss ----
    if is_m2f:
        m2f_loss_cfg = cfg.get("m2f_loss", {})
        m2f_criterion = SetCriterion(
            num_classes=num_classes,
            weight_ce=m2f_loss_cfg.get("weight_ce", 2.0),
            weight_dice=m2f_loss_cfg.get("weight_dice", 5.0),
            weight_mask=m2f_loss_cfg.get("weight_mask", 5.0),
            no_object_coef=m2f_loss_cfg.get("no_object_coef", 0.1),
        ).to(device)
        print(f"Using Mask2Former set prediction loss "
              f"(ce={m2f_criterion.weight_ce}, dice={m2f_criterion.weight_dice}, "
              f"mask={m2f_criterion.weight_mask})")
    else:
        ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        dice_loss = DiceLoss(ignore_index=255)
        dice_weight = cfg.get("dice_weight", 1.0)

    # ---- Optimizer (LoRA params: no weight decay; decoder params: with weight decay) ----
    lora_params = []
    decoder_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name or "alpha_" in name or "dw_conv" in name or "pw_conv" in name:
            lora_params.append(p)
        else:
            decoder_params.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "weight_decay": 0.0},
            {"params": decoder_params, "weight_decay": cfg.get("weight_decay", 0.05)},
        ],
        lr=cfg.get("lr", 1e-4),
    )
    print(f"LoRA params (no wd): {sum(p.numel() for p in lora_params):,}, "
          f"Decoder params (wd={cfg.get('weight_decay', 0.05)}): {sum(p.numel() for p in decoder_params):,}")

    # ---- Scheduler ----
    steps_per_epoch = len(train_loader)
    grad_accum_steps = cfg.get("grad_accum_steps", 1)
    max_iters = cfg.get("max_iters", 0)
    if max_iters <= 0:
        max_iters = cfg.get("epochs", 50) * (steps_per_epoch // grad_accum_steps)
    warmup_iters = cfg.get("warmup_iters", min(1500, max_iters // 10))
    scheduler = get_polynomial_scheduler(optimizer, warmup_iters, max_iters, cfg.get("scheduler_power", 0.9))

    # ---- AMP ----
    use_amp = cfg.get("fp16", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # ---- TensorBoard ----
    writer = SummaryWriter(log_dir=cfg.log_dir)

    # ---- Resume ----
    start_epoch = 0
    best_miou = 0.0
    best_path = os.path.join(cfg.output_dir, "best_model.pth")
    latest_path = os.path.join(cfg.output_dir, "latest_model.pth")

    def _remap_legacy_keys(state_dict):
        """Remap old dw_conv/pw_conv keys to spatial_conv for backward compat."""
        mapping = {}
        for k in list(state_dict.keys()):
            new_k = k
            if ".dw_conv." in k:
                new_k = k.replace(".dw_conv.", ".spatial_conv.0.")
            elif ".pw_conv." in k:
                new_k = k.replace(".pw_conv.", ".spatial_conv.1.")
            if new_k != k:
                mapping[k] = new_k
        for old_k, new_k in mapping.items():
            state_dict[new_k] = state_dict.pop(old_k)
        if mapping:
            print(f"  Remapped {len(mapping)} legacy keys (dw_conv/pw_conv -> spatial_conv)")
        return state_dict

    resume_path = cfg.get("resume", None)
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        ckpt["model"] = _remap_legacy_keys(ckpt["model"])
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", 0.0)
        print(f"Resumed from epoch {start_epoch}, best mIoU: {best_miou:.4f}")

    # ---- Training ----
    epochs = cfg.get("epochs", 50)
    iteration = start_epoch * steps_per_epoch

    print(f"\nStarting training: {epochs} epochs, {steps_per_epoch} steps/epoch, {max_iters} total iters")
    print(f"AMP: {use_amp}, Warmup: {warmup_iters} iters")
    print(f"Decoder: {decoder_type} ({'M2F set loss' if is_m2f else 'CE + Dice loss'})\n")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_dice = 0.0
        t0 = time.time()
        grad_accum_steps = cfg.get("grad_accum_steps", 1)
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                if is_m2f:
                    # M2F forward: returns dict with pred_logits, pred_masks, aux_outputs
                    outputs = model(images)
                    loss_dict = compute_m2f_loss(m2f_criterion, outputs, masks, num_classes)
                    loss = loss_dict["loss"] / grad_accum_steps
                    l_ce = loss_dict["loss_ce"]
                    l_dice = loss_dict.get("loss_mask", torch.tensor(0.0))
                else:
                    # Standard forward: returns (B, C, H, W) logits
                    logits = model(images)
                    l_ce = ce_loss(logits, masks)
                    l_dice = dice_loss(logits, masks)
                    loss = (l_ce + dice_weight * l_dice) / grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            epoch_loss += loss.item() * grad_accum_steps
            epoch_ce += l_ce.item() if isinstance(l_ce, torch.Tensor) else l_ce
            epoch_dice += l_dice.item() if isinstance(l_dice, torch.Tensor) else l_dice
            iteration += 1

            if batch_idx % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch + 1}/{epochs}] Iter [{iteration}/{max_iters}] "
                    f"Loss: {loss.item():.4f} (CE: {l_ce.item() if isinstance(l_ce, torch.Tensor) else l_ce:.4f} "
                    f"{'Mask' if is_m2f else 'Dice'}: {l_dice.item() if isinstance(l_dice, torch.Tensor) else l_dice:.4f}) "
                    f"LR: {lr:.6f}"
                )

        avg_loss = epoch_loss / len(train_loader)
        avg_ce = epoch_ce / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)
        epoch_time = time.time() - t0

        # ---- Validate ----
        miou, iou_per_class, confusion = evaluate(
            model, val_loader, device, num_classes, decoder_type
        )

        print(f"\nEpoch [{epoch + 1}/{epochs}] ({epoch_time:.0f}s) "
              f"Loss: {avg_loss:.4f} | Val mIoU: {miou:.4f}")
        for i, iou_val in enumerate(iou_per_class):
            name = LoveDADataset.CLASS_NAMES[i] if i < len(LoveDADataset.CLASS_NAMES) else f"Class{i}"
            print(f"  {name}: {iou_val:.4f}")
        print()

        # ---- TensorBoard ----
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/ce", avg_ce, epoch)
        if is_m2f:
            writer.add_scalar("Loss/mask", avg_dice, epoch)
        else:
            writer.add_scalar("Loss/dice", avg_dice, epoch)
        writer.add_scalar("mIoU/val", miou, epoch)
        for i, iou_val in enumerate(iou_per_class):
            name = LoveDADataset.CLASS_NAMES[i] if i < len(LoveDADataset.CLASS_NAMES) else f"Class{i}"
            writer.add_scalar(f"IoU/{name}", iou_val, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # ---- Save checkpoints ----
        ckpt_dict = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_miou": best_miou,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        }

        if miou > best_miou:
            best_miou = miou
            torch.save(ckpt_dict, best_path)
            print(f"  ** New best mIoU: {best_miou:.4f} **\n")

        torch.save(ckpt_dict, latest_path)

    writer.close()
    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")
    return best_miou


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HA-LoRA Training")
    parser.add_argument("--config", type=str, default="src/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    train(cfg)
