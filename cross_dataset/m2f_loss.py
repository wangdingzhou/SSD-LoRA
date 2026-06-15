"""
Mask2Former training support: Hungarian Matcher + Set Prediction Loss.
Designed to integrate with the existing HA-LoRA train_cross.py framework.
No detectron2 dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def point_sample(features, point_coords, align_corners=False):
    features = features.float()  # FP32 for grid_sample compatibility under AMP
    """Sample features at 2D point coordinates using grid_sample.
    
    Args:
        features: (B, C, H, W) or (N, C, H, W)
        point_coords: (B, K, 2) or (N, K, 2), values in [0, 1]
    Returns:
        (B, C, K) or (N, C, K) sampled features
    """
    # Normalize coords from [0,1] to [-1,1] for grid_sample
    grid = 2.0 * point_coords - 1.0  # (B, K, 2)
    grid = grid.unsqueeze(1)  # (B, 1, K, 2)
    
    # features: (B, C, H, W), grid: (B, 1, K, 2) -> (B, C, 1, K)
    sampled = F.grid_sample(features, grid, align_corners=align_corners, mode='bilinear')
    return sampled.squeeze(2)  # (B, C, K)


def semantic_masks_to_m2f_targets(masks, num_classes, ignore_index=255):
    """Convert semantic segmentation masks to M2F target format.
    
    Args:
        masks: (B, H, W) semantic label tensor
        num_classes: number of classes (excluding no-object)
        ignore_index: label value to ignore
    Returns:
        list of dicts: [{"labels": (N_i,), "masks": (N_i, H, W)}, ...]
    """
    B = masks.shape[0]
    targets = []
    for b in range(B):
        mask_b = masks[b]  # (H, W)
        labels = []
        binary_masks = []
        for c in range(num_classes):
            binary = (mask_b == c)
            if binary.any():
                labels.append(c)
                binary_masks.append(binary.float())
        
        if len(labels) == 0:
            # Empty image - still need at least one target
            labels = [num_classes]  # no-object
            binary_masks = [torch.zeros_like(mask_b, dtype=torch.float32)]
        
        targets.append({
            "labels": torch.tensor(labels, dtype=torch.int64, device=masks.device),
            "masks": torch.stack(binary_masks),  # (N_i, H, W)
        })
    return targets


def batch_dice_loss(inputs, targets):
    """Compute pairwise Dice cost matrix for matching.
    Args:
        inputs: (N_queries, K) sigmoid predictions at sampled points
        targets: (N_targets, K) binary ground truth at sampled points
    Returns:
        (N_queries, N_targets) cost matrix
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs, targets):
    """Compute pairwise BCE cost matrix for matching.
    Args:
        inputs: (N_queries, K) logits at sampled points
        targets: (N_targets, K) binary ground truth at sampled points
    Returns:
        (N_queries, N_targets) cost matrix
    """
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))
    return loss / hw


class HungarianMatcher(nn.Module):
    """Hungarian matcher for Mask2Former set prediction loss."""
    
    def __init__(self, cost_class=2.0, cost_mask=5.0, cost_dice=5.0, num_points=12544):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs: {"pred_logits": (B, Q, C+1), "pred_masks": (B, Q, H, W)}
            targets: [{"labels": (N_i,), "masks": (N_i, H, W)}, ...]
        Returns:
            list of (src_indices, tgt_indices) tuples
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []
        
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)  # (Q, C+1)
            tgt_ids = targets[b]["labels"]  # (N_i,)
            
            # Classification cost: -prob[target class]
            cost_class = -out_prob[:, tgt_ids]
            
            out_mask = outputs["pred_masks"][b]  # (Q, H, W)
            tgt_mask = targets[b]["masks"].to(out_mask)  # (N_i, H, W)
            
            # Point sampling for efficient matching
            if self.num_points > 0 and out_mask.shape[-1] > 1:
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
                # Sample predictions: (Q, 1, H, W) -> (Q, K)
                out_mask_sampled = point_sample(
                    out_mask.unsqueeze(1), point_coords.repeat(out_mask.shape[0], 1, 1)
                ).squeeze(1)
                # Sample targets: (N_i, 1, H, W) -> (N_i, K)
                tgt_mask_sampled = point_sample(
                    tgt_mask.unsqueeze(1), point_coords.repeat(tgt_mask.shape[0], 1, 1)
                ).squeeze(1)
            else:
                out_mask_sampled = out_mask.flatten(1)
                tgt_mask_sampled = tgt_mask.flatten(1)
            
            with torch.amp.autocast("cuda", enabled=False):
                out_mask_sampled = out_mask_sampled.float()
                tgt_mask_sampled = tgt_mask_sampled.float()
                cost_mask = batch_sigmoid_ce_loss(out_mask_sampled, tgt_mask_sampled)
                cost_dice = batch_dice_loss(out_mask_sampled, tgt_mask_sampled)
            
            C = (self.cost_class * cost_class + 
                 self.cost_mask * cost_mask + 
                 self.cost_dice * cost_dice)
            C = C.reshape(num_queries, -1).cpu()
            
            indices.append(linear_sum_assignment(C))
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]


class M2FSetCriterion(nn.Module):
    """Set prediction loss for Mask2Former (CE + BCE + Dice with deep supervision)."""
    
    def __init__(self, num_classes, matcher, weight_ce=2.0, weight_mask=5.0, weight_dice=5.0,
                 eos_coef=0.1, num_points=12544):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_ce = weight_ce
        self.weight_mask = weight_mask
        self.weight_dice = weight_dice
        self.num_points = num_points
        
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)
    
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def loss_labels(self, outputs, targets, indices, num_masks):
        src_logits = outputs["pred_logits"].float()
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_src_permutation_idx(indices)  # same structure
        
        src_masks = outputs["pred_masks"][src_idx]  # (N_matched, H, W)
        target_masks = torch.cat([t["masks"][J] for t, (_, J) in zip(targets, indices)]).to(src_masks)
        
        if self.num_points > 0 and src_masks.shape[-1] > 1:
            # Random point sampling for efficiency
            point_coords = torch.rand(1, self.num_points, 2, device=src_masks.device)
            src_sampled = point_sample(
                src_masks.unsqueeze(1), point_coords.repeat(src_masks.shape[0], 1, 1)
            ).squeeze(1)
            tgt_sampled = point_sample(
                target_masks.unsqueeze(1), point_coords.repeat(target_masks.shape[0], 1, 1)
            ).squeeze(1)
        else:
            src_sampled = src_masks.flatten(1)
            tgt_sampled = target_masks.flatten(1)
        
        with torch.amp.autocast("cuda", enabled=False):
            src_sampled = src_sampled.float()
            tgt_sampled = tgt_sampled.float()
            
            # BCE loss
            loss_mask = F.binary_cross_entropy_with_logits(src_sampled, tgt_sampled, reduction="none")
            loss_mask = loss_mask.mean(1).sum() / num_masks
            
            # Dice loss
            src_sig = src_sampled.sigmoid()
            numerator = 2 * (src_sig * tgt_sampled).sum(-1)
            denominator = src_sig.sum(-1) + tgt_sampled.sum(-1)
            loss_dice = (1 - (numerator + 1) / (denominator + 1)).sum() / num_masks
        
        return {"loss_mask": loss_mask, "loss_dice": loss_dice}
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: {"pred_logits", "pred_masks", "aux_outputs"}
            targets: [{"labels", "masks"}, ...]
        Returns:
            dict of loss tensors
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)
        
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.clamp(torch.tensor(num_masks, dtype=torch.float), min=1).item()
        
        losses = {}
        losses.update(self.loss_labels(outputs_without_aux, targets, indices, num_masks))
        losses.update(self.loss_masks(outputs_without_aux, targets, indices, num_masks))
        
        # Deep supervision on auxiliary outputs
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices_i = self.matcher(aux_outputs, targets)
                l_dict = {}
                l_dict.update(self.loss_labels(aux_outputs, targets, indices_i, num_masks))
                l_dict.update(self.loss_masks(aux_outputs, targets, indices_i, num_masks))
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)
        
        return losses


def compute_m2f_loss(criterion, outputs, masks, num_classes):
    """Compute M2F set prediction loss.
    
    Args:
        criterion: M2FSetCriterion instance
        outputs: dict from Mask2FormerDecoder.forward()
        masks: (B, H, W) semantic labels
        num_classes: number of classes
    Returns:
        total_loss, loss_dict
    """
    targets = semantic_masks_to_m2f_targets(masks, num_classes)
    loss_dict = criterion(outputs, targets)
    
    total = loss_dict["loss_ce"] * criterion.weight_ce
    total = total + loss_dict["loss_mask"] * criterion.weight_mask
    total = total + loss_dict["loss_dice"] * criterion.weight_dice
    
    # Auxiliary losses
    for i in range(9):  # up to 9 aux layers
        ce_key = f"loss_ce_{i}"
        mask_key = f"loss_mask_{i}"
        dice_key = f"loss_dice_{i}"
        if ce_key in loss_dict:
            total = total + loss_dict[ce_key] * criterion.weight_ce
            total = total + loss_dict[mask_key] * criterion.weight_mask
            total = total + loss_dict[dice_key] * criterion.weight_dice
    
    return total, loss_dict
