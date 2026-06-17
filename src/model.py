"""
HA-LoRA: Hierarchical Adaptive LoRA with Semantic-Structural Decoupling
Model implementation for remote sensing semantic segmentation.
"""

import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

DINOV3_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dinov3")
if DINOV3_DIR not in sys.path:
    sys.path.insert(0, DINOV3_DIR)

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ---------------------------------------------------------------------------
# DINOv3 backbone builder
# ---------------------------------------------------------------------------

def _load_checkpoint_file(checkpoint: str) -> Dict[str, torch.Tensor]:
    """Load a PyTorch or safetensors checkpoint into a CPU state dict."""
    if checkpoint.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading .safetensors checkpoints requires the safetensors package."
            ) from exc
        return load_file(checkpoint, device="cpu")

    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict


def _is_hf_dinov3_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("embeddings.") or k.startswith("layer.") for k in state_dict)


def _allowed_hf_missing_key(key: str) -> bool:
    return (
        key == "rope_embed.periods"
        or key.startswith("local_cls_norm.")
        or key.endswith(".attn.qkv.bias_mask")
    )


def _convert_hf_dinov3_state_dict(
    state_dict: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Convert HuggingFace/ModelScope DINOv3 ViT keys to the local Meta keys."""
    converted: Dict[str, torch.Tensor] = {}
    consumed = set()

    def put(src: str, dst: str) -> None:
        if src in state_dict:
            converted[dst] = state_dict[src]
            consumed.add(src)

    put("embeddings.cls_token", "cls_token")
    put("embeddings.mask_token", "mask_token")
    put("embeddings.register_tokens", "storage_tokens")
    put("embeddings.patch_embeddings.weight", "patch_embed.proj.weight")
    put("embeddings.patch_embeddings.bias", "patch_embed.proj.bias")
    put("norm.weight", "norm.weight")
    put("norm.bias", "norm.bias")

    block_ids = sorted(
        {
            int(match.group(1))
            for key in state_dict
            for match in [re.match(r"layer\.(\d+)\.", key)]
            if match is not None
        }
    )
    for idx in block_ids:
        src_prefix = f"layer.{idx}"
        dst_prefix = f"blocks.{idx}"
        for suffix in ("weight", "bias"):
            q_key = f"{src_prefix}.attention.q_proj.{suffix}"
            k_key = f"{src_prefix}.attention.k_proj.{suffix}"
            v_key = f"{src_prefix}.attention.v_proj.{suffix}"
            if suffix == "bias" and q_key in state_dict and v_key in state_dict and k_key not in state_dict:
                converted[f"{dst_prefix}.attn.qkv.{suffix}"] = torch.cat(
                    [state_dict[q_key], torch.zeros_like(state_dict[q_key]), state_dict[v_key]], dim=0
                )
                consumed.update({q_key, v_key})
                continue

            has_any = any(k in state_dict for k in (q_key, k_key, v_key))
            has_all = all(k in state_dict for k in (q_key, k_key, v_key))
            if has_any and not has_all:
                raise KeyError(f"Incomplete HF q/k/v projection for block {idx} {suffix}")
            if has_all:
                converted[f"{dst_prefix}.attn.qkv.{suffix}"] = torch.cat(
                    [state_dict[q_key], state_dict[k_key], state_dict[v_key]], dim=0
                )
                consumed.update({q_key, k_key, v_key})

        put(f"{src_prefix}.attention.o_proj.weight", f"{dst_prefix}.attn.proj.weight")
        put(f"{src_prefix}.attention.o_proj.bias", f"{dst_prefix}.attn.proj.bias")
        put(f"{src_prefix}.norm1.weight", f"{dst_prefix}.norm1.weight")
        put(f"{src_prefix}.norm1.bias", f"{dst_prefix}.norm1.bias")
        put(f"{src_prefix}.norm2.weight", f"{dst_prefix}.norm2.weight")
        put(f"{src_prefix}.norm2.bias", f"{dst_prefix}.norm2.bias")
        put(f"{src_prefix}.mlp.up_proj.weight", f"{dst_prefix}.mlp.fc1.weight")
        put(f"{src_prefix}.mlp.up_proj.bias", f"{dst_prefix}.mlp.fc1.bias")
        put(f"{src_prefix}.mlp.down_proj.weight", f"{dst_prefix}.mlp.fc2.weight")
        put(f"{src_prefix}.mlp.down_proj.bias", f"{dst_prefix}.mlp.fc2.bias")
        put(f"{src_prefix}.layer_scale1.lambda1", f"{dst_prefix}.ls1.gamma")
        put(f"{src_prefix}.layer_scale2.lambda1", f"{dst_prefix}.ls2.gamma")

    unexpected = sorted(set(state_dict) - consumed)
    if unexpected:
        raise KeyError(f"Unexpected HF DINOv3 keys: {unexpected[:20]}")

    missing = sorted(k for k in model_state if k not in converted and not _allowed_hf_missing_key(k))
    if missing:
        raise KeyError(f"HF DINOv3 conversion did not produce required keys: {missing[:20]}")

    full_state = dict(model_state)
    for key in full_state:
        if key.endswith(".attn.qkv.bias_mask"):
            full_state[key] = torch.zeros_like(full_state[key])

    for key, value in converted.items():
        if key not in model_state:
            raise KeyError(f"Converted key not present in model: {key}")
        if tuple(value.shape) != tuple(model_state[key].shape):
            if value.numel() == model_state[key].numel():
                value = value.reshape_as(model_state[key])
            else:
                raise ValueError(
                    f"Shape mismatch for {key}: checkpoint {tuple(value.shape)} "
                    f"!= model {tuple(model_state[key].shape)}"
                )
        full_state[key] = value
    return full_state


def _build_dinov3_backbone(name: str, checkpoint: Optional[str] = None) -> nn.Module:
    """Build DINOv3 ViT backbone and optionally load pretrained weights."""
    from dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16, dinov3_vitl16, Weights

    # ViT-L SAT493M needs weights=Weights.SAT493M so the factory sets
    # untie_global_and_local_cls_norm=True internally.
    extra_kwargs = {}
    if name == "vitl16" and checkpoint and "sat493m" in checkpoint.lower():
        extra_kwargs["weights"] = Weights.SAT493M

    factories = {"vits16": dinov3_vits16, "vitb16": dinov3_vitb16, "vitl16": dinov3_vitl16}
    if name not in factories:
        raise ValueError(f"Unknown backbone: {name}. Choose from {list(factories.keys())}")

    model = factories[name](pretrained=False, **extra_kwargs)

    if checkpoint:
        state_dict = _load_checkpoint_file(checkpoint)
        if _is_hf_dinov3_state_dict(state_dict):
            state_dict = _convert_hf_dinov3_state_dict(state_dict, model.state_dict())
            print("Converted HuggingFace/ModelScope DINOv3 checkpoint to local key format")
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded DINOv3 {name} weights from {checkpoint}")
    else:
        print(
            f"WARNING: DINOv3 {name} initialized with random weights. "
            f"Set backbone.checkpoint in config to load pretrained weights."
        )

    return model


# ---------------------------------------------------------------------------
# Multi-Scale DW-Conv for Structural Path Enhancement
# ---------------------------------------------------------------------------

class MultiScaleDWConv(nn.Module):
    """Multi-scale depthwise separable convolution (dilation={1,2,3}).

    Replaces single DW-Conv in SSD-LoRA structural path.
    Different dilation rates capture different spatial frequencies:
        d=1: local edges/textures (high-freq, RF 3x3)
        d=2: medium structures (mid-freq, RF 7x7)
        d=3: large context (low-freq, RF 11x11)
    """

    def __init__(self, channels: int, dilations: tuple = (1, 2, 3)):
        super().__init__()
        self.n_scales = len(dilations)
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=d, dilation=d, groups=channels, bias=False)
            for d in dilations
        ])
        self.pw = nn.Conv2d(channels * self.n_scales, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [dw(x) for dw in self.dw_convs]
        return self.pw(torch.cat(outs, dim=1))




# ---------------------------------------------------------------------------
# Texture Channel Attention Module (TCAM) for Structural Path
# ---------------------------------------------------------------------------

class TCAM(nn.Module):
    """Texture Channel Attention Module.

    Computes per-sample, per-channel scaling from token covariance/Gram.
    Used to modulate SSD-LoRA structural path's spatial tokens.

    Input:  h2d of shape (B, r, H, W)
    Output: per-channel scale s of shape (B, r, 1, 1)

    Paths:
      - use_mlp=True:
          centered_cov or normalized_gram -> diag -> log1p -> MLP -> s
          s = 1 + gamma * tanh(mlp_output)
      - use_mlp=False (control):
          s = 1 (identity; isolates MLP+gate contribution; dwconv still applied)

    diag computed in fp32 to preserve magnitude in mixed-precision training.
    No per-sample standardization: keeps absolute texture energy for
    hard/easy patch interpretability.
    """

    def __init__(
        self,
        channels: int,
        tcam_type: str = "cov",        # "cov" (centered) or "gram" (non-centered)
        use_mlp: bool = True,
        gamma: float = 0.2,
        hidden_min: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.tcam_type = tcam_type
        self.use_mlp = use_mlp
        self.gamma = gamma
        self.hidden_min = hidden_min

        if use_mlp:
            hidden = max(hidden_min, 2 * channels)
            self.mlp = nn.Sequential(
                nn.Linear(channels, hidden),
                nn.GELU(),
                nn.Linear(hidden, channels),
            )
            # zero-init last layer: initial s = 1 + gamma*tanh(0) = 1
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h2d: torch.Tensor) -> torch.Tensor:
        """h2d: (B, r, H, W) -> s: (B, r, 1, 1)"""
        B, r, H, W = h2d.shape

        if not self.use_mlp:
            return torch.ones(B, r, 1, 1, device=h2d.device, dtype=h2d.dtype)

        # Disable autocast inside TCAM: stats and MLP both run in fp32 for stability.
        # If module was .half()ed (e.g. pure-fp16 inference), cast MLP weights to fp32.
        weight_dtype = next(self.mlp.parameters()).dtype
        with torch.autocast(device_type=h2d.device.type, enabled=False):
            h2d_fp32 = h2d.float()
            h_flat = h2d_fp32.reshape(B, r, H * W)

            if self.tcam_type == "cov":
                mean = h_flat.mean(dim=2, keepdim=True)          # (B, r, 1)
                centered = h_flat - mean                          # (B, r, N)
                diag = (centered ** 2).mean(dim=2)                # (B, r) per-channel variance
            elif self.tcam_type == "gram":
                diag = (h_flat ** 2).mean(dim=2)                  # (B, r) per-channel energy
            else:
                raise ValueError(f"Unknown tcam_type: {self.tcam_type}")

            # log1p preserves magnitude, compresses dynamic range
            diag = torch.log1p(diag.clamp(min=0))

            mlp = self.mlp.float() if weight_dtype != torch.float32 else self.mlp
            s = mlp(diag)                                        # (B, r)
            s = 1.0 + self.gamma * torch.tanh(s)                 # (B, r)

        return s.view(B, r, 1, 1).to(h2d.dtype)


# ---------------------------------------------------------------------------
# SSD-LoRA Module
# ---------------------------------------------------------------------------

class SSDLoRAModule(nn.Module):
    """
    Semantic-Structural Decoupled LoRA module.

    output = W_frozen(x)
           + alpha_sem * A_sem @ (B_sem @ x)
           + alpha_str * A_str @ spatial_path(B_str @ x)

    The structural path applies DW-Conv3x3 + Conv1x1 to the spatial (patch)
    tokens only; CLS and storage tokens bypass the convolution.
    """

    def __init__(
        self,
        frozen_linear: nn.Linear,
        r_sem: int,
        r_str: int,
        n_non_spatial: int = 5,
        lora_mode: str = "ssd",
        structural_path_type: str = "single",
        use_rslora: bool = False,
        tcam_type: Optional[str] = None,
        tcam_gamma: float = 0.2,
        tcam_hidden_min: int = 8,
    ):
        super().__init__()
        self.frozen_linear = frozen_linear
        self.r_sem = r_sem
        self.r_str = r_str
        self.n_non_spatial = n_non_spatial
        self.lora_mode = lora_mode
        self.use_rslora = use_rslora
        self.tcam_type = tcam_type
        self.tcam_gamma = tcam_gamma
        self.tcam_hidden_min = tcam_hidden_min
        # rsLoRA scaling factors precomputed (alpha / sqrt(r))
        self._rslora_scale_sem = 1.0 / math.sqrt(r_sem) if (use_rslora and r_sem > 0) else 1.0
        self._rslora_scale_str = 1.0 / math.sqrt(r_str) if (use_rslora and r_str > 0) else 1.0

        d_in = frozen_linear.in_features
        d_out = frozen_linear.out_features

        # Expose attributes that downstream code may access
        self.in_features = frozen_linear.in_features
        self.out_features = frozen_linear.out_features

        # Freeze original weights
        self.frozen_linear.weight.requires_grad_(False)
        if self.frozen_linear.bias is not None:
            self.frozen_linear.bias.requires_grad_(False)

        # Semantic path: standard LoRA
        if r_sem > 0:
            self.lora_sem_A = nn.Parameter(torch.empty(d_out, r_sem))
            self.lora_sem_B = nn.Parameter(torch.zeros(r_sem, d_in))
            nn.init.kaiming_uniform_(self.lora_sem_A, a=math.sqrt(5))
            self.alpha_sem = nn.Parameter(torch.ones(1))

        # Structural path: LoRA with spatial processing
        if r_str > 0:
            self.lora_str_A = nn.Parameter(torch.empty(d_out, r_str))
            self.lora_str_B = nn.Parameter(torch.zeros(r_str, d_in))
            nn.init.kaiming_uniform_(self.lora_str_A, a=math.sqrt(5))
            self.alpha_str = nn.Parameter(torch.ones(1))

            if lora_mode in ("ssd", "conv"):
                if structural_path_type == "multi_scale_dwconv":
                    self.spatial_conv = MultiScaleDWConv(r_str, dilations=(1, 2, 3))
                else:
                    # "single" base path; also serves as TCAM preprocessing
                    self.spatial_conv = nn.Sequential(
                        nn.Conv2d(r_str, r_str, 3, padding=1, groups=r_str, bias=False),
                        nn.Conv2d(r_str, r_str, 1, bias=False),
                    )

                # TCAM (optional; only when this block is selected)
                if tcam_type is not None:
                    if tcam_type.endswith("_ln"):
                        raise NotImplementedError(
                            f"TCAM variant '{tcam_type}' is reserved for later, "
                            f"not implemented in this round"
                        )
                    if tcam_type.endswith("_nomlp"):
                        use_mlp_flag = False
                        base = tcam_type[:-len("_nomlp")]
                    else:
                        use_mlp_flag = True
                        base = tcam_type
                    if base == "tcam_cov":
                        cov_or_gram = "cov"
                    elif base == "tcam_gram":
                        cov_or_gram = "gram"
                    else:
                        raise ValueError(f"Unknown tcam_type base: {base}")
                    self.tcam = TCAM(
                        r_str,
                        tcam_type=cov_or_gram,
                        use_mlp=use_mlp_flag,
                        gamma=tcam_gamma,
                        hidden_min=tcam_hidden_min,
                    )
                else:
                    self.tcam = None

    def _apply_spatial_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial convolution to spatial (patch) tokens only."""
        B, N, r = x.shape
        n_ns = self.n_non_spatial
        non_spatial = x[:, :n_ns]
        spatial = x[:, n_ns:]

        h = w = int(math.sqrt(spatial.shape[1]))
        spatial = spatial.reshape(B, h, w, r).permute(0, 3, 1, 2)
        spatial = self.spatial_conv(spatial)
        spatial = spatial.permute(0, 2, 3, 1).reshape(B, h * w, r)

        return torch.cat([non_spatial, spatial], dim=1)

    def _apply_spatial_conv_with_tcam(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial conv + TCAM per-channel modulation to spatial tokens."""
        B, N, r = x.shape
        n_ns = self.n_non_spatial
        non_spatial = x[:, :n_ns]
        spatial = x[:, n_ns:]

        N_spatial = spatial.shape[1]
        side = int(math.sqrt(N_spatial))
        assert side * side == N_spatial, (
            f"Spatial tokens {N_spatial} not a perfect square; "
            f"check image_size and patch_size"
        )

        spatial_2d = spatial.reshape(B, side, side, r).permute(0, 3, 1, 2)

        # 1. base dwconv preprocessing (single or multi_scale_dwconv)
        spatial_2d = self.spatial_conv(spatial_2d)

        # 2. TCAM per-channel scaling (s = 1 + gamma * tanh(mlp(diag)))
        if self.tcam is not None:
            scale = self.tcam(spatial_2d)                  # (B, r, 1, 1)
            spatial_2d = spatial_2d * scale

        spatial_out = spatial_2d.permute(0, 2, 3, 1).reshape(B, N_spatial, r)
        return torch.cat([non_spatial, spatial_out], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.frozen_linear(x)

        if self.r_sem > 0:
            sem = (x @ self.lora_sem_B.T) @ self.lora_sem_A.T
            result = result + (self.alpha_sem * self._rslora_scale_sem) * sem

        if self.r_str > 0:
            str_out = x @ self.lora_str_B.T
            if self.lora_mode in ("ssd", "conv"):
                if self.tcam is not None:
                    str_out = self._apply_spatial_conv_with_tcam(str_out)
                else:
                    str_out = self._apply_spatial_conv(str_out)
            str_out = str_out @ self.lora_str_A.T
            result = result + (self.alpha_str * self._rslora_scale_str) * str_out

        return result


# ---------------------------------------------------------------------------
# Feature Fusion
# ---------------------------------------------------------------------------

class FeatureFusionModule(nn.Module):
    """Multi-layer feature extraction, channel reduction, and fusion."""

    def __init__(self, in_channels_list: List[int], embed_dim: int):
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, embed_dim, 1),
                    nn.BatchNorm2d(embed_dim),
                    nn.GELU(),
                )
                for c in in_channels_list
            ]
        )
        fused_dim = embed_dim * len(in_channels_list)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_dim, fused_dim, 3, padding=1, groups=fused_dim, bias=False),
            nn.BatchNorm2d(fused_dim),
            nn.GELU(),
            nn.Conv2d(fused_dim, fused_dim, 1, bias=False),
            nn.BatchNorm2d(fused_dim),
            nn.GELU(),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[2:]
        projected = []
        for feat, proj in zip(features, self.projections):
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            projected.append(proj(feat))
        fused = torch.cat(projected, dim=1)
        return self.fuse(fused)


# ---------------------------------------------------------------------------
# MLP Decoder (R0 baseline)
# ---------------------------------------------------------------------------

class MLPDecoder(nn.Module):
    """Lightweight MLP decoder for segmentation."""

    def __init__(self, in_channels_list: List[int], embed_dim: int = 256,
                 num_classes: int = 7):
        super().__init__()
        self.fusion = FeatureFusionModule(in_channels_list, embed_dim)
        fused_dim = embed_dim * len(in_channels_list)
        self.fused_dim = fused_dim
        self.head = nn.Sequential(
            nn.Conv2d(fused_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, num_classes, 1),
        )

    def forward(self, features: List[torch.Tensor], img_size: Tuple[int, int]) -> torch.Tensor:
        fused = self.fusion(features)
        out = F.interpolate(fused, size=img_size, mode="bilinear", align_corners=False)
        return self.head(out)


# ---------------------------------------------------------------------------
# HA-LoRA Segmentation Model
# ---------------------------------------------------------------------------

class HALoRASeg(nn.Module):
    """HA-LoRA: Hierarchical Adaptive LoRA with Semantic-Structural Decoupling."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        # --- Backbone ---
        backbone_cfg = cfg.get("backbone", {})
        backbone_name = backbone_cfg.get("name", "vits16")
        checkpoint = backbone_cfg.get("checkpoint", None)
        self.backbone = _build_dinov3_backbone(backbone_name, checkpoint)
        self.embed_dim = self.backbone.embed_dim
        self.depth = len(self.backbone.blocks)
        self.n_storage_tokens = self.backbone.n_storage_tokens
        self.patch_size = self.backbone.patch_size

        # Freeze all backbone params first; SSD-LoRA params are added after
        for param in self.backbone.parameters():
            param.requires_grad = False

        # --- SSD-LoRA ---
        lora_cfg = cfg.get("lora", {})
        # lora_blocks: 1-indexed in config → 0-indexed internally
        if "lora_blocks" in lora_cfg and lora_cfg["lora_blocks"] is not None:
            self._lora_blocks_0idx = [b - 1 for b in lora_cfg["lora_blocks"]]
        else:
            self._lora_blocks_0idx = None
        self._apply_ssd_lora(lora_cfg)

        # --- Feature extraction layers (1-indexed in config, 0-indexed internally) ---
        decoder_type = cfg.get("decoder", {}).get("type", "mlp")
        if "layers_to_extract" in cfg:
            self.layers_to_extract = cfg["layers_to_extract"]
        else:
            self.layers_to_extract = [3, 6, 9, 12]
        self.layers_to_extract_0idx = [l - 1 for l in self.layers_to_extract]

        # --- Decoder ---
        decoder_cfg = cfg.get("decoder", {})
        decoder_embed_dim = decoder_cfg.get("embed_dim", 96 if self.embed_dim < 512 else 256)
        num_classes = cfg.get("num_classes", 7)
        in_channels = [self.embed_dim] * len(self.layers_to_extract)

        if decoder_type == "mlp":
            self.decoder = MLPDecoder(in_channels, decoder_embed_dim, num_classes)
        else:
            raise ValueError(
                f"Unknown decoder type: {decoder_type}. "
                f"Only 'mlp' is supported after cleanup. "
                f"Historical configs (pfu/sru/upernet/mask2former/sc_cmrd_lar) "
                f"are abandoned and no longer buildable."
            )

    # ---- rank helpers ----

    def _get_rank_for_block(self, block_idx: int) -> Tuple[int, int]:
        if self._lora_blocks_0idx is not None and block_idx not in self._lora_blocks_0idx:
            return (0, 0)

        lora_cfg = self.cfg.get("lora", {})

        rank_config = lora_cfg.get("rank_config", "progressive")
        semantic_only = lora_cfg.get("semantic_only", False)

        if rank_config == "progressive":
            progressive_ranks = self.cfg.get("lora", {}).get("progressive_ranks", None)
            if progressive_ranks:
                n_segments = len(progressive_ranks)
                segment_size = max(1, self.depth // n_segments)
                seg_idx = min(block_idx // segment_size, n_segments - 1)
                r = progressive_ranks[seg_idx]
                return (r, 0) if semantic_only else (r // 2, r // 2)
            if self.depth <= 12:
                if block_idx < 4: r = 4
                elif block_idx < 8: r = 8
                else: r = 16
            else:  # 24-block ViT-L
                if block_idx < 8: r = 4
                elif block_idx < 16: r = 8
                else: r = 16
            return (r, 0) if semantic_only else (r // 2, r // 2)
        elif rank_config == "uniform":
            r = self.cfg.get("lora", {}).get("uniform_rank", 8)
            return (r, 0) if semantic_only else (r // 2, r // 2)
        else:
            raise ValueError(f"Unknown rank config: {rank_config}")

    # ---- SSD-LoRA application ----

    def _apply_ssd_lora(self, lora_cfg: dict):
        target_modules = lora_cfg.get("target_modules", ["qkv", "fc1", "fc2"])
        lora_mode = lora_cfg.get("mode", "ssd")
        structural_path_type = lora_cfg.get("structural_path_type", "single")
        n_non_spatial = 1 + self.n_storage_tokens
        alpha_init = lora_cfg.get("alpha_init", 1.0)
        use_rslora = lora_cfg.get("use_rslora", False)

        # TCAM config (new in Phase 3b-S)
        # If structural_path_type starts with "tcam_", enter TCAM mode.
        # tcam_base_path controls non-TCAM blocks (must be single/multi_scale_dwconv).
        # tcam_blocks (1-indexed in config) restricts TCAM to a subset.
        if structural_path_type.startswith("tcam_"):
            tcam_type = structural_path_type
            base_structural_path = lora_cfg.get("tcam_base_path", "single")
            if base_structural_path not in ("single", "multi_scale_dwconv"):
                raise ValueError(
                    f"tcam_base_path must be 'single' or 'multi_scale_dwconv', "
                    f"got: {base_structural_path}"
                )
        else:
            tcam_type = None
            base_structural_path = structural_path_type

        tcam_gamma = lora_cfg.get("tcam_gamma", 0.2)
        tcam_hidden_min = lora_cfg.get("tcam_hidden_min", 8)

        # tcam_blocks: 1-indexed in config -> 0-indexed internally
        tcam_blocks_cfg = lora_cfg.get("tcam_blocks", None)
        if tcam_blocks_cfg is not None:
            tcam_blocks_0idx = sorted(set(int(b) - 1 for b in tcam_blocks_cfg))
            # validate range
            invalid = [b + 1 for b in tcam_blocks_0idx if not (0 <= b < self.depth)]
            if invalid:
                raise ValueError(
                    f"tcam_blocks contains invalid 1-indexed block numbers: {invalid}. "
                    f"Valid range: 1..{self.depth}"
                )
        else:
            tcam_blocks_0idx = None  # means all blocks (only meaningful if tcam_type set)

        n_tcam_modules = 0
        for i, block in enumerate(self.backbone.blocks):
            r_sem, r_str = self._get_rank_for_block(i)
            if r_sem == 0 and r_str == 0:
                continue

            # Decide per-block TCAM
            if (tcam_type is not None and
                (tcam_blocks_0idx is None or i in tcam_blocks_0idx)):
                block_tcam_type = tcam_type
                n_tcam_modules += 1
            else:
                block_tcam_type = None

            common_kwargs = dict(
                r_sem=r_sem, r_str=r_str, n_non_spatial=n_non_spatial,
                lora_mode=lora_mode, structural_path_type=base_structural_path,
                use_rslora=use_rslora, tcam_type=block_tcam_type,
                tcam_gamma=tcam_gamma, tcam_hidden_min=tcam_hidden_min,
            )

            if "qkv" in target_modules:
                block.attn.qkv = SSDLoRAModule(block.attn.qkv, **common_kwargs)
            if "fc1" in target_modules:
                block.mlp.fc1 = SSDLoRAModule(block.mlp.fc1, **common_kwargs)
            if "fc2" in target_modules:
                block.mlp.fc2 = SSDLoRAModule(block.mlp.fc2, **common_kwargs)

        # Print TCAM summary for traceability
        if tcam_type is not None:
            if tcam_blocks_0idx is not None and len(tcam_blocks_0idx) > 1:
                tcam_blocks_print = f"[{tcam_blocks_0idx[0]+1}..{tcam_blocks_0idx[-1]+1}]"
            elif tcam_blocks_0idx is not None:
                tcam_blocks_print = f"{[b+1 for b in tcam_blocks_0idx]}"
            else:
                tcam_blocks_print = "all"
            print(
                f"TCAM: type={tcam_type}, blocks_1idx={tcam_blocks_print}, "
                f"base_path={base_structural_path}, gamma={tcam_gamma}, "
                f"hidden_min={tcam_hidden_min}, "
                f"tcam_modules={n_tcam_modules*len(target_modules)} "
                f"({n_tcam_modules} blocks x {len(target_modules)} targets)"
            )

        # Initialize alpha values if specified
        if alpha_init != 1.0:
            for block in self.backbone.blocks:
                for attr in ["attn.qkv", "mlp.fc1", "mlp.fc2"]:
                    module = attr.split(".")
                    mod = block
                    for m in module:
                        mod = getattr(mod, m)
                    if isinstance(mod, SSDLoRAModule):
                        if mod.r_sem > 0:
                            mod.alpha_sem.data.fill_(alpha_init)
                        if mod.r_str > 0:
                            mod.alpha_str.data.fill_(alpha_init)

        trainable = count_trainable_parameters(self)
        if self._lora_blocks_0idx is not None:
            n_lora = len(self._lora_blocks_0idx)
            blocks_str = f"blocks {self._lora_blocks_0idx} ({n_lora}/{len(self.backbone.blocks)})"
        else:
            blocks_str = f"all {len(self.backbone.blocks)} blocks"
        rslora_str = " (rsLoRA: alpha/sqrt(r))" if use_rslora else ""
        print(
            f"SSD-LoRA (mode={lora_mode}) applied to {blocks_str}, "
            f"targets={target_modules}, alpha_init={alpha_init}{rslora_str}, "
            f"trainable params={trainable:,}"
        )

    # ---- forward ----

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, dict]:
        B, C, H, W = x.shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        features_raw = self.backbone.get_intermediate_layers(
            x, n=self.layers_to_extract_0idx, reshape=False, norm=True
        )

        feature_maps = []
        for feat in features_raw:
            # feat: (B, N_patches, C) — CLS and storage tokens already stripped
            feat = feat.reshape(B, h_patches, w_patches, -1).permute(0, 3, 1, 2).contiguous()
            feature_maps.append(feat)

        out = self.decoder(feature_maps, img_size=(H, W))
        return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model(cfg) -> HALoRASeg:
    """Factory function: create HA-LoRA model from config dict."""
    return HALoRASeg(cfg)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
