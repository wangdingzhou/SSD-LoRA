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
from m2f_modules import MSDeformAttnPixelDecoder


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


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style residual block: DWConv7x7 + PW expand 4x + GELU + PW reduce + LayerScale + residual.

    Used by SpatialPriorModule to refine the from-scratch spatial prior.
    LayerScale init=1e-6 keeps the block near-identity at start so the SPM
    starts by behaving like a plain stem.
    """

    def __init__(self, dim: int, layer_scale_init: float = 1e-6, dropout: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(num_groups=min(32, dim), num_channels=dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), layer_scale_init)) if layer_scale_init > 0 else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)          # (B, H, W, C) for Linear
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)          # back to (B, C, H, W)
        x = self.drop(x)
        return identity + x


class SpatialPriorModule(nn.Module):
    """From-scratch CNN stem that extracts a spatial prior S8 from input RGB.

    Output is at H/8 resolution with `out_channels` channels. Designed to be
    used as the K/V source for SpatialPriorInjector (cross-attention into
    frozen DINOv3 H/16 features).

    Architecture:
        stem: 3 x (Conv3x3 stride2 + GroupNorm + GELU), 3 -> 32 -> 64 -> out_channels
        refine: n_blocks x ConvNeXtBlock(out_channels)

    At 512x512 input, output is (B, out_channels, 64, 64).
    """

    def __init__(self, out_channels: int = 128, n_blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        # 3 stride-2 convs: 512 -> 256 -> 128 -> 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=64),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ConvNeXtBlock(out_channels, dropout=dropout) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalized RGB (ImageNet stats).
        Returns:
            S8: (B, out_channels, H/8, W/8).
        """
        return self.blocks(self.stem(x))


class SpatialPriorInjector(nn.Module):
    """Cross-attention injector: enrich frozen DINOv3 H/16 features with RGB spatial prior.

    Q = projected frozen feature map (B, H/16*W/16, d_bottleneck)
    K,V = projected S8 spatial prior (B, H/8*W/8, d_bottleneck)
    Output = feature_map + gamma * project_back(MHA(Q, K, V))

    gamma is a learnable scalar initialized to 0, so the module is exact
    identity at init (model starts at R0 behavior). This is critical for
    not breaking the frozen-backbone warm start.

    Used in HALoRASeg.forward to inject S8 into selected backbone feature
    maps before they enter the decoder.
    """

    def __init__(
        self,
        d_frozen: int = 1024,           # ViT-L embed_dim
        d_spm: int = 128,               # SpatialPriorModule out_channels
        d_bottleneck: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_frozen = d_frozen
        self.d_bottleneck = d_bottleneck

        # Q/K/V projections into bottleneck space
        self.q_proj = nn.Linear(d_frozen, d_bottleneck)
        self.k_proj = nn.Linear(d_spm, d_bottleneck)
        self.v_proj = nn.Linear(d_spm, d_bottleneck)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_bottleneck,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(d_bottleneck)
        self.norm_out = nn.LayerNorm(d_bottleneck)
        self.out_proj = nn.Linear(d_bottleneck, d_frozen)

        # gamma=0 init => identity at start
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        feature_map: torch.Tensor,
        s8: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feature_map: (B, d_frozen, H/16, W/16) frozen DINOv3 feature.
            s8:          (B, d_spm,    H/8,  W/8)  spatial prior from SPM.
        Returns:
            (B, d_frozen, H/16, W/16) enriched feature (identity at init).
        """
        B, C, Hf, Wf = feature_map.shape
        # Flatten spatial dims: (B, H*W, C)
        q = feature_map.flatten(2).transpose(1, 2)            # (B, Hf*Wf, d_frozen)
        kv = s8.flatten(2).transpose(1, 2)                    # (B, H8*W8, d_spm)

        q = self.q_proj(q)                                    # (B, Nq, d_bottleneck)
        k = self.k_proj(kv)                                   # (B, Nk, d_bottleneck)
        v = self.v_proj(kv)                                   # (B, Nk, d_bottleneck)
        q = self.norm_q(q)

        attn_out, _ = self.attn(q, k, v, need_weights=False)  # (B, Nq, d_bottleneck)
        attn_out = self.norm_out(attn_out)
        out = self.out_proj(attn_out)                         # (B, Nq, d_frozen)

        # Reshape back to spatial: (B, d_frozen, Hf, Wf)
        out = out.transpose(1, 2).reshape(B, C, Hf, Wf)

        return feature_map + self.gamma * out


# ---------------------------------------------------------------------------
# SC-CMRD-LAR/v1.2 components
# ---------------------------------------------------------------------------

def _shift_2d(v: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Shift a (B, C, H, W) tensor by (dy, dx) with replicate padding.

    Sign convention: positive dy shifts content down (output[y] = input[y-dy]).
    Used by MultiBasisLocalAttenderUpsample to sample V at offset positions.
    """
    if dy == 0 and dx == 0:
        return v
    H, W = v.shape[-2:]
    pad_top = max(dy, 0)
    pad_bot = max(-dy, 0)
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    v_padded = torch.nn.functional.pad(
        v, (pad_left, pad_right, pad_top, pad_bot), mode="replicate"
    )
    y0 = max(-dy, 0)
    x0 = max(-dx, 0)
    return v_padded[..., y0:y0 + H, x0:x0 + W]


# Fixed basis offsets in LOW-RES pixel units (applied to V_low via shift*factor).
_BASIS_SQUARE3 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]
_BASIS_AXIAL_STRIP5 = [
    (0, -2), (0, -1), (0, 0), (0, 1), (0, 2),
    (-2, 0), (-1, 0), (1, 0), (2, 0),
]
_BASIS_RING8 = [
    (-2, -2), (-2, 0), (-2, 2),
    (0, -2), (0, 2),
    (2, -2), (2, 0), (2, 2),
]


class _OffsetBasis(nn.Module):
    """Holds a fixed set of (dy, dx) offsets for one basis.

    Stores offsets in low-res units and exposes them scaled to high-res
    via upsample_factor. The anchor positional code is a normalized
    meshgrid in [-1, 1]; per-offset distinction is delegated to the
    learned K_b-dim conv output.
    """

    def __init__(self, offsets: torch.Tensor, upsample_factor: int = 2):
        super().__init__()
        self.register_buffer("offsets_lowres", offsets)
        self.upsample_factor = upsample_factor

    @property
    def offsets_highres(self) -> list:
        return [
            (int(dy * self.upsample_factor), int(dx * self.upsample_factor))
            for dy, dx in self.offsets_lowres.tolist()
        ]

    def anchor_pos_code(self, H: int, W: int, device, dtype) -> torch.Tensor:
        """Return (1, 2, H, W) positional code: normalized meshgrid in [-1, 1]."""
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([grid_y, grid_x], dim=0).unsqueeze(0)


class MultiBasisLocalAttenderUpsample(nn.Module):
    """Multi-basis local attender upsample (SC-CMRD-LAR/v1.2 section 1).

    For each target (high-res) pixel p:
      1. router(pi_b) = softmax_b over bases, conditioned on [query, guide]
      2. per basis: attn_{b,k} = softmax_k over offsets, conditioned on
         [query, guide, positional_code]
      3. U(p) = sum_b pi_b(p) * sum_k attn_{b,k}(p) * V_low_upsampled(p + 2*offset)

    V_low is bilinearly upsampled to high-res first, then shifts are applied
    at high-res using offset*upsample_factor. This keeps all attention weight
    computation at high resolution while ensuring the span of the output is
    contained in span(V_low_upsampled) (bilinear is linear, preserves span).

    Args:
        in_channels: V_low channel dim (e.g., 384 after projection).
        query_channels: Query_high channel dim (e.g., 384).
        guide_channels: Guide_high (RGB structural prior) channel dim (e.g., 192).
        out_channels: output channel dim (typically = in_channels for cascade).
        upsample_factor: typically 2 (e.g., H/16 -> H/8).
        hidden_dim: internal projection dim for router/attn convs (default 256).
    """

    def __init__(
        self,
        in_channels: int,
        query_channels: int,
        guide_channels: int,
        out_channels: int,
        upsample_factor: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_factor = upsample_factor

        bases = {
            "square3": _BASIS_SQUARE3,
            "axial_strip5": _BASIS_AXIAL_STRIP5,
            "ring8": _BASIS_RING8,
        }
        self.basis_names = list(bases.keys())
        self.n_bases = len(self.basis_names)
        self.basis_offsets = nn.ModuleList([
            _OffsetBasis(torch.tensor(bases[name], dtype=torch.float32), upsample_factor)
            for name in self.basis_names
        ])

        # Value projection: V_low -> hidden_dim
        self.v_proj = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)

        # Router over bases
        self.router_conv = nn.Conv2d(query_channels + guide_channels, self.n_bases, 1)

        # Per-basis attention over offsets
        self.attn_convs = nn.ModuleList([
            nn.Conv2d(query_channels + guide_channels + 2, len(bases[name]), 1)
            for name in self.basis_names
        ])

        # Output projection: hidden_dim -> out_channels. Zero-init for clean
        # identity-at-start when composed with ParallelSemanticStructuralFuse
        # at lambda=0.
        self.out_proj = nn.Conv2d(hidden_dim, out_channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        v_low: torch.Tensor,
        query_high: torch.Tensor,
        guide_high: torch.Tensor,
    ):
        """
        Args:
            v_low: (B, in_channels, H_low, W_low) e.g., (B, 384, 32, 32).
            query_high: (B, query_channels, H_high, W_high) e.g., (B, 384, 64, 64).
            guide_high: (B, guide_channels, H_high, W_high) e.g., (B, 192, 64, 64).
        Returns:
            U: (B, out_channels, H_high, W_high).
            diag: dict with basis_probs (B, n_bases, H_high, W_high).
        """
        B, _, H_high, W_high = query_high.shape

        v_proj = self.v_proj(v_low)
        v_high = torch.nn.functional.interpolate(
            v_proj, size=(H_high, W_high), mode="bilinear", align_corners=False
        )

        router_logits = self.router_conv(
            torch.cat([query_high, guide_high], dim=1)
        )
        basis_probs = torch.softmax(router_logits, dim=1)

        attended = torch.zeros(
            B, self.v_proj.out_channels, H_high, W_high,
            device=v_low.device, dtype=v_low.dtype,
        )
        for b_idx, (offset_basis, attn_conv) in enumerate(
            zip(self.basis_offsets, self.attn_convs)
        ):
            pos_code = offset_basis.anchor_pos_code(
                H_high, W_high, device=v_low.device, dtype=v_low.dtype
            )
            attn_logits = attn_conv(
                torch.cat([
                    query_high, guide_high,
                    pos_code.expand(B, -1, -1, -1),
                ], dim=1)
            )
            attn_weights = torch.softmax(attn_logits, dim=1)

            K_b = attn_weights.shape[1]
            attended_b = torch.zeros_like(attended)
            for k_idx in range(K_b):
                dy_high, dx_high = offset_basis.offsets_highres[k_idx]
                v_shifted = _shift_2d(v_high, dy_high, dx_high)
                attended_b = attended_b + attn_weights[:, k_idx:k_idx + 1] * v_shifted

            attended = attended + basis_probs[:, b_idx:b_idx + 1] * attended_b

        U = self.out_proj(attended)
        diag = {"basis_probs": basis_probs.detach()}
        return U, diag


class HREvidenceEncoder(nn.Module):
    """Multi-resolution RGB structural prior encoder (SC-CMRD-LAR/v1.2).

    Produces {S8 @ H/8, S4 @ H/4} from RGB. Each scale has its own small CNN
    stem followed by 1 ConvNeXtBlock for refinement.

    Args:
        out_channels: channel dim of each output scale (default 192).
        n_blocks: ConvNeXtBlocks per scale after stem (default 1).
    """

    def __init__(self, out_channels: int = 192, n_blocks: int = 1):
        super().__init__()
        self.s8_stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 64), nn.GELU(),
            nn.Conv2d(64, out_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels), nn.GELU(),
        )
        self.s8_refine = nn.Sequential(
            *[ConvNeXtBlock(out_channels) for _ in range(n_blocks)]
        )

        self.s4_stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, out_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels), nn.GELU(),
        )
        self.s4_refine = nn.Sequential(
            *[ConvNeXtBlock(out_channels) for _ in range(n_blocks)]
        )

    def forward(self, rgb: torch.Tensor):
        """
        Args:
            rgb: (B, 3, H, W) ImageNet-normalized.
        Returns:
            S8: (B, out_channels, H/8, W/8)
            S4: (B, out_channels, H/4, W/4)
        """
        S8 = self.s8_refine(self.s8_stem(rgb))
        S4 = self.s4_refine(self.s4_stem(rgb))
        return S8, S4


def _avg_pool_2d(x: torch.Tensor, k: int) -> torch.Tensor:
    """AvgPool2d with 'same' padding (k//2 on each side)."""
    return torch.nn.functional.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)


class SpatialReliabilityGate(nn.Module):
    """Spatial reliability gate (SC-CMRD-LAR/v1.2 section 3).

    Inputs are 7 channels: [sem, str, entropy, margin, local_unc,
    boundary_density, low_margin_density]. Output is a (B, 1, H, W) gate
    in (0, 1).

    Gate conv weight zero-init, bias init determines start gate (sigmoid(bias)).
    Default bias=0 gives mean gate ~0.5 at start.

    Args:
        sem_channels: semantic stream channels.
        str_channels: structural stream channels.
        init_bias: gate conv bias init (default 0.0).
    """

    def __init__(self, sem_channels: int, str_channels: int, init_bias: float = 0.0):
        super().__init__()
        in_dim = sem_channels + str_channels + 5
        self.gate_conv = nn.Conv2d(in_dim, 1, 1)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.constant_(self.gate_conv.bias, init_bias)

    def forward(
        self,
        sem: torch.Tensor,
        str_feat: torch.Tensor,
        entropy: torch.Tensor,
        margin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sem: (B, C_sem, H, W).
            str_feat: (B, C_str, H, W).
            entropy: (B, 1, H, W) softmax entropy H(P).
            margin: (B, 1, H, W) top-1 minus top-2 probability.
        Returns:
            gate: (B, 1, H, W) in (0, 1).
        """
        local_unc = _avg_pool_2d(entropy, k=5)
        boundary_density = _avg_pool_2d(entropy * (margin < 1e-3).float(), k=5)
        low_margin_density = _avg_pool_2d(1.0 - margin, k=5)

        gate_input = torch.cat([
            sem, str_feat, entropy, margin,
            local_unc, boundary_density, low_margin_density,
        ], dim=1)
        gate = torch.sigmoid(self.gate_conv(gate_input))
        return gate


class ParallelSemanticStructuralFuse(nn.Module):
    """Parallel semantic-structural fuse with bounded residual.

    Pipeline:
      sem = sem_proj(U)                       # semantic stream
      str = str_local_mix(str_proj(S))        # structural stream (refined)
      coeff = softmax(Conv([sem, str, ent, margin, local_unc])) over {sem, str}
      gate = SpatialReliabilityGate(...)
      out = U + lambda * gate * (coeff_sem*sem + coeff_str*str - U)

    Mathematical guarantee: gate in [0,1], coeff_sem + coeff_str = 1 (softmax),
    lambda >= 0 => output in bounded neighborhood of U.
    At lambda=0 init => out = U exactly (identity).

    Args:
        channels: shared channel dim for sem/str/U (e.g., 384).
        guide_channels: structural input channels (e.g., 192).
        lambda_init: residual scale init (default 0.0 for identity-at-start).
    """

    def __init__(
        self,
        channels: int,
        guide_channels: int,
        lambda_init: float = 0.0,
    ):
        super().__init__()
        self.channels = channels

        self.sem_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
            nn.GELU(),
        )
        self.str_proj = nn.Sequential(
            nn.Conv2d(guide_channels, channels, 1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
            nn.GELU(),
        )
        self.str_local_mix = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.GroupNorm(min(32, channels), channels),
            nn.GELU(),
        )

        # Coefficient mapper: input [sem, str, ent, margin, local_unc] -> {sem, str}
        self.coeff_conv = nn.Conv2d(channels * 2 + 3, 2, 1)
        nn.init.zeros_(self.coeff_conv.weight)
        nn.init.zeros_(self.coeff_conv.bias)  # softmax([0,0]) = [0.5, 0.5]

        self.gate = SpatialReliabilityGate(
            sem_channels=channels, str_channels=channels, init_bias=0.0
        )

        # Learnable residual scale, init=0 for identity at start
        self.lam = nn.Parameter(torch.tensor(lambda_init))

    def forward(
        self,
        U: torch.Tensor,
        S: torch.Tensor,
        entropy: torch.Tensor,
        margin: torch.Tensor,
    ):
        """
        Args:
            U: (B, C, H, W) semantic reconstruction (from MultiBasisLAR).
            S: (B, Cg, H, W) structural prior.
            entropy: (B, 1, H, W).
            margin: (B, 1, H, W).
        Returns:
            out: (B, C, H, W) bounded residual fusion.
            diag: dict with gate_mean, coeff_sem_mean, coeff_str_mean, lambda.
        """
        sem = self.sem_proj(U)
        str_raw = self.str_proj(S)
        str_feat = self.str_local_mix(str_raw)

        local_unc = _avg_pool_2d(entropy, k=5)
        coeff_input = torch.cat([sem, str_feat, entropy, margin, local_unc], dim=1)
        coeff = torch.softmax(self.coeff_conv(coeff_input), dim=1)
        coeff_sem = coeff[:, 0:1]
        coeff_str = coeff[:, 1:2]

        gate = self.gate(sem, str_feat, entropy, margin)

        mixed = coeff_sem * sem + coeff_str * str_feat
        out = U + self.lam * gate * (mixed - U)

        diag = {
            "gate_mean": gate.mean().detach(),
            "coeff_sem_mean": coeff_sem.mean().detach(),
            "coeff_str_mean": coeff_str.mean().detach(),
            "lambda": self.lam.detach(),
        }
        return out, diag


class SCHRRBlock(nn.Module):
    """Semantic-Constrained High-Resolution Reconstruction block.

    Composes MultiBasisLocalAttenderUpsample + ParallelSemanticStructuralFuse.
    Takes a low-res semantic feature V_low and high-res RGB guide S_high,
    produces a high-res output X_high. Includes a native-scale aux head on
    the LAR output U for uncertainty features (entropy, margin) and aux
    supervision.

    Args:
        channels: shared channel dim (e.g., 384).
        guide_channels: RGB structural prior channels at this scale (e.g., 192).
        upsample_factor: typically 2.
        num_classes: for aux head + uncertainty features.
        aux_head: if True, include 1x1 head for aux logits at this scale.
    """

    def __init__(
        self,
        channels: int,
        guide_channels: int,
        upsample_factor: int = 2,
        num_classes: int = 6,
        aux_head: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.aux_head = aux_head

        self.q_proj = nn.Conv2d(guide_channels, channels, 1, bias=False)

        self.lar = MultiBasisLocalAttenderUpsample(
            in_channels=channels,
            query_channels=channels,
            guide_channels=guide_channels,
            out_channels=channels,
            upsample_factor=upsample_factor,
        )

        self.fuse = ParallelSemanticStructuralFuse(
            channels=channels,
            guide_channels=guide_channels,
            lambda_init=0.0,
        )

        if aux_head:
            self.aux_head_conv = nn.Conv2d(channels, num_classes, 1)

    def forward(
        self,
        v_low: torch.Tensor,
        guide_high: torch.Tensor,
    ):
        """
        Args:
            v_low: (B, C, H_low, W_low).
            guide_high: (B, Cg, H_high, W_high).
        Returns:
            X_high: (B, C, H_high, W_high).
            aux_logits: (B, num_classes, H_high, W_high) or None.
            entropy: (B, 1, H_high, W_high) — softmax entropy of aux logits.
            diag: merged dict from sub-modules.
        """
        B, _, H_high, W_high = guide_high.shape
        query_high = self.q_proj(guide_high)

        U, lar_diag = self.lar(v_low, query_high, guide_high)

        if self.aux_head:
            aux_logits = self.aux_head_conv(U)
            probs = torch.softmax(aux_logits, dim=1)
            log_probs = torch.log_softmax(aux_logits, dim=1)
            entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
            top2 = torch.topk(probs, k=2, dim=1).values
            margin = (top2[:, 0:1] - top2[:, 1:2]).clamp(min=0.0)
        else:
            aux_logits = None
            entropy = torch.zeros(B, 1, H_high, W_high, device=v_low.device, dtype=v_low.dtype)
            margin = torch.ones(B, 1, H_high, W_high, device=v_low.device, dtype=v_low.dtype)

        X_high, fuse_diag = self.fuse(U, guide_high, entropy, margin)

        diag = {**lar_diag, **fuse_diag}
        return X_high, aux_logits, entropy, diag


class AffinityPreservationLoss(nn.Module):
    """Sampled affinity preservation loss (SC-CMRD-LAR/v1.2 section 4).

    Computes self-similarity (affinity) matrices for both the model's
    intermediate features (X8) and the frozen anchor (F16), then minimizes
    their L1 distance. This prevents HR RGB evidence from drifting the
    DINO semantic topology.

    Affinity is computed at a fixed S*S spatial anchor grid via adaptive
    avg pool. Channel-dim is L2-normalized before affinity. Train-only,
    no inference cost.

    Args:
        n_anchor_tokens: total anchor tokens, must be a perfect square (e.g., 196=14*14).
        tau: affinity softmax temperature (default 0.2).
    """

    def __init__(self, n_anchor_tokens: int = 196, tau: float = 0.2):
        super().__init__()
        self.n_anchor_tokens = n_anchor_tokens
        side = int(n_anchor_tokens ** 0.5)
        assert side * side == n_anchor_tokens, (
            f"n_anchor_tokens must be perfect square, got {n_anchor_tokens}"
        )
        self.anchor_side = side
        self.tau = tau

    def _affinity(self, x: torch.Tensor) -> torch.Tensor:
        """Compute affinity matrix from (B, C, H, W) feature map.

        Returns (B, N, N) where N = anchor_side**2.
        """
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            x, output_size=(self.anchor_side, self.anchor_side)
        )
        tokens = pooled.flatten(2).transpose(1, 2)             # (B, N, C)
        tokens = torch.nn.functional.normalize(tokens, dim=-1)
        aff = torch.matmul(tokens, tokens.transpose(1, 2)) / self.tau
        aff = torch.softmax(aff, dim=-1)
        return aff

    def forward(
        self,
        x_model: torch.Tensor,
        x_anchor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_model: (B, C_m, H, W) model feature (NOT detached).
            x_anchor: (B, C_a, H, W) anchor (caller should pass detached).
        Returns:
            loss: scalar L1 distance between affinity matrices.
        """
        A_model = self._affinity(x_model)
        with torch.no_grad():
            A_anchor = self._affinity(x_anchor)
        return torch.nn.functional.l1_loss(A_model, A_anchor)


class SCCMRDLarDecoder(nn.Module):
    """SC-CMRD-LAR/v1.2-refined decoder (unified SCHRR).

    Pipeline:
      1. Project 6 frozen DINO layers to embed_dim and sum -> F16_proj
      2. HR encoder: RGB -> {S8 @ H/8, S4 @ H/4}
      3. SCHRR_8: F16_proj + S8 -> X8 @ H/8 (aux logits8)
      4. SCHRR_4: X8 + S4 -> X4 @ H/4 (aux logits4)
      5. Refine X4: n x ConvNeXtBlock
      6. Head: refine(X4) -> logits4 -> bilinear -> logits at full res
      7. Native aux head at F16_proj: logits16

    Args:
        in_channels: frozen DINO feature dim (1024 for ViT-L).
        embed_dim: working channel dim (default 384).
        guide_channels: HR encoder output channels per scale (default 192).
        num_classes: 6 for Potsdam, 7 for LoveDA.
        hr_n_blocks: ConvNeXt blocks in HR encoder per scale (default 1).
        refine_n_blocks: ConvNeXt blocks after SCHRR_4 (default 2).
        n_layers: number of DINO layers to fuse (default 6).
    """

    def __init__(
        self,
        in_channels: int = 1024,
        embed_dim: int = 384,
        guide_channels: int = 192,
        num_classes: int = 6,
        hr_n_blocks: int = 1,
        refine_n_blocks: int = 2,
        n_layers: int = 6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.layer_projs = nn.ModuleList([
            nn.Conv2d(in_channels, embed_dim, 1, bias=False)
            for _ in range(n_layers)
        ])

        self.hr_encoder = HREvidenceEncoder(
            out_channels=guide_channels, n_blocks=hr_n_blocks
        )

        self.schrr_8 = SCHRRBlock(
            channels=embed_dim, guide_channels=guide_channels,
            upsample_factor=2, num_classes=num_classes, aux_head=True,
        )
        self.schrr_4 = SCHRRBlock(
            channels=embed_dim, guide_channels=guide_channels,
            upsample_factor=2, num_classes=num_classes, aux_head=True,
        )

        self.refine = nn.Sequential(
            *[ConvNeXtBlock(embed_dim) for _ in range(refine_n_blocks)]
        )

        self.head = nn.Conv2d(embed_dim, num_classes, 1)

        self.aux_head_16 = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(
        self,
        features: list,
        img_size: tuple,
        rgb: torch.Tensor = None,
    ):
        """
        Args:
            features: list of n_layers tensors, each (B, in_channels, H/16, W/16).
            img_size: (H, W) target output size.
            rgb: (B, 3, H, W) original RGB. REQUIRED.
        Returns:
            dict with: logits, aux_logits8, aux_logits4, aux_logits16,
                       affinity_model, affinity_anchor, diag.
        """
        if rgb is None:
            raise ValueError("SCCMRDLarDecoder requires rgb input")
        assert len(features) > 0
        B, _, H16, W16 = features[0].shape
        H, W = img_size

        f16_proj = sum(
            proj(feat) for proj, feat in zip(self.layer_projs, features)
        )
        f16_anchor = features[-1].detach()

        S8, S4 = self.hr_encoder(rgb)

        X8, aux8, ent8, diag8 = self.schrr_8(f16_proj, S8)
        X4, aux4, ent4, diag4 = self.schrr_4(X8, S4)

        X4_refined = self.refine(X4)
        logits4 = self.head(X4_refined)
        logits = torch.nn.functional.interpolate(
            logits4, size=(H, W), mode="bilinear", align_corners=False
        )

        aux16 = self.aux_head_16(f16_proj)

        diag = {
            "schrr_8": diag8,
            "schrr_4": diag4,
            "f16_proj_mean": f16_proj.mean().detach(),
            "f16_proj_std": f16_proj.std().detach(),
        }

        return {
            "logits": logits,
            "aux_logits8": aux8,
            "aux_logits4": aux4,
            "aux_logits16": aux16,
            "affinity_model": X8,
            "affinity_anchor": f16_anchor,
            "diag": diag,
        }


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

class MLPDecoder(nn.Module):
    """Lightweight MLP decoder for segmentation."""

    def __init__(self, in_channels_list: List[int], embed_dim: int = 256,
                 num_classes: int = 7, feat_plugin: Optional[nn.Module] = None):
        super().__init__()
        self.fusion = FeatureFusionModule(in_channels_list, embed_dim)
        fused_dim = embed_dim * len(in_channels_list)
        self.fused_dim = fused_dim
        self.feat_plugin = feat_plugin
        self.head = nn.Sequential(
            nn.Conv2d(fused_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, num_classes, 1),
        )

    def forward(self, features: List[torch.Tensor], img_size: Tuple[int, int]) -> torch.Tensor:
        fused = self.fusion(features)
        if self.feat_plugin is not None:
            fused = self.feat_plugin(fused)
        out = F.interpolate(fused, size=img_size, mode="bilinear", align_corners=False)
        return self.head(out)


class ShallowFeatureGate(nn.Module):
    """Adaptive gating for shallow/mid-level feature injection.

    Initialized with closed gate (bias=-3 → sigmoid≈0.047) so training starts
    from the deep-only path and gradually learns to open injection. Pure
    identity init (bias→-inf) would freeze gradients; bias=-3 keeps a small
    non-zero signal so the gate can learn to open.
    """

    def __init__(self, dim=256, init_gate_bias: float = -3.0):
        super().__init__()
        self.gate_conv = nn.Conv2d(dim * 2, 1, kernel_size=1)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.constant_(self.gate_conv.bias, init_gate_bias)

    def forward(self, x_deep, x_shallow):
        gate = torch.sigmoid(
            self.gate_conv(torch.cat([x_deep, x_shallow], dim=1))
        )
        return x_deep + gate * x_shallow


class DWPWBlock(nn.Module):
    """DW-Sep Conv + PixelShuffle 2x upsampling."""

    def __init__(self, in_ch, out_ch, upscale=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch * (upscale ** 2), 1, bias=False),
            nn.PixelShuffle(upscale_factor=upscale),
        )

    def forward(self, x):
        return self.conv(x)


class PFUDecoder(nn.Module):
    """PFU Decoder v3 — Progressive Fusion Upsampler.

    Progressive 2x upsampling via PixelShuffle with adaptive gating
    for shallow/mid-level feature injection at intermediate resolutions.
    """

    def __init__(self, in_channels_list, embed_dim=256, num_classes=7, **kwargs):
        super().__init__()
        in_dim = in_channels_list[0]

        self.proj_6 = nn.Sequential(
            nn.Conv2d(in_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        self.proj_12 = nn.Sequential(
            nn.Conv2d(in_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        self.proj_18 = nn.Sequential(
            nn.Conv2d(in_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        self.proj_24 = nn.Sequential(
            nn.Conv2d(in_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))

        self.gate_6 = ShallowFeatureGate(embed_dim)
        self.gate_12 = ShallowFeatureGate(embed_dim)

        self.up_32_64 = DWPWBlock(embed_dim, embed_dim, upscale=2)
        self.up_64_128 = DWPWBlock(embed_dim, embed_dim, upscale=2)
        self.up_128_256 = DWPWBlock(embed_dim, embed_dim, upscale=2)
        self.up_256_512 = DWPWBlock(embed_dim, embed_dim, upscale=2)

        self.seg_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features, img_size):
        f6 = self.proj_6(features[0])
        f12 = self.proj_12(features[1])
        f18 = self.proj_18(features[2])
        f24 = self.proj_24(features[3])

        x = f24 + f18

        x = self.up_32_64(x)
        f6_up = F.interpolate(f6, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = self.gate_6(x, f6_up)

        x = self.up_64_128(x)
        f12_up = F.interpolate(f12, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = self.gate_12(x, f12_up)

        x = self.up_128_256(x)
        x = self.up_256_512(x)

        out = self.seg_head(x)
        if out.shape[2:] != img_size:
            out = F.interpolate(out, size=img_size, mode='bilinear', align_corners=False)
        return out


class SRUDecoder(nn.Module):
    """Spatial Refinement Upsampler — MLP cross-layer fusion + progressive
    PixelShuffle upsampling + shallow feature gating.

    Combines FeatureFusionModule (semantic fusion at H/16) with PFU-style
    progressive upsampling and ShallowFeatureGate injection. Complementary
    to SSD-LoRA's structural path which operates inside ViT blocks at H/16.
    """

    def __init__(self, in_channels_list, embed_dim=256, num_classes=7,
                 n_deep_layers=4, **kwargs):
        super().__init__()
        n_total = len(in_channels_list)
        n_shallow = n_total - n_deep_layers
        deep_channels = in_channels_list[n_shallow:]
        shallow_channels = in_channels_list[:n_shallow]

        self.fusion = FeatureFusionModule(deep_channels, embed_dim)
        fused_dim = embed_dim * n_deep_layers

        self.reduce = nn.Sequential(
            nn.Conv2d(fused_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        self.shallow_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
            )
            for c in shallow_channels
        ])

        self.gates = nn.ModuleList([
            ShallowFeatureGate(embed_dim) for _ in shallow_channels
        ])

        self.ups = nn.ModuleList([
            DWPWBlock(embed_dim, embed_dim, upscale=2)
            for _ in range(3)
        ])

        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features, img_size):
        n_shallow = len(self.shallow_projs)
        deep_features = features[n_shallow:]
        shallow_features = features[:n_shallow]

        shallow_projed = [
            proj(f) for proj, f in zip(self.shallow_projs, shallow_features)
        ]

        fused = self.fusion(deep_features)
        x = self.reduce(fused)

        for i, up in enumerate(self.ups):
            x = up(x)
            if i < len(shallow_projed):
                f_shallow = F.interpolate(
                    shallow_projed[i], size=x.shape[2:],
                    mode='bilinear', align_corners=False,
                )
                x = self.gates[i](x, f_shallow)

        if x.shape[2:] != img_size:
            x = F.interpolate(x, size=img_size, mode='bilinear', align_corners=False)

        return self.head(x)


class SRUPlugin(nn.Module):
    """Spatial Refinement Upsampler — as a PLUGIN on MLP decoder logits.

    The core problem this targets:
        The lightweight MLP decoder fuses deep ViT layers (e.g. blocks 9, 12,
        18, 24) into a feature map at H/16, then bilinearly upsamples to H.
        Shallow ViT layers (blocks 3, 6) carry fine-grained spatial cues that
        are completely lost in this path. PFU decoder beats MLP on Potsdam
        (+1.32%) largely BECAUSE it re-injects these shallow features via
        ShallowFeatureGate at intermediate resolutions.

    SRU plugin's design:
        Take MLP output logits (B, C, H, W) at full resolution and refine them
        by GATED INJECTION of shallow ViT features (already extracted as part
        of HALoRASeg's layers_to_extract). Identity-init: out_proj weights = 0
        so the plugin starts as a no-op and gradually learns to inject.

    Difference from PFEB:
        PFEB:   pure logit refinement — DWConv + ECA, no new information.
        SRU:    INJECTS new information — shallow ViT features → logit space.

    Args:
        num_classes: segmentation classes (logit channels)
        shallow_channels: list of channel dims for the shallow ViT feature(s)
                          to inject (e.g. [1024, 1024] for ViT-L blocks 3,6)
        embed_dim: working channel dim inside the plugin (default 128)
    """

    def __init__(
        self,
        num_classes: int,
        shallow_channels: Sequence[int],
        embed_dim: int = 128,
        init_gate_bias: float = -3.0,
    ):
        super().__init__()
        self.in_proj = nn.Conv2d(num_classes, embed_dim, kernel_size=1, bias=False)

        self.shallow_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim),
            )
            for c in shallow_channels
        ])
        self.gates = nn.ModuleList([
            ShallowFeatureGate(embed_dim, init_gate_bias=init_gate_bias)
            for _ in shallow_channels
        ])

        # Light spatial refinement (PFU-borrowed DW + PW idea, but at H, not progressive)
        self.refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1,
                      groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        self.out_proj = nn.Conv2d(embed_dim, num_classes, kernel_size=1, bias=False)
        # Identity init: starts as a no-op residual; learns to refine
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        logits: torch.Tensor,
        shallow_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes, H, W) — MLP decoder output
            shallow_features: list of (B, C_i, H_i, W_i) shallow ViT feature maps
        Returns:
            refined logits (B, num_classes, H, W)
        """
        assert len(shallow_features) == len(self.shallow_projs), (
            f"SRU plugin expects {len(self.shallow_projs)} shallow features, "
            f"got {len(shallow_features)}"
        )
        x = self.in_proj(logits)
        for proj, gate, f in zip(self.shallow_projs, self.gates, shallow_features):
            f_proj = proj(f)
            if f_proj.shape[2:] != x.shape[2:]:
                f_proj = F.interpolate(
                    f_proj, size=x.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            x = gate(x, f_proj)
        x = self.refine(x)
        return logits + self.out_proj(x)


class PFEBRefiner(nn.Module):
    """Post-Fusion Enhancement Block (Lian 2026 PFEB-style).

    A lightweight residual refiner placed AFTER the segmentation logits, so it
    works as a drop-in plugin for any decoder. Operates on the logits
    (B, num_classes, H, W) and refines class scores using DWConv (local
    structure) + ECA channel re-weighting (per-class importance) + residual.

    Params: ~ num_classes^2 + 9*num_classes + 2*num_classes ≈ 60 for 6 classes,
    plus the DW conv on a small intermediate channel. Total ~ a few hundred
    KB depending on intermediate dim.

    Args:
        num_classes: number of segmentation classes
        hidden_dim: intermediate channel dim for the refinement conv (default 64)
    """

    def __init__(self, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.in_proj = nn.Conv2d(num_classes, hidden_dim, kernel_size=1, bias=False)
        self.dw_conv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1,
            groups=hidden_dim, bias=False,
        )
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.GELU()
        # ECA: 1D adaptive conv on channel descriptor
        self.eca = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.out_proj = nn.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=False)
        # Initialize out_proj to zero so module starts as identity (logits unchanged)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(logits)
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.act(x)
        # ECA: global avg pool → 1D conv → sigmoid → channel-wise scale
        gap = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(1, 2)  # (B, 1, C)
        gate = torch.sigmoid(self.eca(gap)).transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)
        x = x * gate
        delta = self.out_proj(x)
        return logits + delta  # residual: starts as identity since out_proj=0

class FreqPlugin(nn.Module):
    """rFFT2-based frequency-domain refiner (logit-space plugin).

    Computes 2D real FFT of the logits, applies a learned low-frequency mask
    (radial gating in frequency space), inverse FFT, then a small conv,
    and adds as residual. Identity-init via out_conv zero weights.

    Captures global periodic structure (roof/road textures) that local LoRA
    + DWConv plugins cannot easily express.

    Args:
        num_classes: segmentation class count (logit channels)
        hidden_dim: intermediate channel dim for the post-iFFT conv (default 32)
        n_bands: number of learnable radial frequency bands (default 8)
    """

    def __init__(self, num_classes: int, hidden_dim: int = 32, n_bands: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.n_bands = n_bands
        # Learnable per-class per-band gate, init=1 (preserve all freqs initially)
        self.freq_gate = nn.Parameter(torch.ones(num_classes, n_bands))
        # Post-iFFT smoothing
        self.in_proj = nn.Conv2d(num_classes, hidden_dim, kernel_size=1, bias=False)
        self.dw_conv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1,
            groups=hidden_dim, bias=False,
        )
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.GELU()
        self.out_proj = nn.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=False)
        nn.init.zeros_(self.out_proj.weight)  # identity-init

    def _build_radial_mask(self, H: int, W: int, device, dtype) -> torch.Tensor:
        # rFFT2 output shape: (H, W//2+1). Build radial distance from DC corner.
        Wf = W // 2 + 1
        fy = torch.fft.fftfreq(H, device=device).abs()  # (H,)
        fx = torch.arange(Wf, device=device, dtype=fy.dtype) / W  # (Wf,)
        # Normalized radial distance in [0, sqrt(0.5)] approx
        rr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)  # (H, Wf)
        rr = rr / (rr.max() + 1e-6)  # normalize to [0, 1]
        # Assign each (h, w) to a band index in [0, n_bands)
        band_idx = (rr * self.n_bands).clamp(max=self.n_bands - 1).long()  # (H, Wf)
        # Gate: (C, n_bands) → expand to (C, H, Wf) via gather
        gate = self.freq_gate[:, band_idx]  # (C, H, Wf)
        return gate.to(dtype)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        # FFT needs float32 for numerical stability under AMP/fp16
        logits_f32 = logits.float()
        spec = torch.fft.rfft2(logits_f32, norm="ortho")
        gate = self._build_radial_mask(H, W, logits.device, spec.real.dtype)
        spec = spec * gate.unsqueeze(0)
        x_freq = torch.fft.irfft2(spec, s=(H, W), norm="ortho").to(logits.dtype)
        # Small conv refine
        x = self.in_proj(x_freq)
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.act(x)
        delta = self.out_proj(x)
        return logits + delta  # identity at init


# ---------------------------------------------------------------------------
# Feature-space plugins (inserted between FeatureFusion and conv head)
# ---------------------------------------------------------------------------

class FeatureSpacePlugin(nn.Module):
    """Base for plugins inserted into the MLP decoder after FeatureFusion.

    Input/Output: (B, in_channels, H/16, W/16). All subclasses are wrapped
    in reduce(1x1)->core->expand(1x1) so the expand 1x1 zero-init gives
    exact identity at init regardless of what the core does.
    """

    def __init__(self, in_channels: int, mid_channels: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        nn.init.zeros_(self.expand[0].weight)  # identity-init at the gate

    def core(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reduce(x)
        z = self.core(z)
        delta = self.expand(z)
        return x + delta


class MSCAFeatPlugin(FeatureSpacePlugin):
    """SegNeXt-style Multi-Scale Convolutional Attention in feature space.

    Local 3x3 DW + parallel strip DW (1xk + kx1) at multiple dilations,
    aggregated, then 1x1 to a sigmoid gate that re-weights z. Whole block
    is wrapped by zero-init expand → identity at init.
    """

    def __init__(self, in_channels: int, mid_channels: int = 256,
                 strip_kernels=(7, 11, 21)):
        super().__init__(in_channels, mid_channels)
        self.local_conv = nn.Conv2d(mid_channels, mid_channels, 3, padding=1,
                                    groups=mid_channels, bias=False)
        self.strip_convs = nn.ModuleList()
        for k in strip_kernels:
            pad = k // 2
            self.strip_convs.append(nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, (k, 1), padding=(pad, 0),
                          groups=mid_channels, bias=False),
                nn.Conv2d(mid_channels, mid_channels, (1, k), padding=(0, pad),
                          groups=mid_channels, bias=False),
            ))
        self.attn_gen = nn.Conv2d(mid_channels, mid_channels, 1, bias=False)

    def core(self, z: torch.Tensor) -> torch.Tensor:
        local = self.local_conv(z)
        strips = local
        for s in self.strip_convs:
            strips = strips + s(z)
        attn = torch.sigmoid(self.attn_gen(strips))
        return z * attn


class ASPPFeatPlugin(FeatureSpacePlugin):
    """ASPP-style multi-dilation 3x3 + global branch, projected back to mid.

    Identity at init guaranteed by zero-init expand of the parent.
    """

    def __init__(self, in_channels: int, mid_channels: int = 256,
                 branch_channels: int = 64, dilations=(1, 6, 12, 18)):
        super().__init__(in_channels, mid_channels)
        self.branches = nn.ModuleList()
        for d in dilations:
            if d == 1:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(mid_channels, branch_channels, 1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.GELU(),
                ))
            else:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(mid_channels, branch_channels, 3, padding=d,
                              dilation=d, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.GELU(),
                ))
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )
        fused = branch_channels * (len(dilations) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(fused, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )

    def core(self, z: torch.Tensor) -> torch.Tensor:
        size = z.shape[2:]
        feats = [b(z) for b in self.branches]
        g = self.global_branch(z)
        g = F.interpolate(g, size=size, mode="bilinear", align_corners=False)
        feats.append(g)
        return self.project(torch.cat(feats, dim=1))


class OCRFeatPlugin(FeatureSpacePlugin):
    """Object-Contextual Representation: same-class pixel context aggregation.

    Uses an internal aux 1x1 head to generate soft region masks (no extra
    supervision), aggregates per-class region representations from z, then
    refines z by mixing in the soft-attention-aligned region context.
    """

    def __init__(self, in_channels: int, num_classes: int,
                 mid_channels: int = 256, key_channels: int = 128):
        super().__init__(in_channels, mid_channels)
        self.num_classes = num_classes
        self.key_channels = key_channels
        # aux head for soft regions (uses z, not original x)
        self.aux_head = nn.Conv2d(mid_channels, num_classes, 1)
        # projections for pixel and region embeddings
        self.pixel_proj = nn.Sequential(
            nn.Conv2d(mid_channels, key_channels, 1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.GELU(),
        )
        self.region_proj = nn.Sequential(
            nn.Conv1d(mid_channels, key_channels, 1, bias=False),
            nn.BatchNorm1d(key_channels),
            nn.GELU(),
        )
        # final refinement: concat([z, context]) -> mid
        self.refine = nn.Sequential(
            nn.Conv2d(mid_channels + key_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )

    def core(self, z: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z.shape
        # soft region masks (B, K, H, W)
        regions = self.aux_head(z)
        regions = F.softmax(regions, dim=1)
        # per-class region representation: weighted average of z by region mask
        # region_rep: (B, C, K)
        region_rep = torch.einsum("bchw,bkhw->bck", z, regions)
        region_rep = region_rep / (regions.sum(dim=[2, 3]).unsqueeze(1) + 1e-6)
        # pixel and region keys
        pixel_key = self.pixel_proj(z)  # (B, K_emb, H, W)
        region_key = self.region_proj(region_rep)  # (B, K_emb, num_classes)
        # similarity (B, num_classes, H, W)
        sim = torch.einsum("bkhw,bkn->bnhw", pixel_key,
                           region_key) / (self.key_channels ** 0.5)
        sim = F.softmax(sim, dim=1)
        # aligned context in key space (B, K_emb, H, W)
        context = torch.einsum("bnhw,bkn->bkhw", sim, region_key)
        # refine
        return self.refine(torch.cat([z, context], dim=1))


def build_feat_plugin(cfg: dict, in_channels: int, num_classes: int):
    """Factory for feature-space plugins. Returns None if disabled."""
    if not cfg or not cfg.get("enabled", False):
        return None
    plugin_type = cfg.get("type", "msca").lower()
    mid = cfg.get("mid_channels", 256)
    if plugin_type == "msca":
        return MSCAFeatPlugin(
            in_channels=in_channels,
            mid_channels=mid,
            strip_kernels=tuple(cfg.get("strip_kernels", [7, 11, 21])),
        )
    if plugin_type == "aspp":
        return ASPPFeatPlugin(
            in_channels=in_channels,
            mid_channels=mid,
            branch_channels=cfg.get("branch_channels", 64),
            dilations=tuple(cfg.get("dilations", [1, 6, 12, 18])),
        )
    if plugin_type == "ocr":
        return OCRFeatPlugin(
            in_channels=in_channels,
            num_classes=num_classes,
            mid_channels=mid,
            key_channels=cfg.get("key_channels", 128),
        )
    raise ValueError(f"Unknown feat_plugin type: {plugin_type}")


class UPerNetDecoder(nn.Module):
    """UPerNet-style decoder with PPM and FPN.

    Adapted for ViT backbone where all intermediate features share the same
    spatial resolution (H/16 x W/16). The FPN top-down pathway enriches
    lower-level features with higher-level semantics via element-wise addition.

    Improvements over naive UPerNet:
      - PPM fuse uses bottleneck (Conv1x1 shrink + Conv3x3) to reduce params
      - All FPN levels are fused (concat + reduce), not just the last one
      - Segmentation head has Conv3x3 spatial smoothing before final Conv1x1
    """

    def __init__(
        self,
        in_channels_list: List[int],
        embed_dim: int = 256,
        num_classes: int = 7,
        pool_scales: Optional[List[int]] = None,
    ):
        super().__init__()
        if pool_scales is None:
            pool_scales = [1, 2, 3, 6]
        self.num_features = len(in_channels_list)

        # --- PPM (Pyramid Pooling Module) on highest-level feature ---
        self.ppm = nn.ModuleList()
        for scale in pool_scales:
            self.ppm.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels_list[-1], embed_dim, 1, bias=False),
                    nn.BatchNorm2d(embed_dim),
                    nn.GELU(),
                )
            )
        # Bottleneck fuse: reduce concat channels before spatial conv
        ppm_concat_dim = in_channels_list[-1] + embed_dim * len(pool_scales)
        self.ppm_fuse = nn.Sequential(
            nn.Conv2d(ppm_concat_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # --- FPN lateral + smoothing convolutions ---
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_ch in in_channels_list[:-1]:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, embed_dim, 1, bias=False),
                    nn.BatchNorm2d(embed_dim),
                )
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(embed_dim),
                    nn.GELU(),
                )
            )

        # --- Multi-level feature fusion ---
        # Collect all FPN levels + PPM output, concat, reduce
        num_fpn_levels = self.num_features  # lateral features + PPM output
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * num_fpn_levels, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # --- Segmentation head (spatial smoothing + classify) ---
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, num_classes, 1),
        )

    def forward(self, features: List[torch.Tensor], img_size: Tuple[int, int]) -> torch.Tensor:
        # PPM on highest-level feature
        ppm_in = features[-1]
        h, w = ppm_in.shape[2:]
        ppm_feats = [ppm_in]
        for ppm in self.ppm:
            ppm_feats.append(
                F.interpolate(ppm(ppm_in), size=(h, w), mode="bilinear", align_corners=False)
            )
        ppm_out = self.ppm_fuse(torch.cat(ppm_feats, dim=1))

        # FPN top-down: collect all levels
        fpn_features = []
        fpn_out = ppm_out
        for i in range(len(self.lateral_convs) - 1, -1, -1):
            lateral = self.lateral_convs[i](features[i])
            fpn_out = F.interpolate(fpn_out, size=lateral.shape[2:], mode="bilinear", align_corners=False)
            fpn_out = fpn_out + lateral
            fpn_out = self.fpn_convs[i](fpn_out)
            fpn_features.append(fpn_out)
        fpn_features.reverse()  # low-level → high-level

        # Fuse all FPN levels + PPM output
        all_features = fpn_features + [ppm_out]
        # All same spatial size for ViT; handle size mismatch for CNN backbones
        target_size = all_features[0].shape[2:]
        aligned = [
            F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            if f.shape[2:] != target_size else f
            for f in all_features
        ]
        fused = self.fuse(torch.cat(aligned, dim=1))

        out = F.interpolate(fused, size=img_size, mode="bilinear", align_corners=False)
        return self.head(out)


# ---------------------------------------------------------------------------
# Mask2Former Decoder Components
# ---------------------------------------------------------------------------


class PositionEmbeddingSine(nn.Module):
    """Standard sine-based positional encoding for 2D feature maps."""

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


class SelfAttentionLayer(nn.Module):
    """Post-norm self-attention layer for Mask2Former decoder."""

    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    """Post-norm cross-attention layer for Mask2Former decoder."""

    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None,
                     pos=None, query_pos=None):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None,
                    pos=None, query_pos=None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    """Post-norm feed-forward network layer for Mask2Former decoder."""

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation="relu",
                 normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class _MLP(nn.Module):
    """3-layer MLP used for mask embedding prediction in Mask2Former."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Mask2FormerTransformerDecoder(nn.Module):
    """
    Multi-scale masked transformer decoder (core of Mask2Former).

    Implements the iterative masked-attention decoding: each layer performs
    cross-attention (with predicted mask as attention mask) to a multi-scale
    feature pyramid, then self-attention, then FFN.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        mask_dim: int,
        pre_norm: bool = False,
        enforce_input_project: bool = False,
    ):
        super().__init__()

        self.num_heads = nheads
        self.num_layers = dec_layers
        self.num_queries = num_queries

        # Positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # Transformer decoder layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0,
                                   normalize_before=pre_norm)
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0,
                                    normalize_before=pre_norm)
            )
            self.transformer_ffn_layers.append(
                FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward,
                         dropout=0.0, normalize_before=pre_norm)
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Learnable query features and positional embeddings
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Level embeddings for 3 scales
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # Input projection: project multi-scale features to hidden_dim
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                # Xavier initialization
                nn.init.xavier_uniform_(self.input_proj[-1].weight, gain=1)
                if self.input_proj[-1].bias is not None:
                    nn.init.constant_(self.input_proj[-1].bias, 0)
            else:
                self.input_proj.append(nn.Sequential())

        # Output heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = _MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, multi_scale_features, mask_features, mask=None):
        """
        Args:
            multi_scale_features: list of 3 feature maps at different scales,
                each (B, in_channels, H_i, W_i)
            mask_features: (B, mask_dim, H, W) — high-res features for mask generation
            mask: ignored (kept for API compatibility)
        Returns:
            dict with pred_logits, pred_masks, aux_outputs
        """
        assert len(multi_scale_features) == self.num_feature_levels

        # disable mask, it does not affect performance
        del mask

        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            pos.append(self.pe_layer(multi_scale_features[i], None).flatten(2))
            projected = self.input_proj[i](multi_scale_features[i]).flatten(2)
            projected = projected + self.level_embed.weight[i][None, :, None]
            src.append(projected)

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # Prediction heads on learnable query features (before any decoder layer)
        outputs_class, outputs_mask, attn_mask = self._forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # Fix: where all positions are masked (all True), set to False
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # Cross-attention with masked attention
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )

            # Self-attention
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask = self._forward_prediction_heads(
                output, mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(predictions_class, predictions_mask),
        }
        return out

    def _forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        """Run classification and mask prediction heads, compute attention mask for next layer."""
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # QxNxC -> NxQxC
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # Compute attention mask for next cross-attention layer
        # [B, Q, H, W] -> [B, Q, h', w'] -> [B, Q, h'*w'] -> [B, h, Q, h'*w'] -> [B*h, Q, h'*w']
        attn_mask = F.interpolate(
            outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False
        )
        attn_mask = (
            attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1) < 0.5
        ).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]


class Mask2FormerDecoder(nn.Module):
    """
    Mask2Former decoder: Feature2Pyramid + MSDeformAttn pixel decoder + transformer decoder.

    Following Causal-Tune (AAAI 2026) and GeoSA-BaSA (ISPRS 2025) approach for
    plain ViT features:
    1. Each ViT layer feature → bilinear interp to create 4 pseudo-multi-scale features
       at strides [4, 8, 16, 32]
    2. 4 scales → MSDeformAttnPixelDecoder (6-layer deformable attention encoder + FPN)
       → mask_features (stride 4) + 3 multi_scale_features
    3. mask_features + multi_scale → Mask2FormerTransformerDecoder (9-layer masked
       cross-attention) → per-query class + mask predictions
    """

    def __init__(
        self,
        in_channels_list: List[int],
        embed_dim: int = 256,
        num_classes: int = 7,
        hidden_dim: int = 256,
        num_queries: int = 100,
        nheads: int = 8,
        dec_layers: int = 9,
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        in_channels = in_channels_list[0]

        # Pixel decoder: multi-scale deformable attention encoder + FPN lateral
        self.pixel_decoder = MSDeformAttnPixelDecoder(
            in_channels=in_channels,
            conv_dim=hidden_dim,
            mask_dim=hidden_dim,
            num_encoder_layers=6,
            nheads=nheads,
            dim_feedforward=1024,
            num_feature_levels=3,
        )

        # Transformer decoder: masked cross-attention + self-attention + FFN
        self.transformer_decoder = Mask2FormerTransformerDecoder(
            in_channels=hidden_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            mask_dim=hidden_dim,
            pre_norm=False,
            enforce_input_project=False,
        )

    def forward(self, features: List[torch.Tensor], img_size: Tuple[int, int]):
        """
        Args:
            features: list of 4 tensors from ViT layers, each (B, C, H/16, W/16)
            img_size: (H, W) original image size
        Returns:
            dict with pred_logits, pred_masks, aux_outputs
        """
        # Feature2Pyramid: bilinear interpolation to create 4 pseudo-multi-scale features
        # features[0] (earliest layer) → finest, features[3] (latest) → coarsest
        features_4scale = [
            F.interpolate(features[0], scale_factor=4, mode="bilinear", align_corners=False),
            F.interpolate(features[1], scale_factor=2, mode="bilinear", align_corners=False),
            features[2],
            F.interpolate(features[3], scale_factor=0.5, mode="bilinear", align_corners=False),
        ]

        # Pixel decoder: deformable attention cross-scale fusion
        mask_features, multi_scale_features = self.pixel_decoder(features_4scale)

        # Transformer decoder: query-based masked attention
        out = self.transformer_decoder(multi_scale_features, mask_features)
        return out

    def predict(self, features: List[torch.Tensor], img_size: Tuple[int, int]) -> torch.Tensor:
        """Returns (B, num_classes, H, W) logits for evaluation pipeline."""
        out = self.forward(features, img_size)
        pred_logits = out["pred_logits"]
        pred_masks = F.interpolate(
            out["pred_masks"], size=img_size, mode="bilinear", align_corners=False
        )
        class_scores = F.softmax(pred_logits, dim=-1)
        mask_probs = pred_masks.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", class_scores, mask_probs)
        semseg = semseg[:, :self.num_classes, :, :]
        return semseg


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
        elif decoder_type == "sru":
            self.layers_to_extract = [3, 6, 12, 16, 20, 24]
        else:
            self.layers_to_extract = [3, 6, 9, 12]
        self.layers_to_extract_0idx = [l - 1 for l in self.layers_to_extract]

        # --- Decoder ---
        decoder_cfg = cfg.get("decoder", {})
        decoder_embed_dim = decoder_cfg.get("embed_dim", 96 if self.embed_dim < 512 else 256)
        num_classes = cfg.get("num_classes", 7)
        in_channels = [self.embed_dim] * len(self.layers_to_extract)

        # --- Optional feature-space plugin (MLP decoder only, inserted post-fusion) ---
        feat_plugin_cfg = decoder_cfg.get("feat_plugin", None)
        feat_plugin = None
        if decoder_type == "mlp" and feat_plugin_cfg and feat_plugin_cfg.get("enabled", False):
            fused_dim = decoder_embed_dim * len(self.layers_to_extract)
            feat_plugin = build_feat_plugin(feat_plugin_cfg, fused_dim, num_classes)
            print(f"feat_plugin enabled: type={feat_plugin_cfg.get('type')}, "
                  f"in_channels={fused_dim}, mid={feat_plugin_cfg.get('mid_channels', 256)}")

        if decoder_type == "mlp":
            self.decoder = MLPDecoder(in_channels, decoder_embed_dim, num_classes,
                                      feat_plugin=feat_plugin)
        elif decoder_type == "pfu":
            self.decoder = PFUDecoder(in_channels, decoder_embed_dim, num_classes)
        elif decoder_type == "sru":
            n_deep = len(self.layers_to_extract) - 2
            self.decoder = SRUDecoder(in_channels, decoder_embed_dim, num_classes,
                                      n_deep_layers=n_deep)
        elif decoder_type == "upernet":
            self.decoder = UPerNetDecoder(
                in_channels,
                decoder_embed_dim,
                num_classes,
                pool_scales=decoder_cfg.get("pool_scales", [1, 2, 3, 6]),
            )
        elif decoder_type == "mask2former":
            m2f_cfg = decoder_cfg.get("mask2former", {})
            self.decoder = Mask2FormerDecoder(
                in_channels_list=in_channels,
                embed_dim=decoder_embed_dim,
                num_classes=num_classes,
                hidden_dim=m2f_cfg.get("hidden_dim", 256),
                num_queries=m2f_cfg.get("num_queries", 100),
                nheads=m2f_cfg.get("nheads", 8),
                dec_layers=m2f_cfg.get("dec_layers", 9),
                dim_feedforward=m2f_cfg.get("dim_feedforward", 2048),
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        # --- Optional Spatial Prior (P0 diagnostic) ---
        # RGB -> S8 CNN stem + cross-attention injector into selected frozen
        # feature maps. gamma=0 init keeps model at R0 behavior at start.
        sp_cfg = decoder_cfg.get("spatial_prior", None)
        if sp_cfg and sp_cfg.get("enabled", False):
            inject_layers_1idx = sp_cfg.get("inject_layers", [18, 24])
            self._sp_inject_layers_0idx = [l - 1 for l in inject_layers_1idx]
            # Validate inject_layers against layers_to_extract
            for l0 in self._sp_inject_layers_0idx:
                if l0 not in self.layers_to_extract_0idx:
                    raise ValueError(
                        f"spatial_prior.inject_layers entry (1-idx={l0+1}) "
                        f"is not in layers_to_extract {self.layers_to_extract}; "
                        f"injection has no effect on that layer."
                    )
            d_spm = sp_cfg.get("spm_channels", 128)
            d_bottleneck = sp_cfg.get("d_bottleneck", 256)
            n_heads = sp_cfg.get("num_heads", 8)
            spm_blocks = sp_cfg.get("spm_blocks", 2)
            sp_dropout = sp_cfg.get("dropout", 0.0)
            self.spatial_prior_enabled = True
            self.spm = SpatialPriorModule(
                out_channels=d_spm, n_blocks=spm_blocks, dropout=sp_dropout,
            )
            self.spatial_injector = SpatialPriorInjector(
                d_frozen=self.embed_dim,
                d_spm=d_spm,
                d_bottleneck=d_bottleneck,
                num_heads=n_heads,
                dropout=sp_dropout,
            )
            print(
                f"Spatial prior enabled: inject_layers={inject_layers_1idx}, "
                f"spm_channels={d_spm}, d_bottleneck={d_bottleneck}, "
                f"num_heads={n_heads}, spm_blocks={spm_blocks}"
            )
        else:
            self.spatial_prior_enabled = False
            self._sp_inject_layers_0idx = []
            self.spm = None
            self.spatial_injector = None

        # --- Optional SRU plugin (shallow ViT feature injection into logits) ---
        sru_cfg = decoder_cfg.get("sru_plugin", None)
        if sru_cfg and sru_cfg.get("enabled", False):
            # Use the FIRST n_shallow features from layers_to_extract as shallow inputs
            n_shallow = sru_cfg.get("n_shallow", 2)
            if len(self.layers_to_extract) < n_shallow:
                raise ValueError(
                    f"SRU plugin needs {n_shallow} shallow features, but "
                    f"layers_to_extract only has {len(self.layers_to_extract)}"
                )
            self._sru_n_shallow = n_shallow
            shallow_channels = [self.embed_dim] * n_shallow
            self.sru_plugin = SRUPlugin(
                num_classes=num_classes,
                shallow_channels=shallow_channels,
                embed_dim=sru_cfg.get("embed_dim", 128),
            )
            sru_layers = [self.layers_to_extract[i] for i in range(n_shallow)]
            print(f"SRU plugin enabled: shallow layers={sru_layers}, "
                  f"embed_dim={sru_cfg.get('embed_dim', 128)}")
        else:
            self.sru_plugin = None
            self._sru_n_shallow = 0

        # --- Optional PFEB post-fusion refiner ---
        pfeb_cfg = decoder_cfg.get("pfeb", None)
        if pfeb_cfg and pfeb_cfg.get("enabled", False):
            self.pfeb = PFEBRefiner(
                num_classes=num_classes,
                hidden_dim=pfeb_cfg.get("hidden_dim", 64),
            )
            print(f"PFEB refiner enabled: hidden_dim={pfeb_cfg.get('hidden_dim', 64)}")
        else:
            self.pfeb = None

        # --- Optional FreqPlugin (frequency-domain logit refiner) ---
        freq_cfg = decoder_cfg.get("freq_plugin", None)
        if freq_cfg and freq_cfg.get("enabled", False):
            self.freq_plugin = FreqPlugin(
                num_classes=num_classes,
                hidden_dim=freq_cfg.get("hidden_dim", 32),
                n_bands=freq_cfg.get("n_bands", 8),
            )
            print(f"FreqPlugin enabled: hidden_dim={freq_cfg.get('hidden_dim', 32)}, "
                  f"n_bands={freq_cfg.get('n_bands', 8)}")
        else:
            self.freq_plugin = None


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

        # Spatial prior injection (P0 diagnostic): enrich selected frozen
        # feature maps with S8 from a from-scratch CNN stem on RGB.
        # gamma=0 init means feature_maps are unchanged at epoch 0 step 0,
        # so the model starts exactly at R0 behavior and only diverges as
        # gamma learns to grow.
        if self.spatial_prior_enabled:
            s8 = self.spm(x)
            for i, l0 in enumerate(self.layers_to_extract_0idx):
                if l0 in self._sp_inject_layers_0idx:
                    feature_maps[i] = self.spatial_injector(feature_maps[i], s8)

        out = self.decoder(feature_maps, img_size=(H, W))
        if self.sru_plugin is not None and isinstance(out, torch.Tensor):
            shallow = feature_maps[:self._sru_n_shallow]
            out = self.sru_plugin(out, shallow)
        if self.pfeb is not None and isinstance(out, torch.Tensor):
            out = self.pfeb(out)
        if self.freq_plugin is not None and isinstance(out, torch.Tensor):
            out = self.freq_plugin(out)
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
