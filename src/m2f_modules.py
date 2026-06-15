"""Mask2Former utility modules — adapted from DINOv3 official implementation.

Pure PyTorch implementation (no CUDA extension required).
Uses F.grid_sample for deformable attention with autograd support.

References:
- DINOv3 official: dinov3/eval/segmentation/models/
- Causal-Tune (AAAI 2026): DINOv2 + M2F for DGSS
- GeoSA-BaSA (ISPRS 2025): DINOv2 + M2F + side network
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Position Encoding
# ---------------------------------------------------------------------------

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=2 * math.pi):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
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
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# ---------------------------------------------------------------------------
# Multi-Scale Deformable Attention (Pure PyTorch)
# ---------------------------------------------------------------------------

# Try mmcv CUDA extension for faster deformable attention (auto-detect)
# Currently disabled — mmcv CUDA kernel has shape compatibility issues.
# Pure PyTorch F.grid_sample fallback works correctly with autograd.
_HAS_MMCV_CUDA = False


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """Pure PyTorch fallback: multi-scale deformable attention via F.grid_sample."""
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        else:
            raise ValueError(f"Last dim of reference_points must be 2, got {reference_points.shape[-1]}")

        if _HAS_MMCV_CUDA:
            # mmcv CUDA: value(N,L,D), shapes(L,2), start_index(L), locations(N,Q,H,Lv,P,2), weights(N,Q,H,Lv,P)
            output = _mmcv_cuda_fn.apply(
                value, input_spatial_shapes, input_level_start_index,
                sampling_locations, attention_weights, 64
            )
        else:
            # Pure PyTorch fallback: needs (N, Len_in, n_heads, D) for per-head grid_sample
            value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
            output = ms_deform_attn_core_pytorch(
                value, input_spatial_shapes, sampling_locations, attention_weights
            )
        output = self.output_proj(output)
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


# ---------------------------------------------------------------------------
# MSDeformAttn Transformer Encoder (for pixel decoder)
# ---------------------------------------------------------------------------

class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024, dropout=0.1, activation="relu", num_feature_levels=4, enc_n_points=4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)
        self.level_encoding = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.level_encoding)

    @staticmethod
    def get_valid_ratio(mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src_flat = src.flatten(2).transpose(1, 2)
            mask_flat = mask.flatten(1)
            pos_flat = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos = pos_flat + self.level_encoding[lvl].view(1, 1, -1)
            src_flatten.append(src_flat)
            mask_flatten.append(mask_flat)
            lvl_pos_embed_flatten.append(lvl_pos)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes_t = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes_t.new_zeros((1,)), spatial_shapes_t.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        memory = self.encoder(src_flatten, spatial_shapes_t, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        return memory, spatial_shapes, level_start_index


# ---------------------------------------------------------------------------
# MSDeformAttn Pixel Decoder
# ---------------------------------------------------------------------------

class MSDeformAttnPixelDecoder(nn.Module):
    """Pixel decoder using multi-scale deformable attention encoder.

    Adapted from DINOv3 official implementation. Uses pure PyTorch
    (no CUDA extension needed).

    Input: 4 feature maps at strides [4, 8, 16, 32]
    Output: mask_features (stride 4), 3 multi_scale_features for transformer decoder
    """

    def __init__(self, in_channels=1024, conv_dim=256, mask_dim=256, num_encoder_layers=6, nheads=8, dim_feedforward=1024, num_feature_levels=3):
        super().__init__()
        self.num_feature_levels = num_feature_levels

        # Input projections: project from in_channels to conv_dim for the 3 coarsest scales
        # (finest scale handled by FPN lateral)
        self.input_convs = nn.ModuleList()
        for _ in range(num_feature_levels):
            self.input_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, conv_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, conv_dim),
            ))
        for proj in self.input_convs:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)

        self.encoder = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim, nhead=nheads, dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers, num_feature_levels=num_feature_levels,
        )
        self.pe_layer = PositionEmbeddingSine(conv_dim // 2, normalize=True)

        # Mask feature projection
        self.mask_feature = nn.Conv2d(conv_dim, mask_dim, kernel_size=1)
        nn.init.kaiming_uniform_(self.mask_feature.weight, a=1)
        nn.init.constant_(self.mask_feature.bias, 0)

        # FPN lateral for the finest scale (stride 4)
        self.lateral_conv = nn.Sequential(
            nn.Conv2d(in_channels, conv_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, conv_dim),
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, conv_dim),
            nn.ReLU(inplace=True),
        )
        nn.init.xavier_uniform_(self.lateral_conv[0].weight, gain=1)
        nn.init.kaiming_uniform_(self.output_conv[0].weight, a=1)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, features_4scale):
        """Args:
            features_4scale: list of 4 tensors at strides [4, 8, 16, 32],
                each (B, in_channels, H_i, W_i) from Feature2Pyramid
        Returns:
            mask_features: (B, mask_dim, H/4, W/4)
            multi_scale_features: list of 3 tensors for transformer decoder
        """
        # Feed the 3 coarsest scales (strides 8, 16, 32) to deformable attention encoder
        srcs = []
        pos = []
        # Order: coarsest to finest for the 3 levels
        for idx in range(self.num_feature_levels):
            x = features_4scale[self.num_feature_levels - idx].float()  # stride 32, 16, 8
            srcs.append(self.input_convs[idx](x))
            pos.append(self.pe_layer(x))

        y, spatial_shapes, level_start_index = self.encoder(srcs, pos)
        bs = y.shape[0]

        split_size_or_sections = [h * w for h, w in spatial_shapes]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # FPN lateral for finest scale: stride 4 features + upsampled decoder output
        cur_fpn = self.lateral_conv(features_4scale[0].float())  # stride 4
        y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
        y = self.output_conv(y)
        out.append(y)

        # mask_features from finest scale, 3 multi_scale from coarsest to finest
        mask_features = self.mask_feature(out[-1])
        multi_scale_features = out[:3]  # 3 scales for transformer decoder

        return mask_features, multi_scale_features
