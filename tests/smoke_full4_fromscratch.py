"""Smoke test for full-4-expert from-scratch config (GPU 3).

Verifies the new per-block spectral dispatcher produces the correct module
distribution and shapes:
  - blocks 1-8:   RoutedSSDLoRAModule with TCAM, 3 experts (sem/spa/tex), no spectral
  - blocks 9-16:  RoutedSSDLoRAModule no TCAM, 3 experts (sem/spa/spe), gamma_spe=5e-4
  - blocks 17-24: RoutedSSDLoRAModule no TCAM, 3 experts (sem/spa/spe), gamma_spe=2e-3

Run: ssh GPU-Server 'cd /storage2/lijichao/yangchenxi/chao_research && \
    export CONDA_ENVS_PATH=/storage2/lijichao/yangchenxi/conda_envs && \
    source /opt/miniconda3/etc/profile.d/conda.sh && conda activate chao && \
    CUDA_VISIBLE_DEVICES=3 python tests/smoke_full4_fromscratch.py'
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import yaml
from model import HALoRASeg, RoutedSSDLoRAModule, SSDLoRAModule


def main():
    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "cross_dataset", "config_p1_full4_pots_fromscratch_gpu3.yaml",
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print(f"Config: {cfg_path}")
    print(f"  lora.mode = {cfg['lora']['mode']}")
    print(f"  spectral.enabled = {cfg['lora']['spectral']['enabled']}")
    print(f"  spectral.ranges = {cfg['lora']['spectral'].get('ranges')}")
    print(f"  tcam_blocks = {cfg['lora']['tcam_blocks']}")
    print()

    # Build model (this exercises the dispatcher).
    print("Building HALoRASeg with full-4-expert config...")
    model = HALoRASeg(cfg)
    model.eval()
    print("  Build OK.")
    print()

    # Walk blocks and classify each qkv module.
    print("Per-block qkv module types:")
    print(f"  {'blk':>4}  {'module_type':<22}  {'tcam':<6}  {'spectral':<9}  {'γ_spe':<10}  {'n_experts'}")
    counts = {"routed_tcam": 0, "routed_spe_mid": 0, "routed_spe_high": 0, "vanilla": 0, "other": 0}
    for i, block in enumerate(model.backbone.blocks):
        qkv = block.attn.qkv
        if isinstance(qkv, RoutedSSDLoRAModule):
            has_tcam = qkv.tcam is not None
            has_spe = qkv.spectral_enabled
            n_exp = qkv.router.n_experts
            if has_tcam and not has_spe:
                gamma_str = "-"
                kind = "routed_tcam"
            elif has_spe and not has_tcam:
                gamma = qkv.gamma_spe.item()
                gamma_str = f"{gamma:.0e}"
                # Classify mid vs high by gamma
                if abs(gamma - 5e-4) < 1e-6:
                    kind = "routed_spe_mid"
                elif abs(gamma - 2e-3) < 1e-6:
                    kind = "routed_spe_high"
                else:
                    kind = "routed_spe_?"
            else:
                gamma_str = "?"
                kind = "other"
            print(f"  {i+1:>4}  RoutedSSDLoRAModule   {str(has_tcam):<6}  {str(has_spe):<9}  {gamma_str:<10}  {n_exp}")
        elif isinstance(qkv, SSDLoRAModule):
            kind = "vanilla"
            print(f"  {i+1:>4}  SSDLoRAModule         {'-':<6}  {'-':<9}  {'-':<10}  -")
        else:
            kind = "other"
            print(f"  {i+1:>4}  {type(qkv).__name__:<22}")
        counts[kind] = counts.get(kind, 0) + 1

    print()
    print("Block counts (qkv only):")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print()

    # Expected: 8 tcam (1-8) + 8 spe_mid (9-16) + 8 spe_high (17-24) = 24 routed, 0 vanilla
    assert counts["routed_tcam"] == 8, f"Expected 8 routed_tcam blocks, got {counts['routed_tcam']}"
    assert counts["routed_spe_mid"] == 8, f"Expected 8 spe_mid blocks, got {counts['routed_spe_mid']}"
    assert counts["routed_spe_high"] == 8, f"Expected 8 spe_high blocks, got {counts['routed_spe_high']}"
    assert counts.get("vanilla", 0) == 0, f"Expected 0 vanilla blocks, got {counts.get('vanilla', 0)}"
    print("[PASS] Per-block dispatcher distribution correct.")
    print()

    # Verify all 24 blocks are RoutedSSDLoRAModule for qkv, fc1, fc2.
    n_routed_qkv = sum(1 for b in model.backbone.blocks if isinstance(b.attn.qkv, RoutedSSDLoRAModule))
    n_routed_fc1 = sum(1 for b in model.backbone.blocks if isinstance(b.mlp.fc1, RoutedSSDLoRAModule))
    n_routed_fc2 = sum(1 for b in model.backbone.blocks if isinstance(b.mlp.fc2, RoutedSSDLoRAModule))
    print(f"Routed modules: qkv={n_routed_qkv}/24, fc1={n_routed_fc1}/24, fc2={n_routed_fc2}/24")
    assert n_routed_qkv == 24 and n_routed_fc1 == 24 and n_routed_fc2 == 24, "All blocks must be routed"
    print("[PASS] All 72 modules (24 blocks x 3 targets) are RoutedSSDLoRAModule.")
    print()

    # Forward pass on dummy input.
    print("Forward pass test...")
    B = 2
    H = W = 224
    x = torch.randn(B, 3, H, W)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.float16, enabled=False):
        out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    print(f"  Output shape: {tuple(out.shape)}")
    print(f"  Has NaN: {torch.isnan(out).any().item()}, Has Inf: {torch.isinf(out).any().item()}")
    assert not torch.isnan(out).any(), "Forward output has NaN"
    assert not torch.isinf(out).any(), "Forward output has Inf"
    print("[PASS] Forward clean (no NaN/Inf).")
    print()

    # Param count by LR group (mimic train_cross.py classifier).
    LORA_CORE_KEYWORDS = ("lora_sem_", "lora_str_", "alpha_sem", "alpha_str", "spatial_conv", "tcam")
    EXPERT_NEW_KEYWORDS = ("lora_spe_", "feat_spectral_gate", "gamma_spe")
    ROUTER_KEYWORDS = ("router.",)
    counts_by_group = {"decoder_r0": 0, "lora_core": 0, "router_gate": 0, "expert_new": 0}
    n_trainable = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_trainable += p.numel()
        if any(k in name for k in ROUTER_KEYWORDS):
            counts_by_group["router_gate"] += p.numel()
        elif any(k in name for k in EXPERT_NEW_KEYWORDS):
            counts_by_group["expert_new"] += p.numel()
        elif any(k in name for k in LORA_CORE_KEYWORDS):
            counts_by_group["lora_core"] += p.numel()
        elif name.startswith("decoder."):
            counts_by_group["decoder_r0"] += p.numel()
        else:
            counts_by_group["lora_core"] += p.numel()
    print(f"Trainable params: {n_trainable:,}")
    for g, c in counts_by_group.items():
        print(f"  {g}: {c:,}")
    assert counts_by_group["expert_new"] > 0, "expert_new (spectral) must have params when spectral enabled"
    print("[PASS] expert_new has params (spectral enabled).")
    print()
    print("=" * 60)
    print("ALL SMOKE TESTS PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
