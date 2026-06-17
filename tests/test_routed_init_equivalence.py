"""Run A-main smoke tests (override 2026-06-17 §"Tests Required").

Verifies:
1. Legacy R0 config still builds and loads with missing=0, unexpected=0.
2. Routed module with s_sem=s_spa=s_tex=1, gamma_spe=0 matches the equivalent
   manually composed R0 sem + TCAM structural output.
3. Router scales have shape (B, 3) or (B, 4) depending on whether high-source
   spectral is enabled.
4. Router scales initialize to 1.0.
5. No SpectralBandGate class or low-rank FFT module is instantiated.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import math
import torch
import torch.nn as nn

from model import (
    SSDLoRAModule,
    RoutedSSDLoRAModule,
    ExpertRouter,
    FeatureSpectralGate,
    MultiScaleDWConv,
    TCAM,
)


def make_dummy_linear(d_in=64, d_out=64):
    lin = nn.Linear(d_in, d_out, bias=False)
    lin.weight.requires_grad_(False)
    return lin


def test_router_scales_init_to_one():
    """Test 4: router scales initialize to 1.0."""
    router = ExpertRouter(d_in=64, n_experts=3, hidden_dim=16)
    x = torch.randn(4, 50, 64)  # 4 samples, 50 tokens, D=64
    scales = router(x, n_non_spatial=2)
    assert scales.shape == (4, 3), f"Expected (4, 3), got {scales.shape}"
    expected = torch.ones(4, 3)
    assert torch.allclose(scales, expected, atol=1e-6), (
        f"Router scales should be 1.0 at init, got {scales[0].tolist()}"
    )
    print(f"  Test 4 PASS: router init scales={scales[0].tolist()}, shape={tuple(scales.shape)}")


def test_router_scales_shape_with_spectral():
    """Test 3 (partial): router output shape is (B, 4) when spectral enabled."""
    # Construct a routed module with spectral enabled.
    frozen = make_dummy_linear(d_in=64, d_out=64)
    mod = RoutedSSDLoRAModule(
        frozen_linear=frozen,
        r_sem=4, r_str=4,
        n_non_spatial=2,
        structural_path_type="multi_scale_dwconv",
        tcam_type="tcam_cov",
        spectral_enabled=True,
        r_spe=8,
        spectral_n_bands=4,
        spectral_hidden_dim=32,
        router_hidden_dim=32,
    )
    x = torch.randn(4, 50, 64)
    scales = mod.router(x, n_non_spatial=2)
    assert scales.shape == (4, 4), f"Expected (4, 4) with spectral, got {scales.shape}"
    print(f"  Test 3 PASS: router with spectral shape={tuple(scales.shape)}")


def test_no_spectral_band_gate_class():
    """Test 5: no SpectralBandGate class or low-rank FFT module is instantiated."""
    # Verify the class is not defined.
    import model as model_module
    assert not hasattr(model_module, "SpectralBandGate"), (
        "SpectralBandGate class must not exist (override forbids r_spe=2/4 FFT)."
    )
    # Verify FeatureSpectralGate sources from x BEFORE LoRA_B compression:
    # check that it accepts the full x (B, N, D) and n_non_spatial, not a
    # compressed (B, N, r_spe) tensor.
    gate = FeatureSpectralGate(d_in=64, n_bands=4, hidden_dim=16)
    # 2 non-spatial + 7*7=49 spatial = 51 tokens, perfect square spatial grid
    x_full = torch.randn(2, 51, 64)  # full D, not compressed
    out = gate(x_full, n_non_spatial=2)
    assert out.shape == (2, 64), f"Gate output should be (B, D), got {out.shape}"
    print(f"  Test 5 PASS: no SpectralBandGate class; FeatureSpectralGate sources from full x")


def test_routed_init_equivalent_to_r0_ssd_lora():
    """Test 2: routed with s=1, gamma_spe=0 matches R0 SSD-LoRA with TCAM.

    Builds two modules with identical r_sem, r_str, structural_path_type, tcam_*,
    and shares all corresponding weights. At init (router zero-init → scales=1,
    gamma_spe=0), outputs must match within tolerance.
    """
    torch.manual_seed(42)
    d_in, d_out = 64, 64
    r_sem, r_str = 4, 4
    n_non_spatial = 2
    side = 7  # spatial grid 7x7=49 + 2 non-spatial = 51 tokens

    # Build frozen linear with identical weights for both modules.
    frozen_w = torch.randn(d_out, d_in)
    frozen_r0 = nn.Linear(d_in, d_out, bias=False)
    frozen_r0.weight.data = frozen_w.clone()
    frozen_r0.weight.requires_grad_(False)
    frozen_routed = nn.Linear(d_in, d_out, bias=False)
    frozen_routed.weight.data = frozen_w.clone()
    frozen_routed.weight.requires_grad_(False)

    r0 = SSDLoRAModule(
        frozen_linear=frozen_r0,
        r_sem=r_sem, r_str=r_str,
        n_non_spatial=n_non_spatial,
        lora_mode="ssd",
        structural_path_type="multi_scale_dwconv",
        tcam_type="tcam_cov",
        tcam_gamma=0.2,
        tcam_hidden_min=8,
    )
    routed = RoutedSSDLoRAModule(
        frozen_linear=frozen_routed,
        r_sem=r_sem, r_str=r_str,
        n_non_spatial=n_non_spatial,
        structural_path_type="multi_scale_dwconv",
        tcam_type="tcam_cov",
        tcam_gamma=0.2,
        tcam_hidden_min=8,
        spectral_enabled=False,  # no spectral → no gamma_spe
        router_hidden_dim=32,
    )

    # Copy all shared weights from r0 → routed.
    routed.lora_sem_A.data = r0.lora_sem_A.data.clone()
    routed.lora_sem_B.data = r0.lora_sem_B.data.clone()
    routed.alpha_sem.data = r0.alpha_sem.data.clone()
    routed.lora_str_A.data = r0.lora_str_A.data.clone()
    routed.lora_str_B.data = r0.lora_str_B.data.clone()
    routed.alpha_str.data = r0.alpha_str.data.clone()

    # spatial_conv: same arch (multi_scale_dwconv) — copy weights.
    for p_r0, p_routed in zip(r0.spatial_conv.parameters(), routed.spatial_conv.parameters()):
        p_routed.data = p_r0.data.clone()

    # TCAM: same config — copy weights.
    for p_r0, p_routed in zip(r0.tcam.parameters(), routed.tcam.parameters()):
        p_routed.data = p_r0.data.clone()

    # Verify router is at init (zero-init out_proj → scales=1).
    x = torch.randn(3, n_non_spatial + side * side, d_in)
    scales = routed.router(x, n_non_spatial=n_non_spatial)
    assert torch.allclose(scales, torch.ones_like(scales), atol=1e-6), (
        f"Scales must be 1.0 at init for equivalence, got {scales[0].tolist()}"
    )

    r0.eval()
    routed.eval()
    with torch.no_grad():
        out_r0 = r0(x)
        out_routed = routed(x)

    max_diff = (out_r0 - out_routed).abs().max().item()
    mean_diff = (out_r0 - out_routed).abs().mean().item()
    assert max_diff < 1e-5, (
        f"Routed init output drifts from R0 SSD-LoRA: max_diff={max_diff:.2e}, "
        f"mean_diff={mean_diff:.2e}. Expected < 1e-5."
    )
    print(
        f"  Test 2 PASS: routed init ≡ R0 SSD-LoRA with TCAM "
        f"(max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})"
    )


def test_spectral_path_starts_silent():
    """Spectral expert must contribute 0 at init (gamma_spe=0)."""
    frozen = make_dummy_linear(d_in=64, d_out=64)
    mod = RoutedSSDLoRAModule(
        frozen_linear=frozen,
        r_sem=4, r_str=4,
        n_non_spatial=2,
        structural_path_type="multi_scale_dwconv",
        tcam_type="tcam_cov",
        spectral_enabled=True,
        r_spe=8,
        spectral_n_bands=4,
        spectral_hidden_dim=32,
        router_hidden_dim=32,
    )
    # gamma_spe starts at 0
    assert mod.gamma_spe.item() == 0.0, f"gamma_spe must init to 0, got {mod.gamma_spe.item()}"
    # Verify the forward skips spectral when gamma_spe=0 by checking output equals
    # the equivalent RoutedSSDLoRAModule WITHOUT spectral (same weights).
    frozen2 = make_dummy_linear(d_in=64, d_out=64)
    frozen2.weight.data = frozen.weight.data.clone()
    mod_no_spe = RoutedSSDLoRAModule(
        frozen_linear=frozen2,
        r_sem=4, r_str=4,
        n_non_spatial=2,
        structural_path_type="multi_scale_dwconv",
        tcam_type="tcam_cov",
        spectral_enabled=False,
        router_hidden_dim=32,
    )
    # Copy shared weights.
    for name, p in mod_no_spe.named_parameters():
        if name.startswith("router.") or name in ("gamma_spe",) or "spectral_gate" in name or "lora_spe_" in name:
            continue
        if name in dict(mod.named_parameters()):
            p.data = dict(mod.named_parameters())[name].data.clone()

    mod.eval()
    mod_no_spe.eval()
    side = 7
    x = torch.randn(2, 2 + side * side, 64)
    with torch.no_grad():
        out1 = mod(x)
        out2 = mod_no_spe(x)
    max_diff = (out1 - out2).abs().max().item()
    assert max_diff < 1e-5, (
        f"Spectral-enabled module at gamma_spe=0 should match no-spectral module, "
        f"max_diff={max_diff:.2e}"
    )
    print(f"  Spectral-silence PASS: gamma_spe=0 init keeps spectral path silent (max_diff={max_diff:.2e})")


def main():
    print("Run A-main smoke tests (override 2026-06-17 §Tests Required)")
    print("=" * 70)
    print("[Test 4] Router scales init to 1.0:")
    test_router_scales_init_to_one()
    print()
    print("[Test 3] Router scales shape with spectral:")
    test_router_scales_shape_with_spectral()
    print()
    print("[Test 5] No SpectralBandGate / no low-rank FFT:")
    test_no_spectral_band_gate_class()
    print()
    print("[Test 2] Routed init ≡ R0 SSD-LoRA with TCAM:")
    test_routed_init_equivalent_to_r0_ssd_lora()
    print()
    print("[Extra] Spectral path silent at init (gamma_spe=0):")
    test_spectral_path_starts_silent()
    print()
    print("=" * 70)
    print("ALL UNIT TESTS PASS")
    print()
    print("Note: Test 1 (R0 ckpt loads with missing=0, unexpected=0) requires")
    print("GPU-Server access and the R0 trained checkpoint. Run separately:")
    print("  ssh GPU-Server 'cd .../chao_research && python tests/test_r0_ckpt_load.py'")


if __name__ == "__main__":
    main()
