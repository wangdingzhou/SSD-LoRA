"""SCHRR v2 smoke test: forward + backward + gradient flow + router non-collapse.

Run on GPU-Server (needs CUDA + DINOv3 weights):
  CUDA_VISIBLE_DEVICES=2 python -u tests/smoke_schrr_v2.py
"""
import sys
sys.path.insert(0, ".")
import torch
import yaml
from src.model import get_model


def test_full_forward_backward():
    """Build full model with v2 config, forward + backward 1 batch."""
    cfg = yaml.safe_load(open("cross_dataset/config_p1_schrr_v2_pots_r0.yaml"))
    model = get_model(cfg)
    model.cuda()
    model.train()

    images = torch.randn(2, 3, 512, 512, device="cuda")
    masks = torch.randint(0, 6, (2, 512, 512), device="cuda")

    output = model(images)
    assert isinstance(output, dict), f"expected dict, got {type(output)}"
    logits = output["logits"]
    assert logits.shape == (2, 6, 512, 512), f"logits shape wrong: {logits.shape}"
    assert "basis_probs_for_loss" in output, "missing basis_probs_for_loss"

    loss = torch.nn.functional.cross_entropy(logits, masks)
    loss.backward()
    print(f"[PASS] forward+backward: loss={loss.item():.4f}, logits={logits.shape}")
    return model


def test_gradient_flow(model):
    """Check gradient flows to critical v2 params."""
    critical_params = {
        "q_proj": "decoder.schrr_8.q_proj.weight",
        "router_conv": "decoder.schrr_8.lar.router_conv.weight",
        "v_proj": "decoder.schrr_8.lar.v_proj.weight",
        "attn_convs": "decoder.schrr_8.lar.attn_convs.0.weight",
        "raw_lambda": "decoder.schrr_8.fuse.raw_lambda",
        "out_proj": "decoder.schrr_8.lar.out_proj.weight",
    }
    model_state = dict(model.named_parameters())
    for label, name in critical_params.items():
        assert name in model_state, f"{label} ({name}) not found in model"
        param = model_state[name]
        assert param.grad is not None, f"{label} ({name}) has NO gradient"
        assert param.grad.abs().sum() > 0, f"{label} ({name}) gradient is zero"
        print(f"[PASS] {label}: grad nonzero (max={param.grad.abs().max().item():.6e})")


def test_lam_bounded(model):
    """Check λ ∈ (0, 0.3) for both fuse blocks."""
    for blk_name in ["schrr_8", "schrr_4"]:
        fuse = getattr(model.decoder, blk_name).fuse
        lam = fuse.lam.item()
        assert 0.0 < lam < 0.3, f"{blk_name} lam out of (0,0.3): {lam}"
        print(f"[PASS] {blk_name} lam={lam:.6f} (in (0, 0.3))")


def test_router_temperature_method(model):
    """Test set_router_temperature propagates to both blocks + LAR."""
    model.decoder.set_router_temperature(3.0)
    assert model.decoder.schrr_8.lar.temperature == 3.0, "schrr_8 temp not set"
    assert model.decoder.schrr_4.lar.temperature == 3.0, "schrr_4 temp not set"
    print(f"[PASS] set_router_temperature(3.0): schrr_8={model.decoder.schrr_8.lar.temperature}, schrr_4={model.decoder.schrr_4.lar.temperature}")
    model.decoder.set_router_temperature(1.0)


def test_warm_start_loading():
    """Verify warm_start loads backbone+LoRA, skips decoder."""
    cfg = yaml.safe_load(open("cross_dataset/config_p1_schrr_v2_pots_r0.yaml"))
    ws_path = cfg["warm_start"]
    if not ws_path or "<" in ws_path:
        print("[SKIP] warm_start path not filled, skipping")
        return

    model = get_model(cfg)
    ckpt = torch.load(ws_path, map_location="cpu", weights_only=False)
    pretrained = ckpt["model"]
    model_state = model.state_dict()

    ALLOWED_PREFIXES = ("backbone.",)
    BLOCKED_PREFIXES = ("decoder.",)
    matched = 0
    skipped_decoder = 0
    for k in pretrained:
        if any(k.startswith(p) for p in BLOCKED_PREFIXES):
            skipped_decoder += 1
        elif any(k.startswith(p) for p in ALLOWED_PREFIXES) and k in model_state and pretrained[k].shape == model_state[k].shape:
            matched += 1
    print(f"warm_start: {matched} backbone+LoRA keys match, {skipped_decoder} decoder keys skipped")
    assert matched > 1000, f"too few keys matched ({matched}), warm_start config may be wrong"
    assert skipped_decoder > 0, "no decoder keys in R0 ckpt? expected old decoder.*"
    print(f"[PASS] warm_start: {matched} keys load, {skipped_decoder} decoder keys skip")


def test_router_non_collapse():
    """Run 100 steps on random data with temperature=3.0, check router doesn't collapse.

    User-required: 100-batch router anti-collapse test (not 50).
    """
    cfg = yaml.safe_load(open("cross_dataset/config_p1_schrr_v2_pots_r0.yaml"))
    model = get_model(cfg)
    model.cuda()
    model.train()
    model.decoder.set_router_temperature(3.0)  # early-training temperature

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ce = torch.nn.CrossEntropyLoss()

    for step in range(100):
        images = torch.randn(2, 3, 512, 512, device="cuda")
        masks = torch.randint(0, 6, (2, 512, 512), device="cuda")
        output = model(images)
        loss = ce(output["logits"], masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            bp8 = output["diag"]["schrr_8"]["basis_probs"]
            usage = bp8.mean(dim=(0, 2, 3)).tolist()
            print(f"  step {step}: schrr_8 basis usage = {[f'{u:.3f}' for u in usage]}")

    # Check router usage at step 100
    with torch.no_grad():
        images = torch.randn(2, 3, 512, 512, device="cuda")
        output = model(images)
        for blk_name in ["schrr_8", "schrr_4"]:
            bp = output["diag"][blk_name]["basis_probs"]  # (B, n_bases, H, W)
            usage = bp.mean(dim=(0, 2, 3))  # (n_bases,)
            max_usage = usage.max().item()
            min_usage = usage.min().item()
            entropy = -(bp * torch.log(bp + 1e-12)).sum(dim=1).mean().item()
            print(f"[{blk_name}] basis usage: {[f'{u:.4f}' for u in usage.tolist()]}")
            print(f"[{blk_name}] max_usage={max_usage:.4f}, min_usage={min_usage:.4f}, entropy={entropy:.4f}")
            assert max_usage < 0.95, f"{blk_name} router COLLAPSED (max_usage={max_usage:.4f} >= 0.95)"
            assert entropy > 0.3, f"{blk_name} router entropy too low ({entropy:.4f} <= 0.3)"
    print(f"[PASS] router non-collapse: no basis > 0.95 after 100 steps (user-required test)")


def test_param_groups():
    """Verify build_optimizer_with_groups classifies params correctly."""
    import yaml
    import sys
    sys.path.insert(0, "cross_dataset")
    from train_cross import build_optimizer_with_groups
    cfg = yaml.safe_load(open("cross_dataset/config_p1_schrr_v2_pots_r0.yaml"))
    model = get_model(cfg).cuda()
    # Freeze backbone base
    PEFT_KEYWORDS = ("lora_", "alpha_sem", "alpha_str", "spatial_conv", "tcam")
    for name, param in model.named_parameters():
        if name.startswith("backbone.") and not any(k in name for k in PEFT_KEYWORDS):
            param.requires_grad = False
    opt = build_optimizer_with_groups(model, cfg["lr_groups"], 0.05)
    # Verify lora_tcam group is non-empty
    n_groups = len(opt.param_groups)
    assert n_groups == 4, f"expected 4 param groups, got {n_groups}"
    print(f"[PASS] param groups: 4 groups created")
    # Verify lora_tcam has params (catches PEFT_KEYWORDS bug)
    lora_tcam_n = sum(p.numel() for p in opt.param_groups[3]["params"])
    assert lora_tcam_n > 0, f"lora_tcam group has 0 params — PEFT_KEYWORDS bug"
    print(f"[PASS] lora_tcam group: {lora_tcam_n/1e6:.2f}M params (lr=3.0e-5)")


if __name__ == "__main__":
    print("=" * 60)
    print("SCHRR v2 SMOKE TEST")
    print("=" * 60)
    model = test_full_forward_backward()
    test_gradient_flow(model)
    test_lam_bounded(model)
    test_router_temperature_method(model)
    test_warm_start_loading()
    test_param_groups()
    test_router_non_collapse()
    print("=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
