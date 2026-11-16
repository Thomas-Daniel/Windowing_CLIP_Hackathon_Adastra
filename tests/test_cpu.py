"""CPU checks (login node, ~2 min): windowing arms, Atlas forward at a reduced size, and
that the gradient-cache step of train.py gives the same gradients as a plain backward.

    python -m tests.test_cpu
"""
from __future__ import annotations

import copy
import gc
import os

import numpy as np
import torch
import torch.nn.functional as F

import mcw.gemm_conv3d as G
from mcw.config import load_config
from mcw.loss import clip_loss
from mcw.model import CTClip
from mcw.learned_window import _grad_p_hist, _grad_p_matmul
from mcw.windowing_arms import HU_HI, HU_LO, PILLAR_PRESETS, RampConv3d, build_arm, default_edges


def _reference_curves(lw, x):
    """LearnedWindow output through plain autograd (the original formulation, index-based backward)."""
    e, p = lw.edges, lw.theta.softmax(-1)
    cum = F.pad(p.cumsum(-1), (1, 0))
    xc = x.squeeze(1).clamp(e[0], e[-1])
    j = torch.bucketize(xc, e[1:-1])
    frac = (xc - e[j]) / (e[j + 1] - e[j])
    return (cum[:, j] + p[:, j] * frac).transpose(0, 1)


def test_arms() -> None:
    torch.manual_seed(0)
    hu = torch.linspace(HU_LO, HU_HI, 4096).view(1, 1, 16, 16, 16)
    edges = default_edges()
    assert len(edges) == 81 and edges[0] == -1000 and edges[-1] == 1000

    clip = build_arm("clip")["adapter"](hu)
    assert clip.shape == (1, 11, 16, 16, 16) and torch.allclose(clip[:, 5:6], (hu + 1000) / 2000)
    abdo = build_arm("clip", clip_range=(-120.0, 240.0))["adapter"](hu)            # win5: level 60 / width 360
    assert abdo.shape == (1, 11, 16, 16, 16) and torch.allclose(abdo[:, 3:4], ((hu + 120) / 360).clamp(0, 1))

    # pillar11: presets == rad-vision-engine ANATOMICAL_WINDOWS["CT"] (centre, width) -> (c - w/2, c + w/2)
    rave = [(-600, 1500), (50, 400), (40, 400), (80, 150), (400, 1800), (40, 80), (75, 215), (40, 40), (600, 2800), (50, 350)]
    assert PILLAR_PRESETS == [(c - w / 2, c + w / 2) for c, w in rave]
    fw = build_arm("pillar11")["adapter"](hu)
    assert fw.shape == (1, 11, 16, 16, 16)
    for k, (lo, hi) in enumerate(PILLAR_PRESETS):
        assert torch.allclose(fw[:, k:k + 1], ((hu - lo) / (hi - lo)).clamp(0, 1), atol=1e-6)
    assert torch.allclose(fw[:, 10:], (hu + 1000) / 2000, atol=1e-6)            # minmax of a [-1000, 1000] volume

    # learned: init curves = the presets (clipped to the grid) + the full ramp, up to eps; monotone; [0, 1]
    lw = build_arm("learned")["adapter"]
    xs, ys = lw.curves(n=2001)
    assert (ys[:, 1:] >= ys[:, :-1] - 1e-6).all() and ys[:, 0].abs().max() < 1e-6 and (ys[:, -1] - 1).abs().max() < 1e-5
    for k, (lo, hi) in enumerate(PILLAR_PRESETS + [(HU_LO, HU_HI)]):
        # expected init: bin mass = (1 - eps) * overlap with [lo, hi] / total + eps / M, linear inside each bin
        ov = (torch.minimum(edges[1:], torch.tensor(hi)) - torch.maximum(edges[:-1], torch.tensor(lo))).clamp(min=0)
        mass = 0.999 * ov / ov.sum() + 1e-3 / len(ov)
        cum = F.pad(mass.cumsum(0), (1, 0))
        expect = torch.from_numpy(np.interp(xs.numpy(), edges.numpy(), cum.numpy())).float()
        assert (ys[k] - expect).abs().max() < 1e-5, k
        if lo >= HU_LO and hi <= HU_HI and lo % 25 == 0 and hi % 25 == 0:      # window on the grid: the ramp itself
            assert (ys[k] - ((xs - lo) / (hi - lo)).clamp(0, 1)).abs().max() < 2e-3, k
    # explicit backward == autograd of the original formulation
    with torch.no_grad():
        lw.theta.add_(0.3 * torch.randn_like(lw.theta))
    x = torch.randint(-1100, 1100, (2, 1, 6, 7, 8)).float()
    x[0, 0, 0, 0, :3] = edges[[0, 1, 40]]
    g = torch.randn(2, 11, 6, 7, 8)
    (lw(x) * g).sum().backward()
    g_fast, lw.theta.grad = lw.theta.grad.clone(), None
    (_reference_curves(lw, x) * g).sum().backward()
    assert torch.allclose(lw(x), _reference_curves(lw, x), atol=1e-6)
    assert torch.allclose(g_fast, lw.theta.grad, rtol=1e-4, atol=1e-5), (g_fast - lw.theta.grad).abs().max()
    # histogram backward == the supplied chunked-matmul backward (gradient w.r.t. the bin masses)
    gk = torch.randn(11, 2, 6, 7, 8)
    gm = _grad_p_matmul(gk, x.squeeze(1), lw.edges, chunk=100)
    gh = _grad_p_hist(gk, x.squeeze(1), lw.edges)
    assert torch.allclose(gm, gh, rtol=1e-4, atol=1e-4), (gm - gh).abs().max()

    # warp (per-channel HU shift / scale): identical at init; gradients == autograd of the explicit formula
    lww = build_arm("learned", warp=True)["adapter"]
    assert torch.allclose(lww(x), build_arm("learned")["adapter"](x), atol=1e-6)
    with torch.no_grad():
        lww.theta.add_(0.3 * torch.randn_like(lww.theta))
        lww.tau.normal_(0, 0.3)
        lww.log_scale.normal_(0, 0.2)
    xw = torch.randint(-1100, 1100, (2, 1, 5, 6, 7)).float()
    gw = torch.randn(2, 11, 5, 6, 7)
    (lww(xw) * gw).sum().backward()
    g_fast = [lww.theta.grad.clone(), lww.tau.grad.clone(), lww.log_scale.grad.clone()]
    for prm in (lww.theta, lww.tau, lww.log_scale):
        prm.grad = None
    e = lww.edges
    pw = lww.theta.softmax(-1)
    cum = F.pad(pw.cumsum(-1), (1, 0))
    outs = []
    for k in range(11):
        xr = lww.center[k] + torch.exp(-lww.log_scale[k]) * (xw.squeeze(1) - lww.center[k] - 100 * lww.tau[k])
        xc = xr.clamp(e[0], e[-1])
        j = torch.bucketize(xc.detach(), e[1:-1])
        outs.append(cum[k, j] + pw[k, j] * (xc - e[j]) / (e[j + 1] - e[j]))
    ref = torch.stack(outs).transpose(0, 1)
    assert torch.allclose(lww(xw), ref, atol=1e-5)
    (ref * gw).sum().backward()
    for a, b, name in zip(g_fast, (lww.theta.grad, lww.tau.grad, lww.log_scale.grad), ("theta", "tau", "log_scale")):
        assert torch.allclose(a, b, rtol=1e-3, atol=1e-4), (name, (a - b).abs().max())
    ys = lww.curves(n=2001)[1]
    assert (ys[:, 1:] >= ys[:, :-1] - 1e-6).all(), "warped curves must stay monotone"

    # local: == learned at init (gate = 1); gate gets gradient; shuffle reproducible with a seed
    local = build_arm("local", patch=6)["adapter"]
    learned = build_arm("learned")["adapter"]
    xv = torch.randint(-1000, 1000, (2, 1, 12, 12, 18)).float()
    assert torch.allclose(local(xv), learned(xv), atol=1e-6)
    (local(xv) * torch.randn(2, 11, 12, 12, 18)).sum().backward()
    assert local.gate[-1].weight.grad.abs().sum() > 0 and local.curves.theta.grad.abs().sum() > 0
    shuf = build_arm("local_shuf", patch=6)["adapter"]
    with torch.no_grad():
        shuf.gate[-1].weight.normal_()
    shuf.train()
    shuf.shuffle_seed = 123
    a = shuf(xv)
    b = shuf(xv)
    shuf.shuffle_seed = 124
    c = shuf(xv)
    assert torch.equal(a, b) and not torch.equal(a, c)

    # dense: RampConv3d (GEMM conv) == F.conv3d on the 80 materialised ramps, for the 3D stem geometry
    for s, p in ((3, 0), (2, 1)):
        rc = RampConv3d(edges, 8, stride=s, padding=p)
        xd = torch.randint(-1000, 1000, (1, 1, 12, 12, 12)).float()
        ramps = ((xd - edges[:-1].view(1, -1, 1, 1, 1)) / 25.0).clamp(0, 1)
        w = rc.weight.detach().clone().requires_grad_(True)
        ref = F.conv3d(ramps, w, None, s, p)
        out = rc(xd)
        assert torch.allclose(out, ref, atol=1e-4), (s, p, (out - ref).abs().max())
        go = torch.randn_like(ref)
        out.backward(go)
        ref.backward(go)
        assert torch.allclose(rc.weight.grad, w.grad, rtol=1e-4, atol=1e-3), (s, p)
    print("arms OK")



def test_patch_arms() -> None:
    """Per-patch windows from the HU density (mcw/patch_windows.py)."""
    import os
    import mcw.patch_windows as PW
    torch.manual_seed(0)
    hh = os.path.expandvars("$SCRATCHDIR/merlin_clip_data/hu_hist_80.npz")
    hh = hh if os.path.exists(hh) else None
    x = torch.randint(-1000, 1000, (2, 1, 12, 12, 18)).float()
    # density: per-patch 80-bin fractions, sum 1 (no context), matches a manual histogram
    dens = PW.PatchDensity(patch=6, context=1)(x)
    assert dens.shape == (2, 80, 2, 2, 3) and torch.allclose(dens.sum(1), torch.ones(2, 2, 2, 3))
    ref = torch.histc(x[0, 0, :6, 6:12, 12:18].clamp(-1000, 999.99), bins=80, min=-1000, max=1000) / 216
    assert torch.allclose(dens[0, :, 0, 1, 2], ref)
    # bank backward == autograd of the sigmoid formula (double)
    K = 3
    C = (torch.randn(K, 2, 2, 2, 3, dtype=torch.double) * 200).requires_grad_(True)
    Wd = (torch.rand(K, 2, 2, 2, 3, dtype=torch.double) * 300 + 50).requires_grad_(True)
    xs = x.double().squeeze(1)
    g = torch.randn(K, 2, 12, 12, 18, dtype=torch.double)
    (PW._BankFn.apply(C, Wd, xs, 6) * g).sum().backward()
    gC, gW = C.grad.clone(), Wd.grad.clone()
    C.grad = Wd.grad = None
    up = lambda t: t.repeat_interleave(6, 2).repeat_interleave(6, 3).repeat_interleave(6, 4)
    refo = torch.sigmoid(4 * (xs[None] - up(C)) / up(Wd))
    (refo * g).sum().backward()
    assert torch.allclose(PW._BankFn.apply(C, Wd, xs, 6), refo) and torch.allclose(gC, C.grad) and torch.allclose(gW, Wd.grad)
    # per-patch CDF backward == autograd of the index formula
    P, M = 2 * 2 * 2 * 3, 80
    pm = torch.randn(K, P, M, dtype=torch.double).softmax(-1).requires_grad_(True)
    out = PW._PatchCDFFn.apply(pm, xs, 6, 25.0)
    (out * g).sum().backward()
    gp = pm.grad.clone()
    pm.grad = None
    j, frac = PW._bins(xs, -1000.0, 25.0, M)
    pid = PW._patch_ids(xs.shape, 6, xs.device)
    cum = F.pad(pm.cumsum(-1), (1, 0))
    refc = torch.stack([cum[k][pid, j] + pm[k][pid, j] * frac for k in range(K)])
    (refc * g).sum().backward()
    assert torch.allclose(out, refc) and torch.allclose(gp, pm.grad), (gp - pm.grad).abs().max()
    # zero-init networks: per-patch arms == their global version at init; monotone curves in [0, 1]
    bank, glob = PW.PatchWindowBank(k=8, hu_hist=hh), PW.PatchWindowBank(k=8, per_patch=False, hu_hist=hh)
    assert torch.allclose(bank(x), glob(x), atol=1e-6)
    cdf = PW.PatchCDF(k=8, hu_hist=hh)
    o = cdf(torch.linspace(-1000, 999, 6 * 6 * 6 * 4).view(1, 1, 6, 6, 24))
    assert o.min() >= 0 and o.max() <= 1
    # every trainable part gets a gradient once the networks are off zero
    for mod in (bank, cdf, PW.PatchCDF(k=8, use_net=False, local_eq=True, local_eq_init=0.5, hu_hist=hh)):
        with torch.no_grad():
            for prm in mod.parameters():
                prm.add_(0.01 * torch.randn_like(prm))
        mod.train()
        (mod(x) * torch.randn(2, 8, 12, 12, 18)).sum().backward()
        assert all(prm.grad is not None and prm.grad.abs().sum() > 0 for prm in mod.parameters()), type(mod).__name__
    print(flush=True)
    print(f"patch arms OK (init centres {[round(c) for c in bank.center.tolist()]})")

def test_gemm_conv3d() -> None:
    """GemmConv3d == nn.Conv3d (outputs and gradients) for the stem's stride/padding cases,
    with a small column buffer so several slabs are used."""
    torch.manual_seed(0)
    old, G.CHUNK_ELEMS = G.CHUNK_ELEMS, 4000
    try:
        for cin, cout, s, p, bias in ((3, 5, 1, 1, True), (4, 6, 3, 0, False), (5, 7, 2, 1, False)):
            ref = torch.nn.Conv3d(cin, cout, 3, stride=s, padding=p, bias=bias).double()
            m = torch.nn.Sequential(copy.deepcopy(ref))
            assert G.swap_conv3d(m) == 1 and isinstance(m[0], G.GemmConv3d)
            for fmt in (torch.contiguous_format, torch.channels_last_3d):
                m.zero_grad(set_to_none=True)
                ref.zero_grad(set_to_none=True)
                x0 = torch.randn(2, cin, 12, 13, 14, dtype=torch.double)
                x = x0.clone().contiguous(memory_format=fmt).requires_grad_(True)
                x2 = x0.clone().requires_grad_(True)
                y, yr = m(x), ref(x2)
                assert y.is_contiguous(memory_format=fmt)
                gy = torch.randn_like(yr)
                y.backward(gy.contiguous(memory_format=fmt))
                yr.backward(gy)
                assert torch.allclose(y, yr, atol=1e-10), (s, p, fmt)
                assert torch.allclose(x.grad, x2.grad, rtol=1e-5, atol=1e-6), (s, p, fmt)   # fp32 accumulators
                assert torch.allclose(m[0].weight.grad, ref.weight.grad, rtol=1e-5, atol=1e-5), (s, p, fmt)
        x = torch.randn(1, 16, 6, 6, 6).contiguous(memory_format=torch.channels_last_3d)   # complex128 copy path
        conv = torch.nn.Conv3d(16, 8, 3, padding=1).to(torch.bfloat16)
        m = torch.nn.Sequential(copy.deepcopy(conv))
        G.swap_conv3d(m)
        assert torch.allclose(m(x.bfloat16()).float(), conv(x.bfloat16()).float(), atol=0.05)
    finally:
        G.CHUNK_ELEMS = old
    print("gemm conv3d OK")


def small_cfg(arm: str) -> dict:
    cfg = load_config("configs/base.yaml")
    cfg["windowing"]["arm"] = arm
    cfg["model"]["atlas"] = {"image_size": [96, 96, 96], "stages": [1, 1]}   # 16^3 tokens -> 2 scales
    cfg["model"]["precision"] = "fp32"
    if not os.path.exists(cfg["windowing"]["hu_hist"]):     # off the cluster: flat init histogram
        cfg["windowing"]["hu_hist"] = None
    return cfg


def test_model_and_gradcache() -> None:
    torch.manual_seed(0)
    hu = torch.randint(-1024, 2000, (2, 1, 96, 96, 96)).float()
    for arm in ("clip", "pillar11", "learned", "local", "local_shuf", "dense", "patch_bank", "patch_cdf", "patch_eq",
                "global_bank", "patch_bank_shuf"):
        gc.collect()
        model = CTClip(small_cfg(arm), text_dim=64)
        if arm == "clip":                      # config path of win1 (default range) and win5_abdo
            assert (model.adapter.lo, model.adapter.hi) == (-1000.0, 1000.0)
            cfg5 = small_cfg("clip")
            cfg5["windowing"]["clip_range"] = load_config("configs/win5_abdo.yaml")["windowing"]["clip_range"]
            m5 = CTClip(cfg5, text_dim=64)
            assert (m5.adapter.lo, m5.adapter.hi) == (-120, 240)
        first = model.visual.atlas.patch_embeds["abdomen_ct"].conv_down[0]
        assert isinstance(first, RampConv3d) if arm == "dense" else first.in_channels == model.in_chans
        with torch.no_grad():
            f = model.encode_image(hu)
        print(f"{arm:10s}: image feature {tuple(f.shape)}, first conv {type(first).__name__}, "
              f"Atlas {sum(p.numel() for p in model.visual.parameters()) / 1e6:.2f}M, "
              f"arm {sum(p.numel() for p in model.adapter.parameters())} params")

    # gradient cache == plain backward through all micro-batches (same per-micro-batch BN stats),
    # for the arms with trainable front-ends (local_shuf: same permutation seed in both passes)
    for arm in ("local", "local_shuf", "dense", "local_warp", "patch_bank", "patch_cdf", "patch_bank_shuf"):
        cfg = small_cfg(arm.replace("_warp", ""))
        cfg["windowing"]["warp"] = arm.endswith("_warp")
        model = CTClip(cfg, text_dim=64)
        with torch.no_grad():                  # move the gate off its zero init so it matters
            for prm in model.adapter.parameters():
                prm.add_(0.05 * torch.randn_like(prm))
        model.train()
        ref = copy.deepcopy(model)
        hu4 = torch.randint(-1024, 2000, (4, 1, 96, 96, 96)).float()
        txt = F.normalize(torch.randn(4, 64), dim=-1)
        mb = 2

        feats = []
        for j in range(0, 4, mb):
            ref.set_shuffle_seed(j)
            feats.append(ref.encode_image(hu4[j:j + mb]))
        loss_ref, _ = clip_loss(torch.cat(feats), ref.encode_text(txt), ref.scale())
        loss_ref.backward()

        with torch.no_grad():
            feats = []
            for j in range(0, 4, mb):
                model.set_shuffle_seed(j)
                feats.append(model.encode_image(hu4[j:j + mb]))
            feats = torch.cat(feats)
        feats.requires_grad_(True)
        loss, _ = clip_loss(feats, model.encode_text(txt), model.scale())
        loss.backward()
        for j in range(0, 4, mb):
            model.set_shuffle_seed(j)
            model.encode_image(hu4[j:j + mb]).backward(feats.grad[j:j + mb])

        assert torch.allclose(loss, loss_ref)
        worst = 0.0
        for (n, p), (_, q) in zip(model.named_parameters(), ref.named_parameters()):
            if q.grad is None:
                assert p.grad is None, n
                continue
            rel = float((p.grad - q.grad).norm() / q.grad.norm().clamp_min(1e-12))
            worst = max(worst, rel)
            assert rel < 1e-3, (arm, n, rel)
        trained = [p for n, p in model.named_parameters() if n.startswith("adapter.") or "conv_down.0" in n]
        trained = [p for p in trained if p.requires_grad]
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in trained), arm
        print(f"gradient cache OK for {arm} (worst relative grad diff {worst:.2e})", flush=True)
        del model, ref, feats, loss, loss_ref
        gc.collect()


def test_route() -> None:
    """V1 / V2 bank options and the patch_route arm (mcw/patch_windows.py)."""
    from mcw.patch_windows import PatchWindowBank, PatchWindowRouter, _RouteFn
    torch.manual_seed(0)
    # explicit backward of the routing == autograd, in float64
    P, V, K, M = 5, 27, 3, 4
    alpha = torch.softmax(torch.randn(P, K, M, dtype=torch.float64), -1).requires_grad_(True)
    C = (torch.randn(M, dtype=torch.float64) * 300).requires_grad_(True)
    Wd = (torch.rand(M, dtype=torch.float64) * 400 + 100).requires_grad_(True)
    xp = torch.rand(P, V, dtype=torch.float64) * 2000 - 1000
    g = torch.randn(P, V, K, dtype=torch.float64)
    (_RouteFn.apply(alpha, C, Wd, xp, 2) * g).sum().backward()
    got = [t.grad.clone() for t in (alpha, C, Wd)]
    for t in (alpha, C, Wd):
        t.grad = None
    ref = torch.einsum("pvm,pkm->pvk", torch.sigmoid(4 * (xp[..., None] - C) / Wd), alpha)
    (ref * g).sum().backward()
    for a_, b_ in zip(got, (alpha.grad, C.grad, Wd.grad)):
        assert torch.allclose(a_, b_, rtol=1e-8, atol=1e-10)
    # router module: shapes, identical routing at init, passthrough channel, all parameters reached
    x = torch.randint(-1024, 2000, (1, 1, 48, 48, 48)).float()
    r = PatchWindowRouter(k=4, n_windows=8, patch=6, route_context=3)
    al = r.routing(x)
    assert al.shape == (1, 4, 8, 8, 8, 8) and torch.allclose(al, al[..., :1, :1, :1].expand_as(al))
    out = r(x)
    assert out.shape == (1, 5, 48, 48, 48)
    assert torch.allclose(out[:, -1], ((x[:, 0].clamp(-1000, 1000) + 1000) / 2000))
    with torch.no_grad():
        r.net[-1].weight.normal_(0, 0.05)
    r.train()
    r(x).pow(2).mean().backward()
    assert all(p_.grad is not None and p_.grad.abs().sum() > 0 for p_ in r.parameters()), "router params without grad"
    assert "route_center_std_hu" in r.last_stats
    # router v2: warm-started windows + uniform grid, width floor, hard top-1 forward with softmax gradients
    rh = PatchWindowRouter(k=4, n_windows=8, patch=6, route_context=3, hard=True, width_min=50.0,
                           init_windows={"centers": [-900, -100, 40, 200], "widths": [100, 400, 60, 50]})
    assert torch.allclose(rh.center, torch.tensor([-900., -100, 40, 200, -750, -250, 250, 750]), atol=1e-3)
    assert torch.allclose(rh.width, torch.tensor([100., 400, 60, 51, 1000, 1000, 1000, 1000]), atol=1e-3)
    assert (rh.route_logits.argmax(1) == torch.arange(4)).all()
    with torch.no_grad():
        rh.log_width[0] = -20.0
    assert float(rh.width.min()) >= 50.0
    rh.train()
    out = rh(x)
    xc = x[:, 0].clamp(-1000, 1000)
    assert torch.allclose(out[:, 1], torch.sigmoid(4 * (xc - rh.center[1]) / rh.width[1]), atol=1e-5)
    with torch.no_grad():
        rh.net[-1].weight.normal_(0, 0.05)
    rh(x).pow(2).mean().backward()
    assert all(p_.grad is not None and p_.grad.abs().sum() > 0 for p_ in rh.parameters()), "router v2 params without grad"
    # bank: defaults unchanged, V2 bounds widen, V1 passthrough
    b0, b2 = PatchWindowBank(k=4), PatchWindowBank(k=4, shift_max_hu=1000.0, width_max_log=2.5, passthrough=True)
    b2.load_state_dict(b0.state_dict())
    with torch.no_grad():
        for b_ in (b0, b2):
            b_.net[-1].bias.fill_(50.0)                     # saturate the per-patch shifts
    c0, _ = b0.window_maps(x)
    c2, _ = b2.window_maps(x)
    assert float((c0 - b0.center.view(-1, 1, 1, 1, 1)).max()) <= 300.0 + 1e-3
    assert float((c2 - b2.center.view(-1, 1, 1, 1, 1)).max()) > 900.0
    assert b2(x).shape[1] == 5 and b0(x).shape[1] == 4
    # gradient cache exact through CTClip for the new arms
    for arm, extra in (("patch_route", {"k": 4, "n_windows": 8, "route_context": 3}),
                       ("patch_route", {"k": 4, "n_windows": 8, "route_context": 3, "route_hard": True,
                                        "width_min_hu": 50.0}),
                       ("patch_bank", {"k": 4, "shift_max_hu": 1000.0, "width_max_log": 2.5, "passthrough": True})):
        cfg = small_cfg(arm)
        cfg["windowing"] |= extra
        model = CTClip(cfg, text_dim=64)
        assert model.in_chans == 5
        with torch.no_grad():
            for prm in model.adapter.parameters():
                prm.add_(0.05 * torch.randn_like(prm))
        model.train()
        ref = copy.deepcopy(model)
        hu4 = torch.randint(-1024, 2000, (4, 1, 96, 96, 96)).float()
        txt = F.normalize(torch.randn(4, 64), dim=-1)
        loss_ref, _ = clip_loss(torch.cat([ref.encode_image(hu4[j:j + 2]) for j in (0, 2)]), ref.encode_text(txt), ref.scale())
        loss_ref.backward()
        with torch.no_grad():
            feats = torch.cat([model.encode_image(hu4[j:j + 2]) for j in (0, 2)])
        feats.requires_grad_(True)
        loss, _ = clip_loss(feats, model.encode_text(txt), model.scale())
        loss.backward()
        for j in (0, 2):
            model.encode_image(hu4[j:j + 2]).backward(feats.grad[j:j + 2])
        worst = max(float((p_.grad - q.grad).norm() / q.grad.norm().clamp_min(1e-12))
                    for p_, q in zip(model.parameters(), ref.parameters()) if q.grad is not None)
        assert worst < 1e-3, (arm, worst)
        print(f"{arm} {extra}: gradient cache OK (worst relative grad diff {worst:.2e})", flush=True)
    print("route / V1 / V2 OK", flush=True)


def test_vit() -> None:
    """model.backbone=vit (mcw/vit3d.py): shapes, per-patch arms on the token grid, gradient cache exact."""
    torch.manual_seed(0)
    for arm in ("clip", "patch_bank", "global_bank"):
        cfg = small_cfg(arm)
        cfg["model"] |= {"backbone": "vit", "vit": {"image_size": 96, "patch": 16, "dim": 96, "depth": 2, "heads": 4}}
        model = CTClip(cfg, text_dim=64)
        assert model.visual.patch_embed.in_features == model.in_chans * 16 ** 3
        if arm == "patch_bank":
            assert model.adapter.s == 16                    # per-patch windows on the ViT token grid
            with torch.no_grad():
                for prm in model.adapter.parameters():
                    prm.add_(0.05 * torch.randn_like(prm))
        model.train()
        ref = copy.deepcopy(model)
        hu4 = torch.randint(-1024, 2000, (4, 1, 96, 96, 96)).float()
        txt = F.normalize(torch.randn(4, 64), dim=-1)
        loss_ref, _ = clip_loss(torch.cat([ref.encode_image(hu4[j:j + 2]) for j in (0, 2)]), ref.encode_text(txt), ref.scale())
        loss_ref.backward()
        with torch.no_grad():
            feats = torch.cat([model.encode_image(hu4[j:j + 2]) for j in (0, 2)])
        feats.requires_grad_(True)
        loss, _ = clip_loss(feats, model.encode_text(txt), model.scale())
        loss.backward()
        for j in (0, 2):
            model.encode_image(hu4[j:j + 2]).backward(feats.grad[j:j + 2])
        assert torch.allclose(loss, loss_ref)
        worst = max(float((p.grad - q.grad).norm() / q.grad.norm().clamp_min(1e-12))
                    for p, q in zip(model.parameters(), ref.parameters()) if q.grad is not None)
        assert worst < 1e-3, (arm, worst)
        print(f"vit {arm:11s}: feature {tuple(feats.shape)}, ViT {sum(p.numel() for p in model.visual.parameters()) / 1e6:.2f}M, "
              f"gradient cache OK (worst relative grad diff {worst:.2e})", flush=True)


def test_bank_opts() -> None:
    """PatchWindowBank options of the noise / context / hybrid campaign: width floor, smoothed input, per-scan
    context, uniform init, fixed pillar11 channels."""
    from mcw.patch_windows import PatchWindowBank
    from mcw.windowing_arms import PILLAR_PRESETS, FixedWindows
    torch.manual_seed(0)
    x = torch.randint(-1024, 2000, (2, 1, 48, 48, 48)).float()
    xc = x.clamp(-1000, 1000)
    # width floor holds even with the per-patch width scale at its minimum
    b = PatchWindowBank(k=4, width_min=200.0)
    with torch.no_grad():
        b.log_width.fill_(-10.0)
        b.net[-1].bias[4:].fill_(-50.0)
    _, Wd = b.window_maps(x)
    assert float(Wd.min()) >= 200.0
    # uniform init
    u = PatchWindowBank(k=4, init="uniform")
    assert torch.allclose(u.center, torch.tensor([-750., -250, 250, 750]), atol=1e-3)
    assert torch.allclose(u.width, torch.full((4,), 1000.0), atol=1e-2)
    # smoothing: the windows read the 3^3 box mean, the passthrough ramp stays raw
    sm = PatchWindowBank(k=4, init="uniform", smooth=3, passthrough=True)
    out = sm(x)
    xs = F.avg_pool3d(xc, 3, stride=1, padding=1, count_include_pad=False)
    assert torch.allclose(out[:, 0], torch.sigmoid(4 * (xs[:, 0] + 750) / 1000), atol=1e-5)
    assert torch.allclose(out[:, -1], (xc[:, 0] + 1000) / 2000)
    # per-scan context: zero-init = no change; then one shift per scan, constant over the volume
    sc = PatchWindowBank(k=4, per_patch=False, scan_context=True)
    ref = PatchWindowBank(k=4, per_patch=False)
    assert torch.allclose(sc(x), ref(x))
    with torch.no_grad():
        sc.scan_net[-1].weight.normal_(0, 0.5)
    C, _ = sc.window_maps(x)
    assert torch.allclose(C, C[..., :1, :1, :1].expand_as(C)) and not torch.allclose(C[:, 0], C[:, 1])
    sc.train()
    sc(x).pow(2).mean().backward()
    assert all(p_.grad is not None and p_.grad.abs().sum() > 0 for p_ in sc.parameters()), "scan bank params without grad"
    assert "scan_shift_abs_hu" in sc.last_stats
    nz = PatchWindowBank.scan_noise(xc)
    smooth_vol = torch.full_like(xc, 40.0) + torch.randn_like(xc) * 5
    assert float(PatchWindowBank.scan_noise(smooth_vol).mean()) < float(nz.mean())
    # hybrid: the 11 pillar11 channels first, then the learned ones
    hy = PatchWindowBank(k=4, fixed_windows=True)
    assert hy.out_chans == 4 + len(PILLAR_PRESETS) + 1
    out = hy(x)
    assert out.shape[1] == hy.out_chans and torch.allclose(out[:, :11], FixedWindows(PILLAR_PRESETS)(xc))
    # gradient cache exact through CTClip
    for arm, extra in (("patch_bank", {"k": 4, "width_min_hu": 200.0, "smooth": 3, "passthrough": True,
                                       "scan_context": True}),
                       ("patch_bank", {"k": 4, "fixed_windows": True, "window_init": "uniform"})):
        cfg = small_cfg(arm)
        cfg["windowing"] |= extra
        model = CTClip(cfg, text_dim=64)
        with torch.no_grad():
            for prm in model.adapter.parameters():
                prm.add_(0.05 * torch.randn_like(prm))
        model.train()
        ref = copy.deepcopy(model)
        hu4 = torch.randint(-1024, 2000, (4, 1, 96, 96, 96)).float()
        txt = F.normalize(torch.randn(4, 64), dim=-1)
        loss_ref, _ = clip_loss(torch.cat([ref.encode_image(hu4[j:j + 2]) for j in (0, 2)]), ref.encode_text(txt), ref.scale())
        loss_ref.backward()
        with torch.no_grad():
            feats = torch.cat([model.encode_image(hu4[j:j + 2]) for j in (0, 2)])
        feats.requires_grad_(True)
        loss, _ = clip_loss(feats, model.encode_text(txt), model.scale())
        loss.backward()
        for j in (0, 2):
            model.encode_image(hu4[j:j + 2]).backward(feats.grad[j:j + 2])
        worst = max(float((p_.grad - q.grad).norm() / q.grad.norm().clamp_min(1e-12))
                    for p_, q in zip(model.parameters(), ref.parameters()) if q.grad is not None)
        assert worst < 1e-3, (extra, worst)
        print(f"{arm} {extra}: in_chans {model.in_chans}, gradient cache OK (worst relative grad diff {worst:.2e})", flush=True)
    print("bank options OK", flush=True)


def test_win_cdf() -> None:
    """Fixed arms of the "not per patch" campaign: named preset subset (windows) and global HU CDF (cdf)."""
    from mcw.windowing_arms import WINDOW_PRESETS
    hu = torch.linspace(HU_LO, HU_HI, 4096).view(1, 1, 16, 16, 16)
    w4 = load_config("configs/mix_win4.yaml")["windowing"]
    arm = build_arm("windows", windows=w4["windows"], minmax=w4["minmax"])
    out = arm["adapter"](hu)
    assert arm["in_chans"] == 4 and out.shape == (1, 4, 16, 16, 16)
    full = build_arm("pillar11")["adapter"](hu)
    for c, name in enumerate(w4["windows"]):          # the same channels as pillar11
        assert torch.allclose(out[:, c], full[:, PILLAR_PRESETS.index(WINDOW_PRESETS[name])]), name
    assert build_arm("windows", windows=["lung"])["in_chans"] == 2
    # cdf: monotone, 0 at -1000, 1 at 1000, = the body-HU CDF at the bin edges, k copies
    hist = load_config("configs/mix_hu_cdf.yaml")["windowing"]["hu_hist"]
    if not os.path.exists(hist):                            # the cdf arm needs the real body-HU histogram
        print(f"windows OK; cdf skipped ({hist} not found)", flush=True)
        return
    h = np.load(hist)
    arm = build_arm("cdf", hu_hist=hist)
    y = arm["adapter"](hu)
    assert arm["in_chans"] == 11 and y.shape == (1, 11, 16, 16, 16) and torch.equal(y[:, 0], y[:, 10])
    v = y[0, 0].flatten()
    assert (v[1:] >= v[:-1] - 1e-6).all() and abs(float(v[0])) < 1e-6 and abs(float(v[-1]) - 1) < 1e-5
    e = torch.as_tensor(h["edges"]).float().view(1, 1, -1, 1, 1)
    cdf = np.concatenate([[0], np.cumsum(h["hist_body"])]) / h["hist_body"].sum()
    assert torch.allclose(arm["adapter"](e)[0, 0].flatten(), torch.as_tensor(cdf).float(), atol=1e-5)
    assert torch.allclose(arm["adapter"](torch.tensor([-1500.0, 3000.0]).view(1, 1, 2, 1, 1)).flatten()[::11],
                          torch.tensor([0.0, 1.0]))
    # through CTClip: stem width, finite features
    for cfg_name in ("mix_win4", "mix_hu_cdf", "mix_dense"):
        cfg = small_cfg("clip")
        cfg["windowing"] = load_config(f"configs/{cfg_name}.yaml")["windowing"]
        model = CTClip(cfg, text_dim=64).eval()
        first = model.visual.atlas.patch_embeds["abdomen_ct"].conv_down[0]
        assert isinstance(first, RampConv3d) if cfg_name == "mix_dense" else first.in_channels == model.in_chans
        with torch.no_grad():
            f = model.encode_image(torch.randint(-1024, 2000, (1, 1, 96, 96, 96)).float())
        assert torch.isfinite(f).all()
        print(f"{cfg_name}: arm {model.arm}, in_chans {model.in_chans}, features {tuple(f.shape)}", flush=True)
    print("windows / cdf arms OK", flush=True)


def test_select() -> None:
    """window_select (mcw/patch_windows.py WindowSelect): softmax over the 11 pillar11 channels before the stem."""
    from mcw.patch_windows import WindowSelect
    torch.manual_seed(0)
    x = torch.randint(-1024, 2000, (2, 1, 48, 48, 48)).float()
    full = build_arm("pillar11")["adapter"](x.clamp(HU_LO, HU_HI))                      # (2, 11, ...)
    # init: n_out = 1 is the uniform mean of the 11 channels (global and per patch), n_out = 3 leans as documented
    for per_patch in (False, True):
        sel = WindowSelect(n_out=1, per_patch=per_patch, route_context=3)
        assert torch.allclose(sel(x), full.mean(1, keepdim=True), atol=1e-6)
    s3 = WindowSelect(n_out=3, per_patch=True, route_context=3)
    w0 = s3.weights(x)
    assert w0.shape == (2, 3, 11, 8, 8, 8)
    assert [PILLAR_PRESETS[int(i)] for i in w0[0, :, :, 0, 0, 0].argmax(1)] == [PILLAR_PRESETS[i] for i in (0, 1, 4)]
    assert torch.allclose(s3(x), torch.einsum("bcxyz,oc->boxyz", full, w0[0, :, :, 0, 0, 0]), atol=1e-6)
    # per-patch weights: constant within a 6^3 patch, output = that patch's mixture; every parameter gets a gradient
    with torch.no_grad():
        s3.net[-1].weight.normal_(0, 0.5)
    s3.train()
    out, a = s3(x), s3.weights(x)
    ref = torch.einsum("cxyz,oc->oxyz", full[1, :, 6:12, 12:18, 18:24], a[1, :, :, 1, 2, 3])   # patch (1, 2, 3)
    assert torch.allclose(out[1, :, 6:12, 12:18, 18:24], ref, atol=1e-5)
    assert a[1, :, :, 1, 2, 3].sub(a[1, :, :, 1, 2, 4]).abs().max() > 1e-4, "per-patch weights do not vary"
    out.pow(2).mean().backward()
    assert all(q.grad is not None and q.grad.abs().sum() > 0 for q in s3.parameters()), "select params without grad"
    assert {"select_maxw_mean", "select_top2", "select_switched"} <= set(s3.last_stats)
    assert all(isinstance(v, float) for v in s3.last_stats.values())
    # through CTClip with the mix192 configs: stem width = n_out, gradient cache exact
    for name in ("mix192_select_global", "mix192_select_patch", "mix192_select_patch3"):
        cfg = small_cfg("clip")
        cfg["windowing"] = load_config(f"configs/{name}.yaml")["windowing"] | {"route_context": 3}
        model = CTClip(cfg, text_dim=64)
        first = model.visual.atlas.patch_embeds["abdomen_ct"].conv_down[0]
        assert first.in_channels == model.in_chans == cfg["windowing"]["select_out"]
        with torch.no_grad():
            for prm in model.adapter.parameters():
                prm.add_(0.3 * torch.randn_like(prm))
        model.train()
        ref = copy.deepcopy(model)
        hu4 = torch.randint(-1024, 2000, (4, 1, 96, 96, 96)).float()
        txt = F.normalize(torch.randn(4, 64), dim=-1)
        loss_ref, _ = clip_loss(torch.cat([ref.encode_image(hu4[j:j + 2]) for j in (0, 2)]), ref.encode_text(txt), ref.scale())
        loss_ref.backward()
        with torch.no_grad():
            feats = torch.cat([model.encode_image(hu4[j:j + 2]) for j in (0, 2)])
        feats.requires_grad_(True)
        loss, _ = clip_loss(feats, model.encode_text(txt), model.scale())
        loss.backward()
        for j in (0, 2):
            model.encode_image(hu4[j:j + 2]).backward(feats.grad[j:j + 2])
        worst = max(float((p_.grad - q.grad).norm() / q.grad.norm().clamp_min(1e-12))
                    for p_, q in zip(model.parameters(), ref.parameters()) if q.grad is not None)
        assert worst < 1e-3, (name, worst)
        assert model.adapter.select_logits.grad.abs().sum() > 0
        print(f"{name}: in_chans {model.in_chans}, gradient cache OK (worst relative grad diff {worst:.2e})", flush=True)
    print("window_select OK", flush=True)


def test_fixed_bank() -> None:
    """freeze_windows (configs mix_fixed_reg{4,8,16}): global_bank + uniform init + frozen windows = K fixed regular
    sigmoid windows; no window parameter is trained, the rest of the model is."""
    from mcw.windowing_arms import param_groups
    torch.manual_seed(0)
    for k in (4, 8, 16):
        cfg = small_cfg("global_bank")
        cfg["windowing"] |= {"k": k, "window_init": "uniform", "freeze_windows": True}
        model = CTClip(cfg, text_dim=64)
        bank = model.adapter
        sp = 2000.0 / k
        assert torch.allclose(bank.center, -1000 + sp * (torch.arange(k) + 0.5), atol=1e-3)
        assert torch.allclose(bank.width, torch.full((k,), 2 * sp), rtol=1e-5)
        trainable = [n for n, p_ in bank.named_parameters() if p_.requires_grad]
        assert not trainable, trainable
        groups = param_groups(model, 1e-3, 0.05, window_lr=2.5e-2)
        ids = {id(p_) for g in groups for p_ in g["params"]}
        assert id(bank.center100) not in ids and id(bank.log_width) not in ids and len(groups) == 2
        x = torch.randint(-1024, 2000, (2, 1, 96, 96, 96)).float()
        out = bank(x)
        xc = x.clamp(-1000, 1000)[:, 0]
        ref = torch.stack([torch.sigmoid(4 * (xc - c) / w) for c, w in zip(bank.center, bank.width)], 1)
        assert out.shape[1] == k and torch.allclose(out, ref, atol=1e-5)
        model.train()
        opt = torch.optim.AdamW(groups)
        before = {n: p_.detach().clone() for n, p_ in model.named_parameters()}
        txt = F.normalize(torch.randn(2, 64), dim=-1)
        loss, _ = clip_loss(model.encode_image(x), model.encode_text(txt), model.scale())
        loss.backward()
        opt.step()
        moved = {n for n, p_ in model.named_parameters() if not torch.equal(p_, before[n])}
        assert not any(n.startswith("adapter.") for n in moved), moved
        assert any(n.startswith("visual.") for n in moved)
        print(f"fixed bank k={k}: centres {bank.center[0]:.0f}..{bank.center[-1]:.0f} HU, width {bank.width[0]:.0f} HU, "
              f"frozen, {len(moved)} other tensors trained OK", flush=True)
    print("fixed bank OK", flush=True)



def test_downsample() -> None:
    """data.downsample (mix192r_* runs): mcw/data.py block_mean_int16 = mean of every f^3 block rounded half up to
    int16, for the full HU range (negative means included), and the same values through the dataset."""
    from mcw.data import block_mean_int16
    rng = np.random.default_rng(0)
    v = rng.integers(-1024, 3072, size=(12, 16, 20)).astype(np.int16)
    v[:2, :2, :2] = [[[-1, -2], [-1, -2]], [[-1, -2], [-1, -2]]]      # block mean -1.5 -> -1 (half up)
    for f in (2, 4):
        d, h, w = (n // f for n in v.shape)
        ref = np.floor(v.astype(np.float64).reshape(d, f, h, f, w, f).mean(axis=(1, 3, 5)) + 0.5).astype(np.int16)
        out = block_mean_int16(v, f)
        assert out.dtype == np.int16 and out.shape == (d, h, w) and np.array_equal(out, ref), f
    assert block_mean_int16(v, 2)[0, 0, 0] == -1
    try:
        block_mean_int16(v[:, :, :19], 2)
        raise AssertionError("odd shape accepted")
    except ValueError:
        pass
    print("downsample OK (block mean, round half up, int16)", flush=True)



if __name__ == "__main__":
    import sys
    tests = {"arms": test_arms, "patch": test_patch_arms, "gemm": test_gemm_conv3d, "model": test_model_and_gradcache,
             "vit": test_vit, "route": test_route, "bank_opts": test_bank_opts,
             "win_cdf": test_win_cdf, "select": test_select, "fixed": test_fixed_bank,
             "downsample": test_downsample}
    for name in sys.argv[1:] or list(tests):      # e.g. python -m tests.test_cpu patch gemm
        tests[name]()
