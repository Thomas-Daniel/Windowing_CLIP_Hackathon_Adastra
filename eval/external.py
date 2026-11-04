"""Public CT-report models on our Merlin test protocol: Jolia (raidium/Jolia) and Pillar-0 Abdomen CT
(YalaLab/Pillar0-AbdomenCT), downloaded to $SCRATCHDIR/hf_models (both loaded as packages from there).

Each model gets its released preprocessing and text pipeline:
  jolia    centre crop 192^3 of the 1.5 mm volume, its 11 RAVE windows (jolia_windowing, bf16 ops as
           in its preprocessor), Atlas 22M. Image embedding = CLS (576, no projection). Text =
           Qwen3-Embedding-8B last token of the report truncated at 512 tokens, raw (the linear head
           has a bias: raw = L2-normalised x stored norm), through its text projection.
  pillar0  384^3 1.5 mm volume, the same 11 RAVE windows, Atlas small. Image embedding = its 1152-d
           CLIP output. Text = Qwen3-Embedding-8B masked mean over 512 tokens, L2-normalised
           (pillar-pretrain's text cache), through its MLP head.
Volumes are our 384^3 1.5 mm exports (RAS canonical -> (Z, Y, X)). Neither model documents the axis
convention it was trained with, so `orient` embeds N val volumes under the 16 flips / transposes
that keep the axial plane and `select` keeps the one with the best val retrieval (the choice can
only favour the model; test is never looked at).

--data multi evaluates on the multi-source test sets (Merlin, CT-RATE, INSPECT, CancerVerse;
eval/benchmarks.py tasks) with the orientation chosen on Merlin, and reuses the Merlin embeddings
already extracted by --data merlin (same files, same model, same orientation).
pillar0_chest (YalaLab/Pillar0-ChestCT, gated: access must be granted first) uses Pillar-0's code.

Phases (scripts/submit_external.slurm runs them all):
  orient   GPU, one rank per GCD   -> <out>/<model>/orient/rank*.npz
  select   CPU                     -> <out>/<model>/orientation.json
  extract  GPU, one rank per GCD   -> <out>/<model>/emb/rank*_part*.npz (train + test), resumable
  report   CPU                     -> <out>/<model>/summary.json, linear_probe.csv, zeroshot.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.evaluate import linear_probe  # noqa: E402
from mcw.config import load_config  # noqa: E402
from mcw.gemm_conv3d import swap_conv3d  # noqa: E402

HF = Path(os.path.expandvars("$SCRATCHDIR/hf_models"))
REPO = Path(__file__).resolve().parents[1]
# (flip D, flip H, flip W, transpose H/W) on the (Z, Y, X) volume
ORIENTS = [(fd, fh, fw, tr) for tr in (0, 1) for fd in (0, 1) for fh in (0, 1) for fw in (0, 1)]
# every axis permutation x flips (the 48 symmetries of the cube): (perm0, perm1, perm2, flip0, flip1, flip2)
ORIENTS48 = [(*p, *f) for p in __import__("itertools").permutations((0, 1, 2))
             for f in __import__("itertools").product((0, 1), repeat=3)]


def orient(v: torch.Tensor, o) -> torch.Tensor:
    if len(o) == 6:                                   # ORIENTS48: permute the axes, then flip
        v = v.permute(*o[:3])
        dims = [d for d, f in zip((-3, -2, -1), o[3:]) if f]
        return torch.flip(v, dims) if dims else v
    fd, fh, fw, tr = o
    if tr:
        v = v.transpose(-1, -2)
    dims = [d for d, f in zip((-3, -2, -1), (fd, fh, fw)) if f]
    return torch.flip(v, dims) if dims else v


def _pkg(name: str, target: Path) -> None:
    """Expose a downloaded HF repo (relative imports) as an importable package under HF/pkgs."""
    pk = HF / "pkgs"
    pk.mkdir(exist_ok=True)
    link = pk / name
    if not link.exists():
        link.symlink_to(target)
    (target / "__init__.py").touch()
    if str(pk) not in sys.path:
        sys.path.insert(0, str(pk))


def rave_windows(v: torch.Tensor, dtype) -> torch.Tensor:
    """(D, H, W) HU -> (1, 11, D, H, W) in [0, 1]: lung, mediastinum, abdomen, liver, bone, brain,
    subdural, stroke, temporal bone, soft tissue, per-volume minmax (Jolia's RAVE implementation)."""
    from jolia_hf.jolia_windowing import batch_apply_windowing_vectorized
    return batch_apply_windowing_vectorized(v.contiguous()[None, None], windows="all", modality="CT",
                                            torch_operating_dtype=dtype).float()


class Jolia:
    name, text_pool, n_named, orients = "jolia", "last", 102, ORIENTS

    def __init__(self, device: str):
        _pkg("jolia_hf", HF / "Jolia")
        from jolia_hf.modeling_jolia import JoliaModel
        self.m = JoliaModel.from_pretrained(str(HF / "Jolia")).eval()
        swap_conv3d(self.m)
        self.m.to(device)
        self.device = device

    def image_input(self, v: torch.Tensor) -> torch.Tensor:
        s = [max((n - 192) // 2, 0) for n in v.shape]                    # Crop3D, eval: centre
        return rave_windows(v[s[0]:s[0] + 192, s[1]:s[1] + 192, s[2]:s[2] + 192], torch.bfloat16)

    @torch.no_grad()
    def embed(self, x: torch.Tensor, full: bool) -> dict:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            if full:
                cls, organ = self.m.forward_with_queries(x)
            else:
                cls, organ = self.m(x).pooler_output, None
        cls = cls.float()
        out = {"clip": F.normalize(cls, dim=-1), "feat": cls}
        if organ is not None:
            out["organ"] = organ[:, : self.n_named].float()
        return out

    @torch.no_grad()
    def text(self, emb: np.ndarray, norm: np.ndarray) -> torch.Tensor:
        raw = torch.from_numpy(emb).float() * torch.from_numpy(norm).float()[:, None]
        return self.m.encode_text(raw.to(self.device), normalize=True).float().cpu()


class Pillar0:
    name, text_pool, repo, pkg, orients = "pillar0", "mean", "Pillar0-AbdomenCT", "pillar0_abd", ORIENTS

    def __init__(self, device: str):
        import importlib
        _pkg("jolia_hf", HF / "Jolia")                                     # windowing implementation
        p = HF / self.repo
        if not (p / "model.safetensors").exists():
            raise FileNotFoundError(f"{p}/model.safetensors missing (gated repo: request access, then download)")
        _pkg(self.pkg, p)
        from safetensors.torch import load_file
        cfg_mod = importlib.import_module(f"{self.pkg}.config_clip_mmatlas")
        mdl_mod = importlib.import_module(f"{self.pkg}.modeling_clip_mmatlas")
        cfg = cfg_mod.CLIPMultimodalAtlasConfig.from_pretrained(str(p))
        self.modality = next(iter(cfg.vision_cfg["model_config"]["modalities"]))   # abdomen_ct / chest_ct
        m = mdl_mod.CLIPMultimodalAtlas(cfg).eval()
        m.model.load_state_dict({k[len("model."):]: v for k, v in load_file(str(p / "model.safetensors")).items()})
        self.m = m.model
        swap_conv3d(self.m)
        self.m.to(device)
        self.device = device

    def image_input(self, v: torch.Tensor) -> torch.Tensor:
        return rave_windows(v, torch.float32)

    @torch.no_grad()
    def embed(self, x: torch.Tensor, full: bool) -> dict:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            f = self.m.encode_image({self.modality: x}, normalize=False)
        f = f.float()
        return {"clip": F.normalize(f, dim=-1), "feat": f}

    @torch.no_grad()
    def text(self, emb: np.ndarray, norm: np.ndarray) -> torch.Tensor:
        return self.m.encode_text(torch.from_numpy(emb).float().to(self.device), normalize=True).float().cpu()


class Pillar0Chest(Pillar0):
    """rad-vision-engine ct_chest.yaml: 1.25 mm isotropic, centre 256 x 256 in-plane, middle 256 slices (a
    320 mm cube; from our 1.5 mm volume: centre 213 voxels resized to 256). Its 8 x 8 x 4 patches make
    the axis order matter, so its orientation search covers all 48 axis permutations / flips."""
    name, repo, pkg, orients = "pillar0_chest", "Pillar0-ChestCT", "pillar0_chest", ORIENTS48

    def image_input(self, v: torch.Tensor) -> torch.Tensor:
        n = int(round(320 / 1.5))
        s = [max((k - n) // 2, 0) for k in v.shape]
        v = v[s[0]:s[0] + n, s[1]:s[1] + n, s[2]:s[2] + n]
        v = F.interpolate(v[None, None], size=(256, 256, 256), mode="trilinear", align_corners=False)[0, 0]
        return rave_windows(v, torch.float32)


MODELS = {"jolia": Jolia, "pillar0": Pillar0, "pillar0_chest": Pillar0Chest}


# ----------------------------------------------------------------------------- helpers
def rank_env():
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(local)
    return rank, world, device


def manifest(cfg, args=None) -> pd.DataFrame:
    man = pd.read_csv(args.manifest if args is not None and args.manifest else cfg["data"]["manifest"])
    man["study_id"] = man["study_id"].astype(str)
    return man.sort_values("study_id").reset_index(drop=True)


class TextCache:
    """Qwen3-Embedding cache written by scripts/embed_reports.py (L2-normalised rows + norms)."""

    def __init__(self, d: str):
        d = Path(d)
        self.ids = (d / "ids.txt").read_text().split()
        self.row = {s: i for i, s in enumerate(self.ids)}
        self.d = d
        self.prompts = np.load(d / "prompts.npz")

    def get(self, pool: str, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        emb = np.load(self.d / f"{pool}.npy", mmap_mode="r")
        norm = np.load(self.d / f"{pool}_norm.npy", mmap_mode="r")
        idx = [self.row[s] for s in ids]
        return np.asarray(emb[idx], dtype=np.float32), np.asarray(norm[idx], dtype=np.float32)


class Volumes(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return torch.from_numpy(np.load(self.paths[i]))


def loader(paths):
    return torch.utils.data.DataLoader(Volumes(paths), batch_size=None, num_workers=4, prefetch_factor=2)


def recall(img: torch.Tensor, txt: torch.Tensor, ks=(1, 5, 10)) -> dict:
    sim = img @ txt.T
    gt = torch.arange(len(img))
    ri = (sim > sim[gt, gt][:, None]).sum(1)
    rt = (sim.T > sim[gt, gt][:, None]).sum(1)
    return {**{f"R@{k} i2t": float((ri < k).float().mean()) for k in ks},
            **{f"R@{k} t2i": float((rt < k).float().mean()) for k in ks}}


# ----------------------------------------------------------------------------- phases
def phase_orient(args, cfg):
    rank, world, device = rank_env()
    man = manifest(cfg, args)
    tc = TextCache(args.text_cache)
    val = man[(man["split"] == "val") & man["study_id"].isin(tc.row)]
    if args.orient_source:
        val = val[val["source"] == args.orient_source]
    if args.orient_domain:
        val = val[val["domain"] == args.orient_domain]
    val = val.sample(n=min(args.n, len(val)), random_state=0).sort_values("study_id")
    mine = val.iloc[rank::world]
    models = [MODELS[m](device) for m in args.models]
    embs = {m.name: [] for m in models}
    t0 = time.time()
    for k, vol in enumerate(loader(mine["path"].tolist())):
        v = vol.to(device, non_blocking=True).float()
        for m in models:
            embs[m.name].append(np.stack([m.embed(m.image_input(orient(v, o)), full=False)["clip"][0].cpu().numpy()
                                          for o in m.orients]))
        if rank == 0 and (k + 1) % 4 == 0:
            print(f"[orient rank 0] {k + 1}/{len(mine)} volumes ({time.time() - t0:.0f}s)", flush=True)
    for m in models:
        d = Path(args.out) / m.name / "orient"
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"rank{rank:03d}.npz", ids=np.array(mine["study_id"].tolist()), clip=np.stack(embs[m.name]))
    print(f"[orient rank {rank}] done: {len(mine)} volumes x {[len(m.orients) for m in models]} orientations "
          f"in {time.time() - t0:.0f}s", flush=True)


def phase_select(args, cfg):
    tc = TextCache(args.text_cache)
    for name in args.models:
        m = MODELS[name]("cpu")
        parts = [np.load(f) for f in sorted(glob.glob(str(Path(args.out) / name / "orient" / "rank*.npz")))]
        ids = [str(s) for p in parts for s in p["ids"]]
        clip = torch.from_numpy(np.concatenate([p["clip"] for p in parts]))          # (N, 16, D)
        txt = F.normalize(m.text(*tc.get(m.text_pool, ids)), dim=-1)
        rows = []
        for oi, o in enumerate(m.orients):
            r = recall(clip[:, oi], txt)
            rows.append({"orient": list(o), "score": (r["R@5 i2t"] + r["R@5 t2i"]) / 2, **r})
        rows.sort(key=lambda r: -r["score"])
        for r in rows:
            print(f"[select {name}] {'perm + flips' if len(r['orient']) == 6 else 'flip D/H/W, transpose'} {r['orient']}: R@1 {r['R@1 i2t']:.3f}/{r['R@1 t2i']:.3f} "
                  f"R@5 {r['R@5 i2t']:.3f}/{r['R@5 t2i']:.3f} (val pool {len(ids)})", flush=True)
        (Path(args.out) / name / "orientation.json").write_text(json.dumps(
            {"selected": rows[0]["orient"], "n_val": len(ids), "candidates": rows}, indent=1))


def orientation(args, name: str) -> tuple:
    """The model's selected orientation: <out>/<model>/orientation.json, else the --orient-from run's."""
    for d in (args.out, args.orient_from):
        f = Path(d) / name / "orientation.json" if d else None
        if f is not None and f.exists():
            return tuple(json.loads(f.read_text())["selected"])
    raise FileNotFoundError(f"no orientation.json for {name} in {args.out} or {args.orient_from}: run orient + select")


def reused_ids(args, name: str) -> set:
    files = glob.glob(str(Path(args.reuse) / name / "emb" / "rank*_part*.npz")) if args.reuse else []
    return {str(s) for f in files if not f.endswith(".tmp.npz") for s in np.load(f)["ids"]}   # ids only (npz is lazy)


def phase_extract(args, cfg):
    rank, world, device = rank_env()
    man = manifest(cfg, args)
    man = man[man["split"].isin(args.splits)]
    models = [MODELS[m](device) for m in args.models]
    skip = set.intersection(*[reused_ids(args, m.name) for m in models]) if args.reuse else set()
    man = man[~man["study_id"].isin(skip)]                  # already embedded by the --reuse run (same files)
    mine = man.iloc[rank::world].reset_index(drop=True)
    if rank == 0:
        print(f"[extract] {len(man)} volumes to embed ({len(skip)} reused from {args.reuse})", flush=True)
    sel = {m.name: orientation(args, m.name) for m in models}
    for m in models:
        (Path(args.out) / m.name / "emb").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for c0 in range(0, len(mine), args.part):
        names = {m.name: Path(args.out) / m.name / "emb" / f"rank{rank:03d}_part{c0 // args.part:03d}.npz" for m in models}
        if all(p.exists() for p in names.values()):
            continue
        chunk = mine.iloc[c0:c0 + args.part]
        acc = {m.name: {} for m in models}
        for vol in loader(chunk["path"].tolist()):
            v = vol.to(device, non_blocking=True).float()
            for m in models:
                e = m.embed(m.image_input(orient(v, sel[m.name])), full=True)
                for k, t in e.items():
                    acc[m.name].setdefault(k, []).append(t[0].cpu().numpy())
        for m in models:
            arrays = {k: np.stack(v).astype(np.float16 if k in ("clip", "organ") else np.float32) for k, v in acc[m.name].items()}
            tmp = names[m.name].with_suffix(".tmp.npz")
            np.savez(tmp, ids=np.array(chunk["study_id"].tolist()), **arrays)
            os.replace(tmp, names[m.name])
        print(f"[extract rank {rank}] {min(c0 + args.part, len(mine))}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[extract rank {rank}] done in {time.time() - t0:.0f}s", flush=True)


def load_emb(out: str, name: str, reuse: str | None = None) -> dict:
    files = [f for f in sorted(glob.glob(str(Path(out) / name / "emb" / "rank*_part*.npz"))) if not f.endswith(".tmp.npz")]
    if reuse:
        files += [f for f in sorted(glob.glob(str(Path(reuse) / name / "emb" / "rank*_part*.npz"))) if not f.endswith(".tmp.npz")]
    parts = [np.load(f) for f in files]
    keys = [k for k in parts[0].files if all(k in p.files for p in parts)]
    d = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    _, first = np.unique(d["ids"], return_index=True)       # sorted by id, duplicates (reused + new) dropped
    return {k: v[first] for k, v in d.items()}


def report_multi(args, cfg):
    """Per-source retrieval, long-form zero-shot (findings) and linear probes (all tasks), eval/benchmarks.py."""
    from eval.benchmarks import build_labels, linear_probes, retrieval_suite, zero_shot_long
    man = manifest(cfg, args)
    split = dict(zip(man["study_id"], man["split"]))
    lab_csv = Path(args.manifest).parent / "eval_labels.csv"
    labels = pd.read_csv(lab_csv, dtype={"study_id": str}) if lab_csv.exists() else build_labels(man)
    tc = TextCache(args.text_cache)
    for name in args.models:
        m = MODELS[name]("cpu")
        e = load_emb(args.out, name, args.reuse)
        ids = [str(s) for s in e["ids"]]
        res = {"model": name, "orientation": list(orientation(args, name)), "n_embedded": len(ids)}
        img_of = dict(zip(ids, e["clip"].astype(np.float32)))
        text_fn = lambda sids, m=m: m.text(*tc.get(m.text_pool, sids))
        src_of = dict(zip(man["study_id"], man["source"]))
        for src in sorted(man["source"].unique()):
            tids = [s for s in ids if split.get(s) == "test" and src_of.get(s) == src and s in tc.row]
            img = F.normalize(torch.from_numpy(np.stack([img_of[s] for s in tids])).float(), dim=-1)
            txt = F.normalize(text_fn(tids), dim=-1)
            res[src] = {"n_test": len(tids), **retrieval_suite(img, txt)}
        zs = zero_shot_long(img_of, text_fn, labels, split, set(tc.row))
        lp = linear_probes(ids, e["feat"].astype(np.float32), labels, split)
        for src in res:
            if not isinstance(res[src], dict):
                continue
            z, l_ = zs[zs["source"] == src], lp[lp["source"] == src]
            res[src]["zero-shot long AUROC (findings)"] = float(z["auroc_long"].mean())
            res[src]["probe AUROC (findings)"] = float(l_[l_["kind"] == "finding"]["auroc"].mean())
            res[src]["probe AUROC (outcomes)"] = float(l_[l_["kind"] == "outcome"]["auroc"].mean())
        d = Path(args.out) / name
        d.mkdir(parents=True, exist_ok=True)
        zs.to_csv(d / "zeroshot_long.csv", index=False)
        lp.to_csv(d / "linear_probe.csv", index=False)
        (d / "summary.json").write_text(json.dumps(res, indent=1))
        print(f"[report {name}] " + json.dumps(res), flush=True)


def phase_report(args, cfg):
    if args.data == "multi":
        return report_multi(args, cfg)
    from sklearn.metrics import roc_auc_score
    ev, man = cfg["eval"], manifest(cfg)
    split = dict(zip(man["study_id"], man["split"]))
    tc = TextCache(args.text_cache)
    lab = pd.read_csv(ev["findings_csv"])
    lab["study id"] = lab["study id"].astype(str)
    lab = lab.drop_duplicates("study id").set_index("study id")
    for name in args.models:
        m = MODELS[name]("cpu")
        e = load_emb(args.out, name)
        ids = [str(s) for s in e["ids"]]
        res = {"model": name, "orientation": json.loads((Path(args.out) / name / "orientation.json").read_text())["selected"]}
        te = np.array([split.get(s) == "test" and s in tc.row for s in ids])
        tids = [s for s, t in zip(ids, te) if t]
        img = F.normalize(torch.from_numpy(e["clip"][te]).float(), dim=-1)
        txt = F.normalize(m.text(*tc.get(m.text_pool, tids)), dim=-1)
        res["n_test"] = len(tids)
        res |= {f"{k} (all)": v for k, v in recall(img, txt).items()}
        ch = [recall(img[s:s + 128], txt[s:s + 128]) for s in range(0, len(tids) - 127, 128)]
        res |= {f"{k} (128)": float(np.mean([c[k] for c in ch])) for k in ch[0]}
        p100 = []
        for seed in range(10):                                     # Jolia App. G: random pools of 100, rest dropped
            perm = np.random.RandomState(seed).permutation(len(tids))
            subs = [perm[i:i + 100] for i in range(0, len(tids) - 99, 100)]
            p100.append({k: np.mean([recall(img[s], txt[s])[k] for s in subs]) for k in ch[0]})
        res |= {f"{k} (pool100)": float(np.mean([p[k] for p in p100])) for k in ch[0]}
        # zero-shot, our short prompts: cos(img, pos) - cos(img, neg)
        pr = tc.prompts
        pos = F.normalize(m.text(pr[f"pos_{m.text_pool}"].astype(np.float32), pr[f"pos_{m.text_pool}_norm"]), dim=-1)
        neg = F.normalize(m.text(pr[f"neg_{m.text_pool}"].astype(np.float32), pr[f"neg_{m.text_pool}_norm"]), dim=-1)
        score = (img @ pos.T - img @ neg.T).numpy()
        zs = []
        for j, f in enumerate(pr["names"].tolist()):
            y = lab[f].reindex(tids).to_numpy()
            mk = np.isin(y, [0, 1])
            zs.append({"finding": f, "auroc_short": float(roc_auc_score(y[mk], score[mk, j])) if len(np.unique(y[mk])) == 2 else np.nan})
        # zero-shot long-form (Jolia Table 16): prototypes of 50 positive / 50 negative train reports
        train_ids = [s for s in lab.index if split.get(s) == "train" and s in tc.row]
        long = np.full((5, len(zs)), np.nan)
        for seed in range(5):
            rng = np.random.RandomState(seed)
            for j, f in enumerate(pr["names"].tolist()):
                ytr = lab.loc[train_ids, f]
                P, N = ytr[ytr == 1].index.tolist(), ytr[ytr == 0].index.tolist()
                if len(P) < 50 or len(N) < 50:
                    continue
                ep = F.normalize(m.text(*tc.get(m.text_pool, list(rng.choice(P, 50, replace=False)))), dim=-1).mean(0)
                en = F.normalize(m.text(*tc.get(m.text_pool, list(rng.choice(N, 50, replace=False)))), dim=-1).mean(0)
                s = (img @ F.normalize(ep, dim=0) - img @ F.normalize(en, dim=0)).numpy()
                y = lab[f].reindex(tids).to_numpy()
                mk = np.isin(y, [0, 1])
                if len(np.unique(y[mk])) == 2:
                    long[seed, j] = roc_auc_score(y[mk], s[mk])
        for j, z in enumerate(zs):
            z["auroc_long"] = float(np.nanmean(long[:, j]))
        zdf = pd.DataFrame(zs)
        res["zero-shot short AUROC"] = float(zdf["auroc_short"].mean())
        res["zero-shot long AUROC"] = float(np.nanmean(np.nanmean(long, axis=1)))
        res["zero-shot long AUROC sd"] = float(np.nanstd(np.nanmean(long, axis=1)))
        # linear probe, our protocol (eval.evaluate.linear_probe), on the raw image feature
        feats = {"feat": e["feat"].astype(np.float32)}
        if args.flat_probe and "organ" in e:                       # Jolia's [norm(CLS) + norm(organ queries)]
            o = e["organ"].astype(np.float32)
            o /= np.linalg.norm(o, axis=-1, keepdims=True) + 1e-6
            c = e["feat"] / (np.linalg.norm(e["feat"], axis=-1, keepdims=True) + 1e-6)
            feats["flat"] = np.concatenate([c, o.reshape(len(o), -1)], axis=1).astype(np.float32)
        probes = []
        for fk, X in feats.items():
            lp = linear_probe(ids, X, cfg["data"]["manifest"], ev["findings_csv"], ev["five_year_csv"])
            lp["features"] = fk
            probes.append(lp)
            for task, key in (("findings", "probe findings AUROC"), ("five_year", "probe 5-year AUROC")):
                res[f"{key} [{fk}]"] = float(lp[lp["task"] == task]["auroc_test"].mean())
        d = Path(args.out) / name
        pd.concat(probes).to_csv(d / "linear_probe.csv", index=False)
        zdf.to_csv(d / "zeroshot.csv", index=False)
        (d / "summary.json").write_text(json.dumps(res, indent=1))
        print(f"[report {name}] " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["orient", "select", "extract", "report"])
    ap.add_argument("--models", nargs="+", default=["jolia", "pillar0"], choices=list(MODELS))
    ap.add_argument("--data", choices=["merlin", "multi"], default="merlin",
                    help="merlin: Merlin test (merlin_clip_data); multi: Merlin + CT-RATE + INSPECT + CancerVerse")
    ap.add_argument("--out", default=None)
    ap.add_argument("--text-cache", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--reuse", default=None, help="extract/report: embeddings of another run to reuse (same ids)")
    ap.add_argument("--orient-from", default=None, help="run dir whose orientation.json to use if --out has none")
    ap.add_argument("--orient-source", default=None, help="orient: sample val volumes of this source only")
    ap.add_argument("--orient-domain", default=None, help="orient: sample val volumes of this domain only (chest/abdomen)")
    ap.add_argument("--config", default=str(REPO / "configs" / "base.yaml"))
    ap.add_argument("--n", type=int, default=256, help="orient: val volumes")
    ap.add_argument("--splits", nargs="+", default=["train", "test"], help="extract")
    ap.add_argument("--part", type=int, default=256, help="extract: volumes per saved part (resume unit)")
    ap.add_argument("--flat-probe", action="store_true", help="report: also probe Jolia's CLS + organ features")
    args = ap.parse_args()
    S = os.path.expandvars("$SCRATCHDIR")
    if args.data == "multi":
        d = f"{S}/chest_abdo_clip_data"
        args.out = args.out or f"{S}/merlin_clip_runs/_external_multi"
        args.text_cache = args.text_cache or f"{d}/text_emb/qwen3_emb_8b_len512"
        args.manifest = args.manifest or f"{d}/manifest_384.csv"
        args.reuse = args.reuse or f"{S}/merlin_clip_runs/_external"
        args.orient_from = args.orient_from or f"{S}/merlin_clip_runs/_external"
    else:
        args.out = args.out or f"{S}/merlin_clip_runs/_external"
        args.text_cache = args.text_cache or f"{S}/merlin_clip_data/text_emb/qwen3_emb_8b_len512"
    cfg = load_config(args.config)
    {"orient": phase_orient, "select": phase_select, "extract": phase_extract, "report": phase_report}[args.phase](args, cfg)


if __name__ == "__main__":
    main()
