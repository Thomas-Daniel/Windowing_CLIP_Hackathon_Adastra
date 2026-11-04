"""Evaluation shared by train.py (val monitoring) and eval/ (test): retrieval, InfoNCE at
the training batch size, and zero-shot findings AUROC."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import dist as D


@torch.no_grad()
def encode_dataset(model, loader, device) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Unnormalised image and text features for every item of a (rank-sharded) loader,
    gathered on all ranks in dataset order. Assumes the sampler gives rank r items r::world
    of the padded index list (see ``eval_sampler_indices``)."""
    model.eval()
    idx, img, txt = [], [], []
    for batch in loader:
        hu = batch["volume"].to(device, non_blocking=True).float()
        img.append(model.encode_image(hu))
        txt.append(model.encode_text(batch["text"].to(device)))
        idx.append(batch["idx"].to(device))
    idx, img, txt = torch.cat(idx), torch.cat(img), torch.cat(txt)
    idx, img, txt = D.gather_cat(idx), D.gather_cat(img), D.gather_cat(txt)
    order = torch.argsort(idx)
    idx, img, txt = idx[order], img[order], txt[order]
    keep = torch.ones_like(idx, dtype=torch.bool)
    keep[1:] = idx[1:] != idx[:-1]                       # drop the padding duplicates
    ids = loader.dataset.ids
    return [ids[i] for i in idx[keep].tolist()], img[keep], txt[keep]


def eval_sampler_indices(n: int, rank: int, world: int) -> list[int]:
    pad = (-n) % world
    full = list(range(n)) + list(range(pad))
    return full[rank::world]


def retrieval_metrics(img: torch.Tensor, txt: torch.Tensor, ks=(1, 5, 10)) -> dict:
    img, txt = F.normalize(img.float(), dim=-1), F.normalize(txt.float(), dim=-1)
    sim = img @ txt.T
    gt = torch.arange(sim.shape[0], device=sim.device)
    out = {}
    for name, s in (("i2t", sim), ("t2i", sim.T)):
        rank = (s > s[gt, gt][:, None]).sum(1)          # 0 = correct item ranked first
        for k in ks:
            out[f"{name}_R@{k}"] = float((rank < k).float().mean())
        out[f"{name}_medR"] = float(rank.float().median() + 1)
    out["n_candidates"] = int(sim.shape[0])
    return out


def batched_infonce(img: torch.Tensor, txt: torch.Tensor, scale: float, batch: int) -> float:
    """Mean CLIP loss over consecutive chunks of ``batch`` pairs (the training batch size)."""
    img, txt = F.normalize(img.float(), dim=-1), F.normalize(txt.float(), dim=-1)
    losses = []
    for s in range(0, img.shape[0] - batch + 1, batch):
        logits = scale * img[s:s + batch] @ txt[s:s + batch].T
        y = torch.arange(batch, device=logits.device)
        losses.append(0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)))
    return float(torch.stack(losses).mean()) if losses else float("nan")


@torch.no_grad()
def zero_shot_findings(model, img: torch.Tensor, ids: list[str], prompts_npz: str, labels_csv: str,
                       pooling: str = "last") -> tuple[dict, pd.DataFrame]:
    """AUROC of cos(img, pos) - cos(img, neg) per Merlin finding (labels: 1 pos, 0 neg, -1 missing)."""
    from sklearn.metrics import roc_auc_score

    pr = np.load(prompts_npz)
    dev = img.device
    pos = F.normalize(model.encode_text(torch.from_numpy(pr[f"pos_{pooling}"]).to(dev)), dim=-1)
    neg = F.normalize(model.encode_text(torch.from_numpy(pr[f"neg_{pooling}"]).to(dev)), dim=-1)
    im = F.normalize(img.float(), dim=-1)
    score = (im @ pos.T - im @ neg.T).cpu().numpy()          # (N, n_findings)
    lab = pd.read_csv(labels_csv)
    lab["study id"] = lab["study id"].astype(str)
    lab = lab.drop_duplicates("study id").set_index("study id")
    rows = []
    for j, name in enumerate(pr["names"].tolist()):
        if name not in lab.columns:
            continue
        y = lab[name].reindex(ids).to_numpy()
        m = np.isin(y, [0, 1])
        auc = float(roc_auc_score(y[m], score[m, j])) if m.any() and len(np.unique(y[m])) == 2 else float("nan")
        rows.append({"finding": name, "auroc": auc, "n": int(m.sum()), "n_pos": int((y[m] == 1).sum())})
    df = pd.DataFrame(rows)
    return {"zs_macro_auroc": float(df["auroc"].mean()), "zs_n_findings": int(df["auroc"].notna().sum())}, df
