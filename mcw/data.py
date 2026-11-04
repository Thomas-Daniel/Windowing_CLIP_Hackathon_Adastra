"""Merlin volume + cached report-embedding dataset, and the GPU augmentation.

Volumes are the 384^3 int16 HU exports of scripts/preprocess_volumes.py, report embeddings
the (N, D) float16 matrix of scripts/embed_reports.py (rows in ids.txt order). Augmentation
runs on the GPU on raw HU, before the windowing, so all windowing arms see the same
augmented volumes.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


class TextEmbeddings:
    """Read-only view of a cached embedding matrix (memory-mapped, shared by loader workers)."""

    def __init__(self, emb_dir: str, pooling: str = "last"):
        d = Path(emb_dir)
        self.ids = (d / "ids.txt").read_text().split()
        self.row = {s: i for i, s in enumerate(self.ids)}
        self.emb = np.load(d / f"{pooling}.npy", mmap_mode="r")
        self.dim = int(self.emb.shape[1])

    def __getitem__(self, sid: str) -> np.ndarray:
        return np.asarray(self.emb[self.row[sid]], dtype=np.float32)


class MerlinClipDataset(torch.utils.data.Dataset):
    """``crop``: return a crop^3 cube instead of the whole 384^3 volume (Jolia-style long runs), read through a
    memory map so only the cube's rows are loaded. Centred by default (eval); with ``crop_jitter`` > 0 its centre
    moves by up to +-crop_jitter voxels per axis around the volume centre (the volumes are centre-padded, so the
    body stays in view), drawn from (crop_seed, crop_epoch, index): set ``crop_epoch`` before each epoch's
    DataLoader (workers are re-forked every epoch) and the crops do not depend on the GPU layout."""

    def __init__(self, manifest: str, texts: TextEmbeddings, splits: list[str],
                 max_items: int | None = None, seed: int = 0, keep_ids: set | None = None,
                 crop: int | None = None, crop_jitter: int = 0, crop_seed: int = 0):
        self.crop, self.crop_jitter, self.crop_seed, self.crop_epoch = crop, crop_jitter, crop_seed, 0
        df = pd.read_csv(manifest)
        df["study_id"] = df["study_id"].astype(str)
        df = df[df["split"].isin(splits) & df["study_id"].isin(texts.row)]
        if keep_ids is not None:                                      # eval: e.g. a capped train subset
            df = df[df["study_id"].isin(keep_ids)]
        df = df.sort_values("study_id").reset_index(drop=True)
        if max_items is not None and len(df) > max_items:          # fixed random subset (val monitoring)
            df = df.sample(n=max_items, random_state=seed).sort_values("study_id").reset_index(drop=True)
        self.ids = df["study_id"].tolist()
        self.paths = df["path"].tolist()
        self.domains = df["domain"].astype(str).tolist() if "domain" in df.columns else None   # chest / abdomen mix
        self.texts = texts

    def __len__(self) -> int:
        return len(self.ids)

    def crop_start(self, shape: tuple, i: int) -> list[int]:
        c = self.crop
        s = [(n - c) // 2 for n in shape]
        if self.crop_jitter:
            rng = np.random.default_rng([self.crop_seed, self.crop_epoch, i])
            s = [min(max(a + int(rng.integers(-self.crop_jitter, self.crop_jitter + 1)), 0), n - c)
                 for a, n in zip(s, shape)]
        return s

    def __getitem__(self, i: int) -> dict:
        if self.crop:
            v = np.load(self.paths[i], mmap_mode="r")
            z, y, x = self.crop_start(v.shape, i)
            c = self.crop
            vol = torch.from_numpy(np.ascontiguousarray(v[z:z + c, y:y + c, x:x + c]))
        else:
            vol = torch.from_numpy(np.load(self.paths[i]))         # (D, H, W) int16 HU
        return {"idx": i, "study_id": self.ids[i], "volume": vol[None],
                "text": torch.from_numpy(self.texts[self.ids[i]])}


def collate(items: list[dict]) -> dict:
    return {"idx": torch.tensor([x["idx"] for x in items]),
            "study_id": [x["study_id"] for x in items],
            "volume": torch.stack([x["volume"] for x in items]),
            "text": torch.stack([x["text"] for x in items])}


class EpochShardSampler(torch.utils.data.Sampler):
    """Same global permutation on every rank (seed + epoch), rank r takes items r::world of
    the first floor(N / global_batch) * global_batch: every optimizer step sees exactly one
    global batch, and runs with the same seed see the same data order."""

    def __init__(self, n: int, global_batch: int, rank: int, world: int, seed: int, shuffle: bool = True):
        self.n, self.gb, self.rank, self.world, self.seed, self.shuffle = n, global_batch, rank, world, seed, shuffle
        self.steps_per_epoch = n // global_batch
        self.epoch = 0
        self.start_step = 0                     # resume inside an epoch

    def set_epoch(self, epoch: int, start_step: int = 0) -> None:
        self.epoch, self.start_step = epoch, start_step

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            perm = torch.randperm(self.n, generator=g)
        else:
            perm = torch.arange(self.n)
        perm = perm[: self.steps_per_epoch * self.gb][self.start_step * self.gb:]
        return iter(perm[self.rank :: self.world].tolist())

    def __len__(self) -> int:
        return (self.steps_per_epoch - self.start_step) * self.gb // self.world


class BalancedShardSampler(EpochShardSampler):
    """EpochShardSampler whose every global batch holds global_batch / G volumes of each of the G groups
    (e.g. 64 abdomen + 64 chest). Each group is drawn from its own shuffled stream (seed, epoch, group)
    that restarts every epoch and cycles if needed; the batch is then shuffled so ranks see both
    groups. steps_per_epoch = N // global_batch as before, so within an epoch the larger group is
    slightly under-sampled and the smaller one has a few repeats. Same order for any GPU layout."""

    def __init__(self, groups: list[str], global_batch: int, rank: int, world: int, seed: int):
        super().__init__(len(groups), global_batch, rank, world, seed)
        names = sorted(set(groups))
        if global_batch % len(names):
            raise ValueError(f"global batch {global_batch} not divisible by {len(names)} groups {names}")
        g = np.asarray(groups)
        self.groups = names
        self.index = [torch.from_numpy(np.flatnonzero(g == n)) for n in names]
        self.per_group = global_batch // len(names)

    def epoch_order(self) -> torch.Tensor:
        need = self.steps_per_epoch * self.per_group
        cols = []
        for k, ix in enumerate(self.index):
            gen = torch.Generator().manual_seed((self.seed * 1009 + self.epoch) * 101 + k)
            stream = torch.cat([ix[torch.randperm(len(ix), generator=gen)] for _ in range(-(-need // len(ix)))])
            cols.append(stream[:need].view(self.steps_per_epoch, self.per_group))
        batches = torch.cat(cols, dim=1)                                  # (steps, global_batch)
        gen = torch.Generator().manual_seed(self.seed * 7919 + self.epoch)
        order = torch.stack([torch.randperm(self.gb, generator=gen) for _ in range(self.steps_per_epoch)])
        return torch.gather(batches, 1, order).reshape(-1)

    def __iter__(self):
        perm = self.epoch_order()[self.start_step * self.gb:]
        return iter(perm[self.rank :: self.world].tolist())


def random_axial_rotation(hu: torch.Tensor, max_deg: float, p: float, seeds: list[int],
                          fill: float = -1024.0) -> torch.Tensor:
    """Rotate each volume of (B, 1, D, H, W) in the axial (H, W) plane by U(-max_deg, max_deg)
    with probability p (Pillar-0: z-axis rotation, 20 deg, p = 0.5). Bilinear, out-of-volume
    voxels set to ``fill`` HU. One RNG seed per volume (``seeds``), so the augmentation of a
    sample does not depend on how the global batch is split over GPUs / nodes."""
    if max_deg <= 0 or p <= 0:
        return hu
    B, _, D, H, W = hu.shape
    out = hu.clone()
    for b in range(B):
        generator = torch.Generator().manual_seed(int(seeds[b]))
        if torch.rand((), generator=generator).item() >= p:
            continue
        a = math.radians((torch.rand((), generator=generator).item() * 2 - 1) * max_deg)
        theta = torch.tensor([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0]],
                             device=hu.device, dtype=torch.float32)
        grid = F.affine_grid(theta[None], (1, 1, H, W), align_corners=False)
        sl = hu[b, 0].float().unsqueeze(1) - fill                     # (D, 1, H, W), fill -> 0
        rot = F.grid_sample(sl, grid.expand(D, -1, -1, -1), mode="bilinear",
                            padding_mode="zeros", align_corners=False)
        out[b, 0] = (rot.squeeze(1) + fill).to(hu.dtype)
    return out
