"""Embed every Merlin report (and the zero-shot prompts) with a frozen Qwen3-Embedding model.

Basic version: the full ``Findings`` text of each report, as released, is one input. No
instruction prefix (Qwen3-Embedding's recipe for documents), no section splitting, no
cleaning. Official Qwen3-Embedding pooling (left padding, last token, L2 normalisation);
the mean over tokens, which pillar-pretrain's cache uses, is stored alongside since it
comes from the same forward pass. ``--max-length`` defaults to 4096 tokens, above the
longest Merlin report, so nothing is truncated (pillar-pretrain truncates at 512).

Two phases (scripts/submit_embed.slurm runs both):
    embed  one process per GCD, each embeds reports[rank::world] -> <out>/shards/rank*.npz
    merge  -> <out>/{last,mean}.npy (N, D) float16, <out>/ids.txt (row order),
              <out>/prompts.npz (zero-shot prompt embeddings), <out>/meta.json

    python scripts/embed_reports.py embed --reports $D/reports.jsonl \
        --model $SCRATCHDIR/hf_models/Qwen3-Embedding-8B --out $D/text_emb/qwen3_emb_8b
    python scripts/embed_reports.py merge --out $D/text_emb/qwen3_emb_8b
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcw.prompts import finding_prompts  # noqa: E402


def _pool(hidden, mask):
    """Last-token pooling (left padding: the last position) and masked mean pooling."""
    last = hidden[:, -1]
    m = mask.unsqueeze(-1).to(hidden.dtype)
    mean = (hidden * m).sum(1) / m.sum(1).clamp_min(1)
    return last, mean


def embed(args) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(local)

    recs = [json.loads(line) for line in open(args.reports)]
    if args.max_reports:
        recs = recs[: args.max_reports]
    ids = [r["study_id"] for r in recs][rank::world]
    texts = [r["text"] for r in recs][rank::world]
    names, pos, neg = finding_prompts()
    if rank == 0:                                   # prompts embedded once, with the same model and pooling
        ids += [f"__prompt_pos__{n}" for n in names] + [f"__prompt_neg__{n}" for n in names]
        texts += pos + neg

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to(device).eval()

    lengths = [len(t) for t in tok(texts, add_special_tokens=True, truncation=False)["input_ids"]]
    order = np.argsort(lengths)[::-1]               # longest first: padding waste stays small, OOM shows up at once
    n_trunc = int(sum(n > args.max_length for n in lengths))
    print(f"[embed rank {rank}/{world}] {len(texts)} texts, tokens mean {np.mean(lengths):.0f} "
          f"max {max(lengths)}, {n_trunc} above max_length={args.max_length}", flush=True)

    out_last = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    out_mean = np.zeros_like(out_last)
    out_last_n = np.zeros(len(texts), dtype=np.float32)     # norms before L2 normalisation: the raw vectors are
    out_mean_n = np.zeros_like(out_last_n)                   # normalised x norm (heads with a bias need them, e.g. Jolia)
    t0, i, n_batches = time.time(), 0, 0
    with torch.no_grad():
        while i < len(order):
            longest = min(lengths[order[i]], args.max_length)
            bs = max(1, args.max_tokens // longest)
            idx = order[i : i + bs]
            batch = tok([texts[k] for k in idx], padding=True, truncation=True,
                        max_length=args.max_length, return_tensors="pt").to(device)
            hidden = model(**batch).last_hidden_state.float()
            last, mean = _pool(hidden, batch["attention_mask"])
            out_last[idx] = F.normalize(last, dim=-1).cpu().numpy()
            out_mean[idx] = F.normalize(mean, dim=-1).cpu().numpy()
            out_last_n[idx] = last.norm(dim=-1).cpu().numpy()
            out_mean_n[idx] = mean.norm(dim=-1).cpu().numpy()
            i += len(idx)
            n_batches += 1
            if n_batches % 50 == 0:
                print(f"[embed rank {rank}] {i}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)

    shard_dir = Path(args.out) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.savez(shard_dir / f"rank{rank:03d}.npz", ids=np.array(ids), last=out_last, mean=out_mean,
             last_norm=out_last_n, mean_norm=out_mean_n, n_tokens=np.array(lengths))
    meta = {"model": str(args.model), "max_length": args.max_length, "world": world,
            "n_truncated": n_trunc, "pooling": ["last", "mean"], "normalized": True, "instruction": None}
    (shard_dir / f"rank{rank:03d}.json").write_text(json.dumps(meta))
    print(f"[embed rank {rank}] done: {len(texts)} texts in {time.time() - t0:.0f}s", flush=True)


def merge(args) -> None:
    out = Path(args.out)
    shards = sorted(glob.glob(str(out / "shards" / "rank*.npz")))
    if not shards:
        raise FileNotFoundError(f"no shards in {out / 'shards'}")
    metas = [json.loads(Path(s).with_suffix(".json").read_text()) for s in shards]
    if len(shards) != metas[0]["world"]:
        raise RuntimeError(f"{len(shards)} shards for world={metas[0]['world']}: an embed rank did not finish")
    parts = [np.load(s) for s in shards]
    ids = np.concatenate([p["ids"] for p in parts])
    last = np.concatenate([p["last"] for p in parts])
    mean = np.concatenate([p["mean"] for p in parts])
    ntok = np.concatenate([p["n_tokens"] for p in parts])

    is_prompt = np.char.startswith(ids.astype(str), "__prompt_")
    rep = np.where(~is_prompt)[0]
    rep = rep[np.argsort(ids[rep])]                 # rows sorted by study id
    np.save(out / "last.npy", last[rep])
    np.save(out / "mean.npy", mean[rep])
    norms = {}
    if "last_norm" in parts[0].files:                # shards written before the norms were stored have none
        norms = {k: np.concatenate([p[k] for p in parts]) for k in ("last_norm", "mean_norm")}
        for k, v in norms.items():
            np.save(out / f"{k}.npy", v[rep])
    (out / "ids.txt").write_text("\n".join(ids[rep].tolist()) + "\n")

    names, pos, neg = finding_prompts()
    row = {s: i for i, s in enumerate(ids.tolist())}
    pidx = [row[f"__prompt_pos__{n}"] for n in names]
    nidx = [row[f"__prompt_neg__{n}"] for n in names]
    np.savez(out / "prompts.npz", names=np.array(names), pos_text=np.array(pos), neg_text=np.array(neg),
             pos_last=last[pidx], neg_last=last[nidx], pos_mean=mean[pidx], neg_mean=mean[nidx],
             **{f"{sign}_{k}": v[ix] for k, v in norms.items() for sign, ix in (("pos", pidx), ("neg", nidx))})

    meta = {**metas[0], "n_reports": int(len(rep)), "dim": int(last.shape[1]),
            "n_truncated": int(sum(m["n_truncated"] for m in metas)),
            "tokens_mean": float(ntok[rep].mean()), "tokens_max": int(ntok[rep].max())}
    meta.pop("world")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[merge] {len(rep)} reports x {last.shape[1]} -> {out}/{{last,mean}}.npy, ids.txt, prompts.npz; {meta}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["embed", "merge"])
    ap.add_argument("--reports", help="reports.jsonl from build_manifest.py (embed)")
    ap.add_argument("--model", help="local Qwen3-Embedding directory (embed)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=32768, help="padded tokens per forward (embed)")
    ap.add_argument("--max-reports", type=int, default=None, help="debug: only the first N reports")
    args = ap.parse_args()
    embed(args) if args.phase == "embed" else merge(args)


if __name__ == "__main__":
    main()
