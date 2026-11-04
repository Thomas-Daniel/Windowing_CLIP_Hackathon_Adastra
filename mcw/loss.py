"""Symmetric InfoNCE (CLIP) over the global batch."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import dist as D


def clip_loss(img: torch.Tensor, txt: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """img, txt: this rank's (b, d) features (unnormalised). Every rank evaluates the full
    (B, B) loss, B = b * world; only its own rows carry gradient (gather_with_local_grad).
    Summing parameter gradients over ranks then gives the exact global-batch gradient,
    except for ``scale``, whose gradient each rank already has in full (divide by world)."""
    img = F.normalize(img.float(), dim=-1)
    txt = F.normalize(txt.float(), dim=-1)
    img_all = D.gather_with_local_grad(img)
    txt_all = D.gather_with_local_grad(txt)
    logits = scale * img_all @ txt_all.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    loss = 0.5 * (loss_i2t + loss_t2i)
    with torch.no_grad():
        acc = 0.5 * ((logits.argmax(1) == labels).float().mean() + (logits.argmax(0) == labels).float().mean())
    return loss, {"loss_i2t": loss_i2t.detach(), "loss_t2i": loss_t2i.detach(), "acc": acc}
