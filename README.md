# merlin-clip-window

Contrastive (CLIP) pretraining of a 3D CT encoder against radiology reports, used to study the effect of CT
intensity windowing on the learned representation.

The image encoder is Atlas (79 M parameters, trained from scratch) on 384³ volumes at 1.5 mm; reports are encoded
with a frozen Qwen3-Embedding-8B model. Training uses an exact global batch of 128 volume–report pairs and runs on
AMD MI250X and MI300A GPUs. Runs differ only in the windowing front-end that maps Hounsfield units (HU) to the
encoder's input channels, from a single linear ramp to windows learned per 9 mm patch.

## Architecture

```
CT volume (384³, int16 HU)
  → random axial rotation (GPU), clamp to [-1000, 1000] HU
  → windowing front-end: HU → K channels in [0, 1]                 mcw/windowing_arms.py, mcw/patch_windows.py
  → normalisation, bf16
  → Atlas encoder: convolutional stem at 128³,                      mcw/backbone.py, mcw/gemm_conv3d.py, mcw/atlas/
                   multi-scale windowed attention on 64³ tokens
  → 1152-d image embedding ─┐
                            ├─ symmetric InfoNCE, global batch of 128
  1152-d text embedding ────┘
  ← 2-layer MLP ← Qwen3-Embedding-8B report embedding (pre-computed, frozen)
```

The convolutional stem accounts for about 90 % of the training FLOPs. On ROCm, its 3D convolutions are computed as
im2col + GEMM (`mcw/gemm_conv3d.py`), as MIOpen's native 3D convolutions are impractically slow on MI250X.

## Training

A single 384³ volume requires 32–39 GB for a forward and backward pass, so each GPU processes one volume at a time.
The contrastive gradient over the full batch of 128 is obtained with a gradient cache (Gao et al., 2021):

1. forward every volume without gradients and keep its embedding;
2. all-gather the embeddings across GPUs and compute the loss and its gradient with respect to each embedding;
3. forward every volume again with gradients and backpropagate the cached embedding gradient;
4. all-reduce the parameter gradients, clip, and apply an AdamW step.

This yields the exact gradient of a 128-volume batch at the cost of one additional forward pass per volume. Data
order, augmentation and every random permutation are seeded per sample: the data stream does not depend on the GPU
layout, and a resumed run is bit-identical to an uninterrupted one.

A detailed description of the training step, tensor shapes, modules and invariants is given in
[`docs/code.md`](docs/code.md).

## Windowing front-ends

| Front-end | Input channels to the encoder |
|---|---|
| `clip` | Linear ramp [-1000, 1000] HU → [0, 1] |
| `pillar11` | The 10 Pillar-0 radiological window presets and a per-volume min–max channel |
| `learned`, `local` | Learned monotone transfer curves, optionally modulated per patch |
| `global_bank` | 8 learned sigmoid windows shared by all patches |
| `patch_bank` | 8 learned sigmoid windows whose centre and width are predicted per 6³ patch from its HU histogram |

The complete list, including controls and variants, is in [`docs/code.md`](docs/code.md#the-windowing-arms); every
run configuration is indexed in [`configs/README.md`](configs/README.md).

## Installation

```bash
git clone https://github.com/Thomas-Daniel/Windowing_CLIP_Hackathon_Adastra.git
cd Windowing_CLIP_Hackathon_Adastra
pip install torch            # build matching the target GPU, or CPU
pip install -r requirements.txt
```

The reference environment is PyTorch 2.5.1 with ROCm 6.2 (`scripts/setup_env_adastra.sh`).

## Usage

Correctness tests run on CPU and require no data:

```bash
python -m tests.test_cpu gemm arms patch    # subset, about 1 min
python -m tests.test_cpu                    # full suite, about 20 min
```

Benchmarks on a single GPU with synthetic 384³ inputs:

```bash
python scripts/bench_step.py --windowing clip patch_bank --micro-batch 1
python scripts/profile_model.py --windowing pillar11 --profile
```

Training requires the preprocessed volumes and report embeddings:

```bash
torchrun --nproc-per-node=8 train.py --config configs/win8_patch_bank.yaml

# SLURM
REPO_DIR=$PWD VENV_DIR=<venv> CONFIG=configs/win8_patch_bank.yaml \
    sbatch --account=<account> --nodes=2 scripts/submit_train.slurm
```

Preprocessing, report embedding and evaluation are run with the `scripts/submit_*.slurm` jobs; each documents its
parameters in its header.

## Repository structure

```
train.py              training loop: gradient cache, gradient all-reduce, checkpointing, validation
mcw/
├── model.py            CTClip model: windowing front-end, image encoder, text projection
├── windowing_arms.py   fixed and learned windowing front-ends
├── patch_windows.py    per-patch windowing front-ends
├── learned_window.py   learned monotone transfer curves
├── backbone.py         Atlas wrapper
├── atlas/              Atlas encoder (vendored from pillar-pretrain)
├── gemm_conv3d.py      3D convolution as im2col + GEMM
├── vit3d.py            ViT-B/16 alternative encoder
├── data.py             dataset, samplers, augmentation
├── loss.py, dist.py    contrastive loss, distributed utilities
└── config.py, evaluate.py, prompts.py
eval/                 test evaluation: retrieval, zero-shot classification, linear probes, public models
configs/              run configurations
scripts/              preprocessing, embedding, SLURM jobs, benchmarks, run tracking
tests/                CPU correctness tests
docs/                 code documentation
```

## Documentation

- [`docs/code.md`](docs/code.md): training step, tensor shapes, modules and invariants.
- [`configs/README.md`](configs/README.md): index of run configurations.

## Acknowledgements

- Atlas encoder and Pillar-0 training recipe: [YalaLab/pillar-pretrain](https://github.com/YalaLab/pillar-pretrain).
  Window presets: [YalaLab/rave](https://github.com/YalaLab/rave). Both are distributed under the Educational
  Community License 2.0 ([`mcw/atlas/LICENSE_pillar_pretrain`](mcw/atlas/LICENSE_pillar_pretrain)).
- Text encoder: [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B).
- Datasets: Merlin, CancerVerse, CT-RATE and INSPECT (not redistributed).
- Compute: Adastra (CINES), AMD MI250X and MI300A.
