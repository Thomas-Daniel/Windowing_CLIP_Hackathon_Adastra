# How the code works

A walk through the code, in the order it runs. Read this before changing anything.

- [The program in one picture](#the-program-in-one-picture)
- [One training step, line by line](#one-training-step-line-by-line)
- [Tensor shapes through the model](#tensor-shapes-through-the-model)
- [Module by module](#module-by-module)
- [Invariants an optimisation must keep](#invariants-an-optimisation-must-keep)
- [Outside the training loop](#outside-the-training-loop)

## The program in one picture

```
configs/<run>.yaml ──► mcw/config.py (inheritance through `base:`)
                              │
train.py main() ──────────────┤
  ├─ mcw/dist.py setup()            one process per GPU (RANK / WORLD_SIZE / LOCAL_RANK from SLURM or torchrun)
  ├─ mcw/data.py                    MerlinClipDataset (.npy volumes + cached text embeddings), samplers
  ├─ mcw/model.py CTClip            arm → PostNorm → backbone | text MLP
  │     ├─ mcw/windowing_arms.py, mcw/learned_window.py, mcw/patch_windows.py   (the arms)
  │     ├─ mcw/backbone.py → mcw/atlas/*                                        (Atlas encoder, vendored)
  │     │     └─ mcw/gemm_conv3d.py                                             (every Conv3d swapped for a GEMM conv)
  │     └─ mcw/vit3d.py                                                         (alternative backbone)
  ├─ mcw/loss.py clip_loss          symmetric InfoNCE over the global batch
  ├─ mcw/evaluate.py                validation metrics (retrieval, InfoNCE@128, zero-shot)
  └─ scripts/track_runs.py          background process: progress.png / progress.txt
```

There is no DDP wrapper, no Lightning, no Accelerate: the training loop, the gradient synchronisation and the
checkpointing are all in `train.py` (477 lines) and `mcw/dist.py` (111 lines).

## One training step, line by line

All references are to `train.py`. Default Merlin layout: 8 GPUs (GCDs), `micro_batch = 1`, `accum_steps = 16`,
global batch 128.

| # | what | where | cost |
|---|---|---|---|
| 1 | **Load** 16 volumes (one per accumulation step) for this rank. A DataLoader worker `np.load`s each 113 MB int16 `.npy` from Lustre and stacks them (1.8 GB per batch). The text embeddings come from a memory-mapped `(N, 4096)` matrix. | `MerlinClipDataset.__getitem__`, `collate` (`mcw/data.py`) | logged as `data_wait_s` |
| 2 | **Set the LR** of each param group (warmup + cosine, × the group's `lr_mult`). | `lr_at`, l.348 | – |
| 3 | **Augment on GPU.** Copy the batch to the GPU, then for each volume a seeded random axial rotation (±20°, p = 0.5) with `grid_sample` over its 384 slices. Each augmented volume is stored as fp16 until it is used. | `random_axial_rotation`, l.358-361 | small |
| 4 | **Pass 1: forward without grad** of the 16 micro-batches, one by one, keeping only the 1152-d features. | l.366-371 | 1 forward / volume |
| 5 | **Global loss.** The 16 features of this rank are all-gathered with the other ranks (128 × 1152), the text tower runs on the 16 cached embeddings, InfoNCE is computed on the full 128 × 128 logits, and backpropagated **only to the features** (and the text tower / temperature). | `clip_loss` (`mcw/loss.py`), l.372-375 | negligible |
| 6 | **Pass 2: forward + backward with grad** of each micro-batch again, injecting the cached feature gradient (`backward(fgrad[j])`). Parameter gradients accumulate over the 16 micro-batches. | l.377-379 | 1 forward + 1 backward / volume |
| 7 | **Sum gradients over ranks**: one flat all-reduce (~370 MB) after the whole backward. | `D.all_reduce_grads`, l.383 | small at this step time |
| 8 | **Clip and step.** Clip the global norm at 1.0; skip the update if the loss or norm is not finite; AdamW step. | l.386-397 | – |
| 9 | **Bookkeeping.** Every 10 steps: metrics and windowing statistics to `metrics.jsonl`; every 50: tracker refresh. Every 100 steps and at epoch ends: BN running stats averaged over ranks, checkpoint; every 2 epochs: validation, before the checkpoint. On SIGUSR1: save and exit. | l.399-467 | validation logged as `val_time_s` |

Steps 4 and 6 are the **gradient cache** (Gao et al. 2021). It makes a batch of 128 volumes possible when only one
384³ volume fits on a GPU (32-39 GB for a forward + backward), and gives exactly the gradient of one big batch.
The price is one extra forward per volume.

## Tensor shapes through the model

For one 384³ volume (micro-batch 1), Atlas backbone, default sizes (`mcw/backbone.py` `PILLAR0_ATLAS_SMALL`):

```
volume                (1, 1, 384, 384, 384)   int16 on disk → fp32 HU → fp16 after augmentation → fp32
clamp [-1000, 1000]   (1, 1, 384, 384, 384)   fp32                                  CTClip.window
arm                   (1, K, 384, 384, 384)   fp32, K = 11 (clip, pillar11, learned, local) or 8-19 (patch arms)
PostNorm, cast        (1, K, 384, 384, 384)   bf16   (x - 0.289) / 0.198            CTClip.encode_image
── Atlas conv stem (ChestCTEmbed3D, channels_last_3d, every conv is a GemmConv3d) ─────────────────
conv 3³ stride 3      (1, 64, 128, 128, 128)  + BN + ReLU
conv 3³               (1, 192, 128, 128, 128) + BN + ReLU
3 × ConvBlock3D       (1, 192, 128, 128, 128) each: conv 3³ → BN → GELU → conv 3³ → BN, + residual
Downsample3D stride 2 (1, 384, 64, 64, 64)    = 64³ tokens of 384-d
── Atlas multi-scale attention (bf16 autocast) ─────────────────────────────────────────────────────
3 scales by 4³ max-pooling: 64³, 16³, 4³ tokens; 4³ attention windows; stages of 2 / 2 / 8 blocks, 6 heads
readout               (1, 1152)                max-pool of each scale, concatenated (3 × 384)
── text tower ──────────────────────────────────────────────────────────────────────────────────────
Qwen embedding (1, 4096) → Linear 4096→2624 → GELU → Linear 2624→1152 → (1, 1152)
```

The stem's six 192→192 convolutions at 128³ are about 4.2 TFLOP each per forward; the stem is **about 90 % of the
FLOPs** of a step. The ViT backbone (`model.backbone: vit`) replaces all of this with a linear embedding of 16³
patches (24³ = 13 824 tokens of 768-d) and 12 global-attention blocks.

## Module by module

### `mcw/config.py`
YAML with single inheritance (`base: parent.yaml`, deep-merged), `${ENV}` expansion, and `--set a.b=value`
overrides from the command line.

### `mcw/data.py`
- `TextEmbeddings`: memory-mapped `(N, 4096)` float16 matrix + `ids.txt`; one row per report.
- `MerlinClipDataset`: reads the manifest, keeps the requested splits, returns `{idx, study_id, volume (1, D, H, W)
  int16, text (4096,)}`. With `crop` (192³ runs), only the crop is read, through a memory map.
- `EpochShardSampler`: one global permutation per epoch (seed + epoch); rank r takes every `world`-th item, so each
  optimizer step sees exactly one global batch, whatever the number of GPUs. Can resume in the middle of an epoch.
- `BalancedShardSampler`: same, but each global batch holds the same number of volumes per domain (64 chest + 64
  abdomen on the mix).
- `random_axial_rotation`: the only augmentation, on the GPU, seeded per volume.

### `mcw/model.py` — `CTClip`
Builds the arm (`build_arm` or `build_patch_arm`), the backbone and the text projection.
`window()` = clamp → arm → PostNorm; `encode_image()` = window → bf16 → backbone under autocast;
`encode_text()` = MLP; `scale()` = clamped learnable temperature. `set_shuffle_seed()` fixes the permutation of the
shuffle controls so that passes 1 and 2 of the gradient cache see the same input.

### The windowing arms
Input `(B, 1, D, H, W)` HU, output `(B, K, D, H, W)` in [0, 1]. The arm is set by `windowing.arm` in the config.

**What each arm computes**

| arm | output channels |
|---|---|
| `clip` | one ramp [-1000, 1000] → [0, 1] (or `clip_range`), repeated on 11 channels |
| `pillar11` | the 10 fixed Pillar-0 windows (radiologist presets) + a per-volume min-max channel |
| `windows` | a named subset of those presets (e.g. lung, abdomen, liver, bone) |
| `cdf` | one fixed curve: the CDF of body-voxel HU (histogram equalisation), repeated |
| `dense` | no channels: 80 fixed 25-HU ramps fused into the stem's first conv |
| `learned` | 11 learned monotone curves (80 bins of 25 HU), initialised on the presets; `warp` adds a learned shift / width |
| `local`, `local_shuf` | `learned` × a per-patch gate predicted from the patch's HU histogram; `_shuf` permutes the histograms (control) |
| `patch_bank` | 8 sigmoid windows whose centre / width are shifted per 6³ patch by a small net on the patch's HU histogram |
| `global_bank`, `patch_bank_shuf` | controls: the same 8 windows everywhere / with permuted histograms |
| `patch_cdf`, `patch_eq` | per-patch free curves / learned local histogram equalisation |
| `patch_route` | each channel is a per-patch softmax mixture of 16 learned windows |
| `window_select` | the 11 `pillar11` channels mixed by a softmax (global or per patch) into 1 or 3 channels |

Options of the patch arms (`passthrough`, `shift_max_hu`, `width_min_hu`, `smooth`, `scan_context`, `window_init`,
`fixed_windows`, `k`) are documented in the `PatchWindowBank` docstring and in each config's header.

**How it is computed**

| file | classes | how it is computed |
|---|---|---|
| `mcw/windowing_arms.py` | `ClipChannels`, `FixedWindows`, `HUCdf` | element-wise ramps, `expand` for the repeated channels (no copy until PostNorm) |
| | `RampConv3d` (`dense`) | 80 ramps fed straight into the first conv, by blocks of 8 channels, recomputed in backward (checkpoint) |
| | `LocalWindow` (`local`) | learned curves × a gate from 32-bin patch histograms (`scatter_add`), upsampled with `repeat_interleave` |
| `mcw/learned_window.py` | `LearnedWindow` (`learned`) | piecewise-linear CDF lookup (`bucketize` + gather); **custom backward** with two `bincount` histograms per channel (33 ms instead of 2.5 s for the matmul version, kept as `_grad_p_matmul` for the tests); optional warp with its own backward (`_WarpedCDFFn`) |
| `mcw/patch_windows.py` | `PatchDensity` | 80-bin histogram of each 6³ patch with one `bincount` (12-13 ms per volume), optionally averaged over 3³ neighbours (`_box3`) |
| | `PatchWindowBank` (`patch_bank`, `global_bank`, `patch_bank_shuf`) | 1×1×1 conv net on the log density → per-patch centre / width shifts → `_BankFn`: K sigmoids, **custom backward** that sums per-patch gradients with a reshape (no atomics) |
| | `PatchWindowRouter` (`patch_route`), `WindowSelect` (`window_select`), `PatchCDF` (`patch_cdf`, `patch_eq`) | same pattern: per-patch parameters from the density, explicit backward |

`windowing_arms.py` and `learned_window.py` were supplied for this study (hence their French comments); every local
edit is marked `# [mcw]`.

### `mcw/backbone.py` and `mcw/atlas/`
`mcw/atlas/` is Pillar-0's Atlas, vendored **unmodified** from YalaLab/pillar-pretrain @ `ab06083` (Educational
Community License 2.0, `mcw/atlas/LICENSE_pillar_pretrain`). `AtlasBackbone` configures it and patches it from the
outside:
- `swap_conv3d` replaces every `nn.Conv3d` with `GemmConv3d` (same parameters, same `state_dict`);
- the conv stem's `forward` runs in `channels_last_3d`, so the GEMM convs never transpose;
- optional activation checkpointing (`model.grad_checkpointing`), off by default;
- `train()` clears `RelativePosEmb`'s eval-mode cache.

### `mcw/gemm_conv3d.py`
3D convolution as im2col + GEMM, because MIOpen's 3D convs take minutes on MI250X with torch 2.5.1+rocm6.2.
- **Forward**: pad once in channels-last, then for each slab of output depth slices copy an `as_strided` window view
  into a column buffer (`CHUNK_ELEMS` = 2²⁹ elements, 1 GB in bf16) and `torch.mm` it with the `(27·C_in, C_out)`
  weight matrix. Copies move 16-byte groups (viewed as `complex128`), ~4× faster than element-wise bf16 copies.
- **Backward**: `grad_W` recomputes the columns and accumulates `colᵀ @ grad_out` in fp32; `grad_x` is a forward
  conv with the flipped kernel (stride 1), a direct write through the window view (stride = kernel), or k³ shifted
  adds (other strides).
- Python loops over the batch and the depth slabs; only `x` and `W` are saved for backward.

### `mcw/loss.py` and `mcw/dist.py`
`clip_loss`: normalise, all-gather with the local slice keeping its graph (`gather_with_local_grad`), 128 × 128
logits, symmetric cross-entropy. Every rank computes the full loss; summing the parameter gradients over ranks then
gives the exact global gradient (the temperature's gradient is divided by `world`, since every rank already has all
of it). `dist.py` also has the flat gradient all-reduce, BN buffer averaging, and small collective helpers.

### `mcw/evaluate.py`, `eval/`
`mcw/evaluate.py`: feature extraction over a loader and the validation metrics. `eval/evaluate.py`: the test
evaluation (`extract` sharded over GPUs, then `report` on CPU: retrieval, zero-shot, linear probes).
`eval/benchmarks.py`: the per-source tasks of the mix. `eval/external.py`: the same pipeline for public models.

## Invariants an optimisation must keep

These are what make runs comparable. Each has a check.

1. **Exact gradient of the 128-volume batch.** Passes 1 and 2 must compute the same features: same input, same
   permutations, same BN statistics. → `python -m tests.test_cpu model` (gradient cache == plain backward).
2. **Same numerics as the reference ops**, up to summation order: `GemmConv3d` == `nn.Conv3d`, explicit backwards ==
   autograd. → `python -m tests.test_cpu gemm arms patch route bank_opts win_cdf select`.
3. **Determinism.** Data order, augmentation and shuffles are functions of (seed, step, sample index), never of the
   GPU layout or of a stateful RNG. A run stopped and resumed is bit-identical to a continuous one (verified on 2
   nodes). Don't introduce unseeded randomness or layout-dependent seeds.
4. **Identical runs except for the arm.** A speed-up must apply to every arm (or not change the result), otherwise
   comparisons between arms break.
5. **Checkpoint compatibility.** `GemmConv3d` keeps `nn.Conv3d`'s parameter names; keep `state_dict` keys stable so
   existing checkpoints load.

## Outside the training loop

| step | code | runs |
|---|---|---|
| Pair reports and volumes | `scripts/build_manifest.py`, `build_manifest_multi.py`, `build_split_chestabdo.py` | once, CPU |
| NIfTI → 384³ int16 `.npy` | `scripts/preprocess_volumes.py`, `preprocess_multisource.py` | once, ~30 min / node |
| Report embeddings | `scripts/embed_reports.py` (Qwen3-Embedding-8B) | once, 8 GPUs, ~15 min |
| Body-HU histogram (window init) | `scripts/hu_histogram.py` | once |
| Live tracking | `scripts/track_runs.py` | background, every 50 steps |
| Test evaluation | `eval/evaluate.py`, `scripts/compare_evals.py`, `scripts/bootstrap_evals.py` | after each run |
| Window maps | `scripts/viz_patch_windows.py` | on demand |
| Benchmarks / profiling | `scripts/bench_*.py`, `scripts/profile_model.py` (synthetic 384³ data, one GPU) | on demand |
