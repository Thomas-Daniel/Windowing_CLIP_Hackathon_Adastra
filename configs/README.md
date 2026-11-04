# Configs

One YAML file = one run. Each file names its parent in `base:` and overrides only what differs (usually `run_name`,
`seed` and the `windowing:` block); `mcw/config.py` merges the chain. The header comment of each file explains why
the run exists. Arms are described in [`docs/code.md`](../docs/code.md#the-windowing-arms).

```
base.yaml                       Merlin, 384³, 20 epochs
├── win*.yaml                   Merlin study
└── mix_base.yaml               chest / abdomen mix, 384³, 8 epochs, balanced batches
    ├── mix_*.yaml              mix study (Atlas)
    ├── mixvit_base.yaml        same, ViT-B/16 backbone
    │   └── mixvit_*.yaml
    ├── mix192_base.yaml        same, 192³ crops (cheap protocol; compare mix192_* only with each other)
    │   └── mix192_*.yaml
    └── long_base.yaml          192³ crops, 75 epochs (Jolia-like budget)
        └── long_*.yaml
```

## Merlin study (`win*`, 15 300 train volumes)

| config | arm | role |
|---|---|---|
| `win1_clip` | `clip` | no windowing: [-1000, 1000] ramp |
| `win2_pillar11` | `pillar11` | the 10 Pillar-0 presets + min-max |
| `win3_learned` | `learned` | learned global curves, preset init |
| `win3c_learned_warp` | `learned` + warp | + learned shift / width |
| `win4_local` | `local` | learned curves × per-patch gate |
| `win4c_local_warp` | `local` + warp | + learned shift / width |
| `win5_abdo` | `clip` [-120, 240] | classic abdominal window |
| `win6_dense` | `dense` | control: 80 ramps in the first conv |
| `win7_local_shuf` | `local_shuf` | control: shuffled histograms |
| `win8_patch_bank` | `patch_bank` | **per-patch windows from the HU density** |
| `win8_patch_bank_seed1` | `patch_bank` | seed 1 |
| `win9_patch_cdf` | `patch_cdf` | per-patch free curves |
| `win10_patch_eq` | `patch_eq` | learned local equalisation |
| `win11_global_bank` | `global_bank` | control: same windows for every patch |
| `win11_global_bank_seed1` | `global_bank` | seed 1 |
| `win12_patch_bank_shuf` | `patch_bank_shuf` | control: shuffled densities |

## Chest / abdomen mix (`mix_*`, 77 506 train volumes, Atlas, 384³)

| config | role |
|---|---|
| `mix_clip`, `_seed1`, `_seed2` | no windowing |
| `mix_pillar11`, `_seed1`, `_seed2` | fixed Pillar-0 presets |
| `mix_global_bank`, `_seed1`, `_seed2` | learned global windows |
| `mix_patch_bank`, `_seed1`, `_seed2` | **learned per-patch windows** |
| `mix_global_bank_v1`, `mix_patch_bank_v1` | V1: + ramp channel |
| `mix_patch_bank_v2` | V2: wider per-patch bounds |
| `mix_patch_bank_k16` | 16 windows |
| `mix_patch_bank_wmin200` | windows ≥ 200 HU |
| `mix_patch_bank_smooth3` | smoothed HU + ramp |
| `mix_patch_bank_scan` | + per-scan context |
| `mix_patch_bank_uniform` | uniform window init |
| `mix_patch_bank_uniform_seed1`, `_seed2` | seeds 1-2 |
| `mix_hybrid_pillar_patch` | pillar11 + per-patch windows (19 channels) |
| `mix_hybrid_uniform` | hybrid × uniform init |
| `mix_patch_bank_scan_uniform` | scan context × uniform init |
| `mix_patch_route` | per-patch routing over 16 windows |
| `mix_patch_route_v2` | hard top-1 routing, warm-started |
| `mix_win4` | 4 fixed presets (lung, abdomen, liver, bone) |
| `mix_hu_cdf` | global histogram equalisation |
| `mix_dense` | 80 ramps in the first conv |

## Other backbones and budgets

| configs | what |
|---|---|
| `mixvit_{clip,pillar11,global_bank,patch_bank}` | ViT-B/16 counterparts of the mix runs |
| `mix192_{clip,pillar11}` | references on 192³ crops |
| `mix192_select_{global,patch,patch3}` | softmax selection among the 11 Pillar windows (1 or 3 channels) |
| `long_{clip,pillar11,patch_bank}` | 192³ crops, 75 epochs = 45 375 steps (Jolia-like) |
