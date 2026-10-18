# DECD: Decoupled Experts with Confidence-aware Dispatching

Reference implementation of **DECD**, a dual-expert framework for cross-domain
detection of AI-generated images. DECD keeps the two complementary evidence
sources in fully decoupled experts and fuses them through a scalar,
confidence-aware dispatcher:

1. **Visual Expert (Stage I-A).** An OpenCLIP ViT-L/14 binary Real/Fake
   detector trained with risk-guided decision-boundary refinement: a B100
   reference detector is first measured on six domain-label cells, then
   continued for B50 batches with selective real-side squared-hinge margin
   repair (25.2M trainable parameters).
2. **Semantic Expert (Stage I-B).** A *frozen* Qwen2.5-VL-7B-Instruct MLLM
   prompted with three fixed reasoning prompts. Only a consensus-gated
   bottleneck adapter (`d/4`) and one shared linear head are trained
   (6.43M trainable parameters, ~0.085% of the backbone), using prompt-level
   BCE plus a neighborhood-preservation KL term.
3. **Confidence-aware Dispatcher (Stage II).** Both experts are frozen.
   Per-expert temperatures are fitted on the validation split by NLL grid
   search, a 6→64→1 gating MLP combines the two calibrated probabilities as a
   convex combination, and a validation quantile threshold abstains on the
   15% least reliable samples.

No gradient ever crosses between the experts; the dispatcher sees only six
scalar reliability signals per sample.

```text
            ┌──────────────────────┐   cv logit
 image ───► │  Visual Expert (CLIP)│ ─────────────┐
            └──────────────────────┘              ▼
                                      ┌────────────────────┐
            ┌──────────────────────┐  │  calibrated fusion │  p, route,
 image ───► │ MLLM Expert (frozen  │──┤  + gate MLP (64)   │──► abstain?
            │ + PP adapter, K=3)   │  │  + abstention (b)  │
            └──────────────────────┘  └────────────────────┘
```

## Environment

Linux with Python 3.10+ is recommended; a CUDA-capable NVIDIA GPU is required
for the reported runs (Ampere or newer for bfloat16 feature extraction).
Create an isolated environment and install a PyTorch build matching the
machine's CUDA runtime via the
[official selector](https://pytorch.org/get-started/locally/), then:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Stage II (dispatcher) and the full test suite also run on CPU.

## Repository layout

```text
decd/
  visual_expert/    # OpenCLIP detector, six-cell risk profile, margin repair
  mllm_expert/      # frozen Qwen2.5-VL backbone, PP adapter, losses, head
  dispatcher/       # calibration, reliability signals, gate MLP, abstention
scripts/            # released entry points for every training stage
tools/              # MOSAIC manifest preparation
configs/            # frozen reported hyper-parameters (one YAML per stage)
tests/              # 40 CPU unit tests, no weights or network access needed
```

No datasets, images, trained checkpoints, or prediction files are bundled.

## Data manifests

Every manifest is a CSV with at least:

- `image_path`: absolute or manifest-relative image path.
- `domain`: `Cartoon`, `Chart`, or `Photo_and_poster` (optional for plain
  evaluation; reported under `all` when omitted).
- `label`: `0` Real, `1` Fake.

`sample_id`, `label_name`, and `sha256` are optional. Example:

```csv
sample_id,image_path,domain,label,label_name,sha256
example_real_001,../images/Cartoon/Real/a.jpg,Cartoon,0,Real,
example_fake_001,../images/Cartoon/Fake/b.jpg,Cartoon,1,Fake,
```

`tools/prepare_mosaic.py` scans the MOSAIC layout
(`ROOT/{Cartoon,Chart,Photo}/{0_real,1_fake}/...`), removes exact SHA256
duplicates and held-out overlaps, standardizes accepted images as JPEG, and
writes a relative-path manifest without touching the source images.

## Stage I-A — Visual Expert

Reported configuration:
[`configs/visual_expert_reported.yaml`](configs/visual_expert_reported.yaml).
All paths in the YAML are repository-root-relative; every command accepts the
equivalent path/device overrides.

| Setting | Reported value |
|---|---:|
| Backbone | OpenCLIP `ViT-L-14-quickgelu` / `openai` |
| Trainable visual layers | last 2 Transformer blocks + `ln_post` |
| Head | one shared binary linear head |
| Image size | 224 |
| Batch size | 6, balanced over 3 domains × 2 labels |
| Reference / repair length | 100 / 50 batches |
| Margin `gamma` | 0.5 |
| Repair coefficient / ramp | 0.1 / 10 batches |
| Maximum auxiliary scale `rho_max` | 10 |
| Repair side | Real only |
| Risk column | `risk_mean_m0p5` (mean hinge risk) |
| Risk normalization | label-wise min-max, exponent 1.0 |
| Seed | 2026 |

The repair gradient is
`g_det + lambda_now * rho * g_rep` with
`rho = min(||g_det||_2 / (||g_rep||_2 + eps), rho_max)` and `lambda_now`
ramping to 0.1 over the first 10 batches. The profiled risk is the mean hinge
(`risk_mean_m0p5`); the training repair objective uses a squared hinge.

```bash
# 1. B100 reference detector
python scripts/train_reference.py \
  --config configs/visual_expert_reported.yaml \
  --train-manifest data/manifests/mosaic_train.csv \
  --dev-manifest data/manifests/mosaic_dev.csv \
  --output outputs/reference/reference.pth --device cuda:0
# add --dry-run (CPU) to validate the config, letterbox preprocessing and a
# balanced batch without downloading OpenCLIP weights.

# 2. Fixed six-cell risk profile (same training manifest as the reference)
python scripts/build_risk_profile.py \
  --config configs/visual_expert_reported.yaml \
  --manifest data/manifests/mosaic_train.csv \
  --checkpoint outputs/reference/reference.pth \
  --output outputs/risk/risk_profile.json --device cuda:0

# 3. B50 continuation with real-side margin repair
python scripts/train_visual_expert.py \
  --config configs/visual_expert_reported.yaml \
  --train-manifest data/manifests/mosaic_train.csv \
  --dev-manifest data/manifests/mosaic_dev.csv \
  --reference-checkpoint outputs/reference/reference.pth \
  --risk-profile outputs/risk/risk_profile.json \
  --output outputs/visual_expert/visual_expert.pth --device cuda:0

# 4. Scored predictions for any split (predictions.csv keeps per-sample logits;
#    these logits are the cv_logit input of Stage II)
python scripts/evaluate_visual_expert.py \
  --config configs/visual_expert_reported.yaml \
  --manifest data/manifests/mosaic_train.csv \
  --output-dir outputs/visual_eval/train --device cuda:0
```

Run the evaluator once per split (train, dev, eval). Checkpoints embed
sanitized configuration, manifest file names, and SHA256 fingerprints; local
absolute paths are stripped before serialization and continuation restores the
reference optimizer state (part of the B100→B50 protocol).

## Stage I-B — MLLM Semantic Expert

Reported configuration:
[`configs/mllm_expert_reported.yaml`](configs/mllm_expert_reported.yaml)
(frozen `Qwen/Qwen2.5-VL-7B-Instruct`, `feature_dim: 3584`).

The frozen backbone is run **once per split** with three fixed reasoning
prompts; the mean-pooled last-layer features `z0` are cached to disk, and all
adapter training happens on the cache (backbone-free, CPU-capable).

```bash
# 1. Cache frozen features for the training and dev manifests
python scripts/extract_mllm_features.py \
  --config configs/mllm_expert_reported.yaml \
  --manifest data/manifests/mosaic_train.csv \
  --output outputs/mllm_features/train_features.pt --device cuda:0
python scripts/extract_mllm_features.py \
  --config configs/mllm_expert_reported.yaml \
  --manifest data/manifests/mosaic_dev.csv \
  --output outputs/mllm_features/dev_features.pt --device cuda:0

# 2. Train the consensus-gated PP adapter + head (30 epochs,
#    checkpoint with best dev macro-F1)
python scripts/train_mllm_expert.py \
  --config configs/mllm_expert_reported.yaml --device cuda:0 --overwrite
```

Key settings: bottleneck ratio 4, BCE averaged over the three prompts,
neighborhood-preservation KL with `tau_np = 0.1`, `lambda_np = 0.5`, AdamW,
lr 1e-4, 30 epochs, best-dev-macro-F1 selection. The adapter's second
projection is zero-initialized, so training starts at the frozen baseline.

Features are extracted in **bfloat16**: fp16 summation inside the Qwen2.5-VL
vision tower overflows on some inputs and yields NaN pooled features. On
GPUs without bf16 support, use `--device cpu` (float32).

A smaller Qwen2.5-VL backbone can be used by copying the YAML and updating
`backbone_name` and `feature_dim` (e.g. 3B-Instruct → 2048), then passing the
copy with `--config` and optionally `--model-name`.

## Stage II — Confidence-aware Dispatcher

Reported configuration:
[`configs/dispatcher_reported.yaml`](configs/dispatcher_reported.yaml).

```bash
# 1. Align the frozen experts' scalar outputs for every split
for split in train dev eval; do
  python scripts/build_dispatcher_signals.py \
    --features outputs/mllm_features/${split}_features.pt \
    --mllm-checkpoint outputs/mllm_expert/mllm_expert.pth \
    --cv-predictions outputs/visual_eval/${split}/predictions.csv \
    --output data/signals/${split}_signals.csv --device cuda:0
done
# (Use the matching manifest names; the eval split needs mllm features too:
#  run extract_mllm_features.py once more for the eval manifest.)

# 2. Fit temperatures on dev and train the gate MLP + abstention threshold
python scripts/train_dispatcher.py \
  --config configs/dispatcher_reported.yaml --device cuda:0 --overwrite

# 3. Evaluate the fused system (overall, per-domain, selective prediction)
python scripts/evaluate_dispatcher.py \
  --config configs/dispatcher_reported.yaml \
  --signals data/signals/eval_signals.csv \
  --output-dir outputs/dispatcher_evaluation --device cuda:0
```

The six reliability signals per sample are:

```text
p_cv, mean_k p_mllm^(k), 2|p_cv - 0.5|, 2|mean_k p_mllm^(k) - 0.5|,
|p_cv - mean_k p_mllm^(k)|, variance_k p_mllm^(k)
```

Temperatures default to dev-NLL grid search (0.05–5.0, step 0.01); pin values
via `temperature_cv` / `temperature_mllm` in the YAML. The abstention
threshold is the dev quantile that covers exactly the `1 - b` fraction of
samples (`b = 0.15`). Evaluation writes `metrics.json`, `metrics.csv`, and
`predictions.csv`; predictions never contain resolved local image paths.

## Tests

```bash
python -m pytest tests/ -q
```

40 CPU unit tests cover the risk profile and margin repair, the PP adapter /
gates / losses, frozen-feature cache semantics (including the lazy
`transformers` import), temperature calibration, reliability-vector
construction, the gate MLP, abstention bookkeeping, the signals CSV pipeline,
and end-to-end training loops with dummy backbones. No GPU, network access, or
downloaded weights are required.

## Reproducibility notes

- Real is label 0 and Fake is label 1; the classifier is domain-agnostic at
  inference time.
- Seed 2026 across all stages; the reference and repair stages share the same
  six-cell balanced sampler.
- Only the last two visual Transformer blocks, `ln_post`, and the binary head
  are trainable; the MLLM backbone is fully frozen; both experts are frozen
  during Stage II.
- The risk profile is fixed before Visual Expert continuation begins; the
  repair loss is real-only (fakes still contribute to binary detection loss).
- Prompt order in feature caches is fixed by `REASONING_PROMPTS`; caches with
  different prompts are rejected at training time.
- `transformers` (>= 4.49) is imported lazily, so inspection, configuration,
  and the entire test suite work without the 7B checkpoint.
