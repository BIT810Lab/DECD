# DECD: Decoupled Experts with Confidence-aware Dispatching

This repository provides training code only. Trained model checkpoints are not distributed.

**DECD** is a dual-expert framework for cross-domain detection of AI-generated images. It combines complementary visual and semantic evidence using a confidence-aware dispatcher:

1. **Visual Expert (Stage I-A).** An OpenCLIP ViT-L/14 Real/Fake detector trained with risk-guided decision-boundary refinement. A 100-mini-batch reference stage establishes domain–label risks, followed by 50 mini-batches of real-side margin repair (25.2M trainable parameters).
2. **Semantic Expert (Stage I-B).** A frozen Qwen2.5-VL-7B-Instruct backbone with three fixed reasoning prompts. A consensus-gated bottleneck adapter (`d/4`) and a shared linear head are trained using prompt-level BCE and neighborhood-preservation KL (6.43M trainable parameters).
3. **Confidence-aware Dispatcher (Stage II).** With both experts frozen, per-expert temperature calibration and a 6→64→1 gating MLP combine their probabilities. A validation-based reliability threshold supports selective abstention.

The experts are trained independently. The dispatcher uses six scalar reliability signals per image.

```text
            ┌──────────────────────┐   cv logit
 image ───► │ Visual Expert (CLIP) │ ─────────────┐
            └──────────────────────┘              ▼
                                       ┌────────────────────┐
            ┌──────────────────────┐  │ calibrated fusion │  p, route,
 image ───► │ MLLM Expert (frozen  │──┤ + gate MLP (64)    │──► abstain?
            │ + PP adapter, K=3)   │  │ + abstention (b)   │
            └──────────────────────┘  └────────────────────┘
```

## Environment

Linux and Python 3.10+ are recommended. Install a CUDA-enabled PyTorch build appropriate for your system using the [official selector](https://pytorch.org/get-started/locally/), then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt`. The Visual Expert uses OpenCLIP 3.3.0; `transformers` supports the Semantic Expert. The dispatcher can also run on CPU.

## Repository layout

```text
decd/
  visual_expert/    # OpenCLIP detector, six-cell risk profile, margin repair
  mllm_expert/      # frozen Qwen2.5-VL backbone, PP adapter, losses, head
  dispatcher/       # calibration, reliability signals, gate MLP, abstention
scripts/            # training, profiling, evaluation and inference
tools/              # MOSAIC data preparation and manifest utilities
configs/            # model and training configurations
tests/              # unit and integration tests
```

## MOSAIC dataset

MOSAIC contains 6,000 images across the Photo, Cartoon, and Chart domains, balanced between real and AI-generated images. The dataset follows a stratified 70/10/20 train–development–test split over six domain–label groups:

| Partition | Per domain–label group | Total | Use |
|---|---:|---:|---|
| Train | 700 | 4,200 | Reference training, risk profiling, margin repair |
| Dev | 100 | 600 | Validation and calibration |
| Radar/Test | 200 | 1,200 | Final evaluation |

## Data manifests

Each data manifest is a CSV containing:

- `image_path`: absolute or manifest-relative image path.
- `domain`: `Cartoon`, `Chart`, or `Photo_and_poster` (optional for domain-agnostic evaluation).
- `label`: `0` for Real, `1` for Fake.

Optional columns include `sample_id`, `label_name`, and `sha256`.

```csv
sample_id,image_path,domain,label,label_name,sha256
example_real_001,../images/Cartoon/0_real/a.jpg,Cartoon,0,Real,
example_fake_001,../images/Cartoon/1_fake/b.jpg,Cartoon,1,Fake,
```

`tools/prepare_mosaic.py` reads images from `ROOT/{Cartoon,Chart,Photo}/{0_real,1_fake}/...`, checks image identities and exclusions, standardizes selected images to JPEG, and writes a relative-path manifest. Use repeatable `--exclude-manifest` arguments to exclude other partitions.

Metadata with columns `file_name,domain,label,sha256` can be converted into training manifests without re-encoding the images. Supply the appropriate selection CSV for each partition:

```bash
python tools/convert_mosaic_manifest.py \
  --metadata data/pool/metadata.csv \
  --image-root data/pool/images \
  --selection-manifest data/selections/train.csv \
  --split train \
  --output data/manifests/mosaic_train.csv

python tools/convert_mosaic_manifest.py \
  --metadata data/pool/metadata.csv \
  --image-root data/pool/images \
  --selection-manifest data/selections/dev.csv \
  --split dev \
  --output data/manifests/mosaic_dev.csv
```

The data utilities normalize `Photo` to `Photo_and_poster` and accept `0_real`/`1_fake` label aliases. SHA256 fingerprints verify image identity and loaded file contents; optional `group_id` annotations support cross-partition group isolation.

To create a balanced selection of 1,000 images per domain–label group from a candidate pool, run:

```bash
python tools/build_mosaic_benchmark.py \
  --metadata data/pool/metadata.csv \
  --image-root data/pool/images \
  --seed 2026 \
  --output-dir data/mosaic6000_selection
```

The tool writes `mosaic_selected_6000.csv`, remaining candidate records, and SHA256 metadata. The selected rows use `split=unassigned`; provide explicit partition selection manifests to organize the 6,000 selected images into Train (4,200), Dev (600), and Radar/Test (1,200). The `--verify` option checks an existing selection.

## Stage I-A — Visual Expert

Configuration: [`configs/visual_expert_reported.yaml`](configs/visual_expert_reported.yaml).

| Setting | Value |
|---|---|
| Backbone | OpenCLIP `ViT-L-14-quickgelu`, `openai` initialization |
| Trainable layers | Last two visual Transformer blocks, `ln_post`, linear head |
| Image size | 224 × 224 |
| Batch size | 6, balanced across three domains × two labels |
| Reference / repair | 100 / 50 mini-batches |
| Margin `gamma` | 0.5 |
| Repair coefficient / ramp | 0.1 / 10 repair mini-batches |
| Auxiliary scale cap `rho_max` | 10 |
| Repair side | Real only |
| Risk profiling | `risk_mean_m0p5` (mean hinge) |
| Risk normalization | Label-wise min–max, exponent 1.0 |
| Seed | 2026 |

The fixed risk profile uses mean hinge deficiency. Real-side repair uses a squared-hinge margin loss and combines gradients as

`g_det + lambda_now * rho * g_rep`, where
`rho = min(||g_det||_2 / (||g_rep||_2 + eps), rho_max)`.

`lambda_now` ramps to 0.1 over the first 10 repair mini-batches.

### Pretrained initialization

Reference training initializes OpenCLIP using `pretrained: openai`, which resolves official initialization weights through the supported cache/download mechanism. To load a compatible local initialization file, pass `--pretrained-path weights/ViT-L-14.pt` or set `pretrained_path` in the YAML.

CLI paths resolve from the working directory; YAML paths resolve from the project root, with CLI values taking precedence. Supported local files include trusted OpenAI CLIP TorchScript weights, complete OpenCLIP state dictionaries, and `{'state_dict': ...}` wrappers. Initialization checks tensor keys, shapes, and finite values and rejects incompatible files.

Initialization metadata records the loader and OpenCLIP version; a local initialization file also records its SHA256. Reference checkpoints retain the model and AdamW state for repair continuation.

### Training and evaluation

```bash
# 1. B100 reference training
python scripts/train_reference.py \
  --config configs/visual_expert_reported.yaml \
  --train-manifest data/manifests/mosaic_train.csv \
  --dev-manifest data/manifests/mosaic_dev.csv \
  --output outputs/reference/reference.pth --device cuda:0

# 2. Fixed domain–label risk profile from Train
python scripts/build_risk_profile.py \
  --config configs/visual_expert_reported.yaml \
  --manifest data/manifests/mosaic_train.csv \
  --checkpoint outputs/reference/reference.pth \
  --output outputs/risk/risk_profile.json --device cuda:0

# 3. B50 real-side margin repair
python scripts/train_visual_expert.py \
  --config configs/visual_expert_reported.yaml \
  --train-manifest data/manifests/mosaic_train.csv \
  --dev-manifest data/manifests/mosaic_dev.csv \
  --reference-checkpoint outputs/reference/reference.pth \
  --risk-profile outputs/risk/risk_profile.json \
  --output outputs/visual_expert/visual_expert.pth --device cuda:0

# 4. Visual Expert predictions
python scripts/evaluate_visual_expert.py \
  --config configs/visual_expert_reported.yaml \
  --manifest data/manifests/mosaic_dev.csv \
  --checkpoint outputs/visual_expert/visual_expert.pth \
  --output-dir outputs/visual_eval/dev --device cuda:0
```

For a data and preprocessing check, `train_reference.py` supports `--device cpu --dry-run`. The evaluator outputs per-sample logits and probabilities along with classification metrics. Run it separately for each data partition using the corresponding manifest.

## Stage I-B — MLLM Semantic Expert

Configuration: [`configs/mllm_expert_reported.yaml`](configs/mllm_expert_reported.yaml) (`Qwen/Qwen2.5-VL-7B-Instruct`, `feature_dim: 3584`).

The frozen backbone processes three fixed reasoning prompts per image. Mean-pooled features are cached per partition, and the adapter and classification head are trained using the cached features.

```bash
# 1. Cache frozen features for Train and Dev
python scripts/extract_mllm_features.py \
  --config configs/mllm_expert_reported.yaml \
  --manifest data/manifests/mosaic_train.csv \
  --output outputs/mllm_features/train_features.pt --device cuda:0

python scripts/extract_mllm_features.py \
  --config configs/mllm_expert_reported.yaml \
  --manifest data/manifests/mosaic_dev.csv \
  --output outputs/mllm_features/dev_features.pt --device cuda:0

# 2. Train the adapter and head
python scripts/train_mllm_expert.py \
  --config configs/mllm_expert_reported.yaml --device cuda:0 --overwrite
```

The adapter uses a bottleneck ratio of 4, prompt-level BCE, and neighborhood-preservation KL (`tau_np = 0.1`, `lambda_np = 0.5`). Optimization uses AdamW with learning rate `1e-4` for 30 epochs, selecting by Dev macro-F1. The second adapter projection is initialized to zero.

Feature extraction uses bfloat16 to maintain numerical stability during Qwen2.5-VL visual processing; CPU execution uses float32. A smaller backbone can be configured by updating `backbone_name` and `feature_dim` in a copy of the YAML (for example, 3B-Instruct with `feature_dim: 2048`).

## Stage II — Confidence-aware Dispatcher

Configuration: [`configs/dispatcher_reported.yaml`](configs/dispatcher_reported.yaml).

```bash
# 1. Build reliability signals from aligned expert outputs
for split in train dev eval; do
  python scripts/build_dispatcher_signals.py \
    --features outputs/mllm_features/${split}_features.pt \
    --mllm-checkpoint outputs/mllm_expert/mllm_expert.pth \
    --cv-predictions outputs/visual_eval/${split}/predictions.csv \
    --output data/signals/${split}_signals.csv --device cuda:0
done

# 2. Calibrate experts and train the dispatcher
python scripts/train_dispatcher.py \
  --config configs/dispatcher_reported.yaml --device cuda:0 --overwrite

# 3. Evaluate fusion and selective prediction
python scripts/evaluate_dispatcher.py \
  --config configs/dispatcher_reported.yaml \
  --signals data/signals/eval_signals.csv \
  --output-dir outputs/dispatcher_evaluation --device cuda:0
```

Extract features and Visual Expert predictions for every partition used by the dispatcher, including the evaluation partition.

The six scalar reliability features per sample are:

```text
p_cv, mean_k p_mllm^(k), 2|p_cv - 0.5|, 2|mean_k p_mllm^(k) - 0.5|,
|p_cv - mean_k p_mllm^(k)|, variance_k p_mllm^(k)
```

Expert temperatures are fitted by Dev NLL grid search (0.05–5.0, step 0.01); values may be supplied as `temperature_cv` and `temperature_mllm` in the YAML. A Dev quantile sets the abstention threshold for a rejection budget of 15%. Evaluation writes `metrics.json`, `metrics.csv`, and `predictions.csv`.

## Tests

Run the Visual Expert unit tests:

```bash
python -B -m unittest discover -s decd/visual_expert/tests -p test_visual_release_unittest.py -v
```

Run the Visual Expert CPU smoke test:

```bash
python tools/visual_cpu_smoke.py --output-dir outputs/visual_cpu_data
```

Run the DECD test suite:

```bash
python -m pytest tests/ -q
```

These tests cover the Visual Expert's data manifests, stratified selection, risk profiling, margin repair, gradient scaling, checkpoint handling, and the DECD training interfaces.

## Visual-only archive

To package the Visual Expert training code:

```bash
python tools/package_visual_release.py --output-dir outputs/visual_release
```

The packager selects files from `RELEASE_FILES.txt` and records checksums for the archive and its members. The resulting package includes code, configuration, documentation, and tests, without model weights or image data.

The repository's code license is Apache-2.0; see `LICENSE` and `decd/visual_expert/THIRD_PARTY_NOTICES.md` for applicable third-party notices.

## Implementation notes

- Real is label 0 and Fake is label 1. The Visual Expert uses a shared threshold without domain-specific inference.
- Seed 2026 is used across the Visual Expert stages; Reference and Repair use the six-cell balanced sampler.
- Only the final two visual Transformer blocks, `ln_post`, and the binary head are trainable. Both experts are frozen during dispatcher training.
- The Visual risk profile is computed once from Train and held fixed during Repair; Fake samples continue to contribute to the detection loss.
- Prompt order in the Semantic Expert cache is defined by `REASONING_PROMPTS`.
