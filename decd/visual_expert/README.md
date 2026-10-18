# DECD Visual Expert — extracted training code

This package contains only the Visual Expert training chain:
**B100 reference → one fixed Train-only risk profile → B50 real-side repair**.
It does not include Qwen, Prompt-Purification, Dispatcher, Autoresearch,
baselines, real images, historical checkpoints or paper result tables.
It does not claim to reproduce the complete DECD paper or its reported metrics.
No open-source license is selected: LICENSE must be supplied by the author.

## Preserved method

- OpenCLIP `ViT-L-14-quickgelu`, OpenAI pretrained initialization, 224px.
- Train only the last two visual Transformer blocks, `ln_post`, and binary head.
- Real=0, Fake=1; batch6, balanced over three domains × two labels.
- AdamW backbone LR1e-5 / head LR1e-4, weight decay1e-4, seed2026.
- Exactly one B100 reference endpoint and one B50 continuation (not alternating cycles).
- Frozen reference **mean hinge** risk (`risk_mean_m0p5`), label-wise minmax,
  exponent1; the original tied-risk policy (all-one weights) is preserved.
- Real-only **squared hinge** repair, gamma0.5, lambda0.1 ramped over10 batches,
  rho=min(norm(g_det)/(norm(g_rep)+1e-30),10), combined detached gradients.
- B50 restores the reference model and AdamW state, including step100.
  Its sampler is recreated with seed2026, as in the extracted implementation;
  this repeats the reference sampler's first50 draws, not a resume at draw101.
- Letterbox: EXIF transpose, RGB, bicubic fit, gray127 padding, CLIP normalization.
  Existing FP16/autocast behavior is preserved; invalid losses/gradients now fail.

**Author decision pending:** the current manuscript describes squared-hinge
profiling, while historical V9 metadata and this configuration use mean hinge.
This extraction deliberately does not change the historical training method.

## Environment

Linux / Python3.10+ is recommended. Install an appropriate PyTorch/CUDA build
for your machine first. Then, from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
```

These are dependency ranges, not a universally validated environment lock.
CPU checks used Python3.12, torch2.5.1, open-clip-torch3.3.0 and Pillow11.
No dependency or model is downloaded by tests or the CPU dry-run.
Real B100 construction requests OpenAI weights through OpenCLIP; arrange an
approved cache/network source yourself before real training. Missing weights
must fail, never fall back to random-backbone training.

## Data and file identity

The candidate pool layout is:

```text
MOSAIC/
  metadata.csv
  Cartoon/{0_real,1_fake}/...
  Chart/{0_real,1_fake}/...
  Photo/{0_real,1_fake}/...
```

Pool metadata uses `file_name,domain,label,sha256`. `0_real/1_fake` and numeric
0/1 are accepted. `Photo` is canonicalized to `Photo_and_poster`. Portable
relative paths are supported; foreign absolute Windows paths fail explicitly.

Training manifests use:

```csv
sample_id,image_path,domain,label,label_name,sha256,original_sha256,group_id,split,benchmark_id
train_001,../images/example.png,Cartoon,0,Real,<current-file-sha>,<source-file-sha>,verified-group-001,train,
```

`image_path` is relative to the manifest directory. `sha256` means the bytes
actually loaded; `original_sha256` means the original source-file identity.
Without conversion these are equal. Legacy manifests with `sha256` for the
source and `standardized_sha256` for the converted JPEG are normalized explicitly.
Every referenced file is hashed before training/profile creation, supplied hashes
must match, and duplicate paths, actual/source identities and sample IDs fail.
Train/Dev exact identity and supplied trusted `group_id` overlap fail.
`group_id_claim` is not treated as verified grouping. Near-duplicate/source-group
isolation still requires an externally reviewed protocol.

### Explicit conversion; no automatic split

Obtain **independently selected Train and Dev lists** first. A selection CSV
must contain `file_name` paths relative to the pool root (plus optional matching
`split`). This tool does not invent Train/Dev/Test membership:

```bash
python tools/convert_mosaic_manifest.py --metadata /data/MOSAIC/metadata.csv --image-root /data/MOSAIC --selection-manifest /data/splits/train_selection.csv --split train --output data/manifests/mosaic_train.csv
python tools/convert_mosaic_manifest.py --metadata /data/MOSAIC/metadata.csv --image-root /data/MOSAIC --selection-manifest /data/splits/dev_selection.csv --split dev --output data/manifests/mosaic_dev.csv
```

Selections, pool bytes and output identities are checked; outputs cannot be
overwritten. Missing selection/Dev manifests cause errors, not automatic resplits.
The 6718-image candidate pool is not automatically approved public data, and is
not the paper's historical 6000-image benchmark. Rights and quality need review.

The copied `tools/prepare_mosaic.py` is an **optional legacy JPEG preparation**
tool. Use it only on an independently selected Train-only directory, with
`--input-is-train-pool`. Conversion/re-encoding is explicit, never used by the
benchmark tool. The metadata converter above preserves original image bytes.

## Training commands

Run from the package root. YAML defaults resolve relative to this package;
explicit CLI paths resolve relative to the current directory. Always specify
separate outputs and independently selected, disjoint Train/Dev manifests.

### 1. B100 reference

```bash
python -u scripts/train_reference.py --config configs/visual_expert_reported.yaml --train-manifest data/manifests/mosaic_train.csv --dev-manifest data/manifests/mosaic_dev.csv --output outputs/reference/reference.pth --device cuda:0
```

### 2. Fixed risk profile (same Train only)

```bash
python -u scripts/build_risk_profile.py --config configs/visual_expert_reported.yaml --manifest data/manifests/mosaic_train.csv --checkpoint outputs/reference/reference.pth --output outputs/risk/risk_profile.json --device cuda:0
```

The parent must be the completed B100 endpoint with matching config, Train CSV
SHA and ordered content identity. Dev is never used for risk statistics.

### 3. B50 real-side repair

```bash
python -u scripts/train_visual_expert.py --config configs/visual_expert_reported.yaml --train-manifest data/manifests/mosaic_train.csv --dev-manifest data/manifests/mosaic_dev.csv --reference-checkpoint outputs/reference/reference.pth --risk-profile outputs/risk/risk_profile.json --output outputs/visual_expert/visual_expert.pth --device cuda:0
```

The fixed final endpoint is saved; Dev is reported after training, not used for
checkpoint selection, calibration or repair weights. There is no Test argument.
CLI `--max-batches` is for explicit shortened debugging only; shortened B100
checkpoints cannot pass the normal B100 parent gate.

Checkpoint format_version1 retains model_state_dict, optimizer_state_dict,
config, feature_dim and metadata. Added provenance binds the completed stage,
Train CSV/content SHA, parent SHA and profile SHA. Loading is strict and uses
PyTorch's restricted `weights_only=True` path. Old checkpoints lacking source
proof are not silently accepted as B100 parents. These are endpoint continuation
checkpoints, not exact mid-run RNG/sampler/GradScaler recovery snapshots.

Reserve enough disk for **both** full OpenCLIP endpoints, AdamW state and atomic
save temporary files (plan at least 6GiB free; actual size depends on versions).
Do not overwrite historical runs. Training refuses an existing checkpoint unless
`--overwrite` is explicitly supplied; a risk JSON is never overwritten.

## CPU verification — no pretrained model required

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B -m pytest tests -q -p no:cacheprovider
CUDA_VISIBLE_DEVICES='' python -B tools/cpu_smoke.py
```

The smoke command creates only tiny synthetic images/manifests in a temporary
directory, invokes B100 `--device cpu --dry-run`, and loads no model or weights.
It prints the artifact directory. Tests include a tiny mock-model B100100 /
profile / B5050 integration; these are synthetic CPU tests, **not real Expert1
training and not reproduction results**. No test uses real datasets or GPUs.

For an existing, approved Train/Dev pair, data-only validation is also available:

```bash
python -B scripts/train_reference.py --config configs/visual_expert_reported.yaml --train-manifest data/manifests/mosaic_train.csv --dev-manifest data/manifests/mosaic_dev.csv --device cpu --dry-run
```

## Separate NEW Benchmark6000 manifest tool

This is a newly sampled **test-only** candidate benchmark, not the historical
MOSAIC test and not evidence that historical paper scores apply. It never moves,
copies or re-encodes images. It validates the whole metadata pool's file paths,
labels and current bytes, then excludes specified historical identities/groups,
audits duplicate identities, and samples exactly1000 distinct images per cell.
Any insufficient cell is an error: Cartoon/Real has only1000 candidates in the
current pool, so even one exclusion there can make the requested freeze impossible.

```bash
python tools/build_mosaic_benchmark.py --metadata /data/MOSAIC/metadata.csv --image-root /data/MOSAIC --exclude-manifest /data/splits/historical_train.csv --exclude-manifest /data/splits/historical_dev.csv --exclude-manifest /data/splits/historical_test.csv --seed 2026 --output-dir outputs/benchmark6000_seed2026
```

It writes `mosaic_benchmark_6000.csv`, `mosaic_remaining.csv`,
`mosaic_rejected.csv`, `benchmark_freeze.json` and its SHA sidecar. The freeze
records seed, input/output SHA, selected identities, rules, counts and exclusions.
Output order is fixed; remaining count is **not guaranteed to be718** after
exclusions/deduplication. No replacement, silent padding or performance selection.
Missing/malformed exclusion identities fail rather than becoming an empty set.
With no exclusion arguments, the report explicitly says none were supplied.
Near duplicates, source-group isolation and publication rights remain NOT_VERIFIED.

Existing output directories are refused; reuse is read-only and never resamples:

```bash
python tools/build_mosaic_benchmark.py --verify --metadata /data/MOSAIC/metadata.csv --image-root /data/MOSAIC --exclude-manifest /data/splits/historical_train.csv --exclude-manifest /data/splits/historical_dev.csv --exclude-manifest /data/splits/historical_test.csv --output-dir outputs/benchmark6000_seed2026
```

The benchmark rows carry `split=test` and `benchmark_id`; training, risk and Dev
loaders reject them. Do not relabel benchmark rows as Train, Calibration or Dev.

## Acceptance boundaries

CPU syntax/import, numerical, manifest, sampling, mock checkpoint and synthetic
CLI checks are implemented. **Real OpenAI weight initialization, actual ViT-L/14
GPU/FP16 training, real-pool Benchmark6000 creation and paper metric reproduction
have not been accepted in this extraction.** No full training starts automatically.
The author must provide independent split manifests, approve data rights, resolve
the manuscript risk-definition discrepancy, and authorize any real GPU validation.
