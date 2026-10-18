# Root-repository Visual Expert integration check

The later, bounded Visual code-only acceptance entry is documented in the root
`README.md` (Visual Expert and Tests sections); use
`test_visual_release_unittest.py` for that check.
The29-test whole-repository run described below is an earlier audit record,
not the code-only acceptance command. It includes a full synthetic100/50 run
and non-Visual counterexamples and is not executed by the bounded release suite.

Checked on 2026-10-18. This is an engineering connectivity check, not a claim
that the paper's trained models or reported benchmark numbers were reproduced.

## Active entry points

Run the four Visual Expert scripts from the repository root:

1. `scripts/train_reference.py`
2. `scripts/build_risk_profile.py`
3. `scripts/train_visual_expert.py`
4. `scripts/evaluate_visual_expert.py`

They import the active `decd/visual_expert/*.py` modules. The independently
packaged copy under `decd/visual_expert/decd/` and its own scripts, tools,
configuration and tests were already present and have been retained unchanged.
That copy is not the active root package. Do not assume testing that copy
tests the root scripts.

Engineering safeguards from the independent copy were integrated into the
active Visual Expert modules: explicit manifest roles, image-byte SHA checks,
source/standardized identity separation, exact-file/group isolation checks,
strict B100 and AdamW continuation checks, finite-value guards, inference
count/order checks, and explicit evaluation overwrite protection.

Training still uses a single B100 endpoint, a fixed reference risk profile,
and a single B50 Real-only continuation. Existing reported configuration,
sampler, model freezing, optimizer mathematics, loss formulas and preprocessing
have not been changed. Evaluation accepts generator names such as BigGAN as
reporting domains; training remains restricted to the three MOSAIC domains.

## Data and checkpoint contracts

- Real=0, Fake=1. Training domains are Cartoon, Chart, Photo_and_poster;
  the Photo alias is accepted.
- `image_path` and `file_name` are accepted. Relative paths resolve against
  the manifest directory. Public metadata whose paths are relative to a
  separate image root must be explicitly converted or rebased first.
- `sha256` identifies bytes actually loaded. Legacy `standardized_sha256`
  identifies standardized bytes; legacy `sha256` then remains source identity.
  `original_sha256` is a separate source-identity field, not another hash of
  the transformed training tensor.
- When supplied, split/benchmark roles and group_id are checked. Missing group
  metadata and near-duplicate/source-family isolation remain NOT_VERIFIED.
  No absent role field can establish that data are truly independent.
- Do not feed a test-only benchmark into training, risk profiling or Dev.
- Both experts should receive the same explicit, nonempty, stable sample_id
  manifest. The non-Visual data/cache/join pipeline does not establish this
  identity or split provenance on behalf of Visual Expert.
- B50 requires a completed B100 checkpoint, matching Train CSV/content/config,
  complete strict model state, and the corresponding AdamW moments/step.
  A legacy checkpoint missing provenance is rejected, not silently converted.
  Preserve historical artifacts; independently verify their provenance before
  planning an explicit migration, or produce a fresh B100 chain.
- Evaluator output files cannot be overwritten implicitly. Prefer a new output
  directory; `--overwrite` is an explicit opt-in.

## CPU check

From the repository root, with the existing CPU PyTorch dependencies:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
python -B -m unittest discover -s decd/visual_expert/tests -p test_repository_integration_unittest.py -v
```

Result on this machine: 29 tests run, **24 normal checks passed and 5 known
non-Visual counterexamples reproduced as expected failures**. They are not 29
successful product acceptance checks. The counterexamples expose missing MLLM
Train/Dev isolation, NaN feature acceptance, label-conflicting signal joins,
abstention tie behavior, and inconsistent nondefault per-domain thresholds.
Those implementations were not modified.

The synthetic integration test uses a tiny CLIP-shaped model and deterministic
8-dimensional fake frozen semantic features. It executes the root B100100 ->
Risk -> B5050 -> predictions -> adapter -> signal join -> dispatcher -> fusion
CLI chain, checks AdamW continuation to step150, and asserts CUDA is not
initialized. Its synthetic scores are not scientific results. Another test
executes the data-only `--dry-run` without constructing any backbone.

All 68 Python sources compiled in memory, 28 root modules imported, and all
nine root script `--help` commands plus the preparation CLI passed. The README
CLI flag names exist; this does not verify its missing real data/weight inputs.
The original pytest suites were not executed because pytest is unavailable.
No dependency installation, model download, or actual GPU run was performed.

## Paper discrepancies intentionally not changed

For the supplied manuscript (pages4-7,12-13), the author should resolve:

- Eq(5) says mean squared hinge risk; reported YAML selects mean hinge risk.
- Eq(6) says ties give zero weights; current code gives all-one weights, and
  uses ordinary min-max division without an additive epsilon otherwise.
- Eq(7) adds epsilon to the weight sum; code clamps the sum below at 1e-12.
- Wording about alternating100/50 blocks does not fix the number of cycles;
  the scripts execute one B100->B50 sequence, not a repeated alternation loop.
- Domain labels also drive six-cell balanced sampling, not just profiling.
- The full system's documented FP16 measurements do not match the semantic
  extraction CLI's GPU bfloat16 default. No hardware measurements were repeated.
- Historical splits, trained models, three-seed results and full ablation,
  robustness and figure-generation pipelines are not bundled.
- The manuscript has TBD probability cells and a Figure?? cross-reference.

The existing mathematical choices have been preserved for author review; they
were not changed to force agreement with the paper. MLLM, dispatcher, root
configuration, root README, frozen/historical artifacts and the PDF are untouched.
