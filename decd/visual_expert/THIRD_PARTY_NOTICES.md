# Licensing boundaries

The existing repository-root `LICENSE` is Apache-2.0 and is retained unchanged
for the first-party code and documentation in `RELEASE_FILES.txt`. This release
does not grant rights beyond those owned by its contributors. The standalone
code archive omits Git history. Its contents do not certify historical ownership
or provenance of previously tracked assets.

No dependency source or pretrained weight file is vendored in the ZIP. Install
dependencies separately and retain their own notices when redistributing them:

| Component | License / primary source |
|---|---|
| OpenCLIP 3.3.0 | [MIT](https://github.com/mlfoundations/open_clip/blob/v3.3.0/LICENSE) |
| Original OpenAI CLIP implementation | [MIT](https://github.com/openai/CLIP/blob/main/LICENSE) |
| PyTorch | BSD-3-Clause; see installed distribution notices |
| NumPy | BSD-3-Clause and bundled-component notices; see installed distribution |
| Pillow | MIT-CMU in the tested installation; retain upstream notices |
| PyYAML | MIT |
| scikit-learn | BSD-3-Clause |

These are dependency notices, not a new license selection. Binary wheels can
include separately licensed components; consult their complete license files.
No upstream implementation has been copied into the local initialization
loader; it uses PyTorch public APIs and checks the documented CLIP state schema.

The training-code license is separate from pretrained-model usage terms and
dataset rights. An initialization file is not an author's trained B100/B50
checkpoint. Users must have permission to use their chosen weights and images;
MOSAIC metadata and Benchmark manifests do not confer redistribution rights.
This release distributes neither weights nor datasets. Authors must confirm
first-party provenance and any notices required by their actual source inputs
before public publication; no license/history audit here establishes ownership.
