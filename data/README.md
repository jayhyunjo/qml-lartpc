# data/

- `patches_31.npz` — 415,509 labelled 31×31 patches from the WC_AIML campaign (see
  `../docs/wc-aiml-dataset-inventory.md`). Built by `../scripts/build_patches.py`; verified in
  notebook 00. Raw charge — normalise at load time (`qmllar.preprocess`).

Loading the compressed arrays takes ~1–2 minutes (the `patches` array is 1.6 GB uncompressed).
