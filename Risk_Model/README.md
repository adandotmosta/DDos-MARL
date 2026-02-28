# Risk_Model Workflow

This project is now split as:

- `BCC-2024/`: CSV data only
- `Risk_Model/`: model code + artifacts

## Data Location

The scripts in `Risk_Model/` now default to reading CSVs from:

- `../BCC-2024/`

So you do not need to pass `--base-dir` unless your data is elsewhere.

## Main Scripts

- Random Forest eval: `Risk_Model/baseline_eval_rf.py`
- LightGBM train/eval + save bundle: `Risk_Model/train_eval_save_lgbm.py`
- Runtime MARL inference helper: `Risk_Model/lgbm_runtime_infer.py`

## Rerun Commands

Run from repository root.

### 1) Random Forest (day split)

```bash
python Risk_Model/baseline_eval_rf.py
```

### 2) LightGBM (random split) + save artifacts

```bash
python Risk_Model/train_eval_save_lgbm.py \
  --split-mode random \
  --backend lightgbm \
  --model-bundle-out Risk_Model/artifacts/vae/v1/models/lgbm_bundle.pkl \
  --metrics-json-out Risk_Model/artifacts/vae/v1/models/lgbm_metrics.json
```

### 3) MARL runtime inference check

```bash
python Risk_Model/lgbm_runtime_infer.py \
  --bundle-path Risk_Model/artifacts/vae/v1/models/lgbm_bundle.pkl
```

## Where Artifacts Are Saved

Use and keep model outputs under:

- `Risk_Model/artifacts/vae/v1/models/`

Recommended files:

- `lgbm_bundle.pkl`
- `lgbm_metrics.json`
