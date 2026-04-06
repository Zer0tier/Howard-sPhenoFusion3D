# SWE Tasks Quickstart (Intrinsics Recovery Sprint)

This document maps the Software Dev tasks to runnable commands.

## T-01 Baseline runs

Run two baseline focal lengths (`1108`, `900`) and write:
- `baseline/metrics.json`
- `baseline/baseline_fx1108.ply`
- `baseline/baseline_fx900.ply`

```bash
python experiments/baseline/run_baseline.py \
  --rgb-dir "data/main/test_plant_rs13_1/rgb" \
  --depth-dir "data/main/test_plant_rs13_1/depth" \
  --output-dir "baseline"
```

Optional overrides:
- `--fx-values 1108 900 1000`
- `--step 2`
- `--depth-scale 1000`

## T-08 Evaluation harness

Compare multiple intrinsics files on the first 30 frames and write:
- `evaluation/comparison_results.csv`
- `evaluation/comparison_results.json`
- `evaluation/ply_outputs/*.ply`

```bash
python evaluation/compare_intrinsics.py \
  --rgb-dir "data/main/test_plant_rs13_1/rgb" \
  --depth-dir "data/main/test_plant_rs13_1/depth" \
  --frames 30 \
  "baseline_K.json" "candidate_intrinsics.json" "optimized_K.json" "sfm_K.json"
```

The script prints a Markdown table to stdout for quick report copy/paste.

## Notes

- If your dataset path differs, only change `--rgb-dir` and `--depth-dir`.
- If depth units are in millimetres, use `--depth-scale 1000`.
- If depth units are metres (ICL-NUIM/TUM style), use `--depth-scale 1`.
