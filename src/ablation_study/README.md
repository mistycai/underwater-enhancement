**Ablation Study**

This folder contains the ablation study runner for the underwater image
enhancement pipeline. The study systematically varies algorithm parameters to
measure their impact on objective quality metrics (UIQM, UCIQE, contrast).

**Purpose**:
- **Summary:** Run controlled experiments to evaluate how individual
  parameters (e.g. RCC alpha, CLAHE clip_limit, fusion levels) affect
  enhancement quality on the RUOD dataset.
- **Outputs:** JSON/CSV summaries and LaTeX tables suitable for inclusion in
  papers.

**Location**:
- **Script:** `src/ablation_study/ablation_study.py`

**Prerequisites**:
- **Python:** 3.8+ recommended.
- **Dependencies:** See the top-level `requirements.txt`. Typical packages used
  include `opencv-python`, `numpy`, `tqdm`, and the project's internal modules
  (fusion, metrics). Install with:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Quick Usage**:
- Sample images are collected from a folder tree under `--input-root` and a
  random subset is used (seedable via `--seed`). By default the script uses
  `200` samples (recommended for ablation studies).

Run a quick test (50 images):

```
python src/ablation_study/ablation_study.py --input-root data/RUOD/RUOD_pic/train --num-samples 50
```

Run a single experiment (RCC alpha sweep):

```
python src/ablation_study/ablation_study.py --input-root data/RUOD/RUOD_pic/train --experiment rcc_alpha --num-samples 200
```

Run multiple experiments:

```
python src/ablation_study/ablation_study.py --input-root data/RUOD/RUOD_pic/train --experiment rcc_alpha wb_lambda clahe_clip --num-samples 200
```

Run a grid search (fine-tuning multiple params):

```
python src/ablation_study/ablation_study.py --input-root data/RUOD/RUOD_pic/train --grid-search color_grid --num-samples 200
```

List available experiments:

```
python src/ablation_study/ablation_study.py --list-experiments
```

**Default behavior & outputs**:
- **Default output dir:** `ablation_results` (adjust with `--output-dir`).
- For each experiment `X` the script saves:
  - `X_results.json` — structured JSON with metrics and per-configuration stats.
  - `X_results.csv` — tabular CSV summary for quick plotting/analysis.
  - `X_table.tex` — LaTeX table ready to include in a manuscript.
- The script also prints a console summary showing mean UIQM, UCIQE and contrast
  for each parameter value and highlights best values.

**Experiments available**:
- The script includes many pre-configured experiments such as `rcc_alpha`,
  `wrcc_alpha`, `wrcc_window`, `wb_lambda`, `clahe_clip`, `clahe_tile`,
  `denoise_bilateral`, `denoise_sharpen`, `fusion_levels`, `fusion_mode`, and
  `color_method`. Use `--list-experiments` to see descriptions.

**Notes & tips**:
- The ablation runner depends on the `run_fusion` pipeline and metric
  computation (`compute_metrics`). If metrics are not available the script
  still runs but will omit metric fields.
- Recommended sample sizes are defined in the script (quick_test=50,
  ablation=200, thorough=500). Larger sample sizes increase statistical
  confidence but also runtime.
- Use `--quiet` / `-q` to suppress progress prints (useful when running many
  experiments in batch).

**Interpreting results**:
- Focus on mean metric deltas (improvement over original) and the
  `*_improved_rate` columns which report the percentage of images
  showing positive improvement.
- Use the generated LaTeX table (`*_table.tex`) to compare parameter values
  side-by-side in your paper.

**Next steps / integration**:
- Run a small quick-test first to validate environment and imports.
- Automate repeated runs (different seeds or additional datasets) by scripting
  calls to the above CLI.

**Contact / Citation**:
- If you use these experiments in published work, please cite the project and
  mention the RUOD dataset and the specific fusion/color-correction methods
  evaluated.
