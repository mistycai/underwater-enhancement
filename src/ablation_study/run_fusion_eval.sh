
set -euo pipefail

VAL_IMG_DIR="./data/ruod_yolo_val_sub_500_raw/images/val"
VAL_LABELS="./data/ruod_yolo_val_sub_500_raw/labels/val"

# Check if paths exist
if [[ ! -d "$VAL_IMG_DIR" ]]; then
    echo "[ERROR] Validation images directory not found: $VAL_IMG_DIR"
    exit 1
fi

if [[ ! -d "$VAL_LABELS" ]]; then
    echo "[ERROR] Validation labels directory not found: $VAL_LABELS"
    exit 1
fi

# Model path - update this to your actual model path
MODEL="runs_yolov8m_yolov8n/ruod/yolov8m_raw/weights/best.pt"

if [[ ! -f "$MODEL" ]]; then
    echo "[WARNING] Model not found: $MODEL"
    echo "Please update MODEL variable to your actual model path"
    echo "Continuing anyway - will fail at YOLO evaluation stage"
fi

OUTPUT_DIR="./results/fusion_ablation"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "FUSION ABLATION STUDY"
echo "=============================================="
echo ""
echo "Validation images: $VAL_IMG_DIR"
echo "Validation labels: $VAL_LABELS"
echo "Model: $MODEL"
echo ""
echo "Best standalone parameters:"
echo "  - RCC: alpha=0.5 (NOT WRCC)"
echo "  - CLAHE: clip=6.0, tile=32"
echo "  - Denoise: gauss=3, median=3, bilateral=3, sharpen=0.1"
echo ""

# ============================================================================
# PREPARE ENHANCED IMAGES FOR MAP EVALUATION
# ============================================================================
echo ""
echo "=============================================="
echo "PREPARING IMAGES FOR MAP EVALUATION"
echo "=============================================="

MAP_DIR="$OUTPUT_DIR/map_eval"
mkdir -p "$MAP_DIR"

# Best parameters
RCC_ALPHA=0.5
CLAHE_CLIP=6.0
CLAHE_TILE=32
DENOISE_GAUSS=3
DENOISE_MEDIAN=3
DENOISE_BILATERAL=3
DENOISE_SHARPEN=0.1

# Configurations for mAP evaluation
declare -A CONFIGS=(
    ["raw"]="none"
    ["rcc_only"]="rcc"
    ["clahe_only"]="clahe"
    ["denoise_only"]="denoise"
    ["2input_rcc_clahe"]="2input"
    ["3input_rcc_clahe_denoise"]="3input"
)

for config_name in "${!CONFIGS[@]}"; do
    config_type="${CONFIGS[$config_name]}"
    
    DS_ROOT="$MAP_DIR/$config_name"
    IMG_OUT="$DS_ROOT/images"
    
    if [[ -d "$IMG_OUT" ]] && [[ $(ls -1 "$IMG_OUT"/*.jpg 2>/dev/null | wc -l) -gt 100 ]]; then
        echo "[SKIP] $config_name already exists ($(ls -1 "$IMG_OUT"/*.jpg 2>/dev/null | wc -l) images)"
        continue
    fi
    
    echo "Preparing: $config_name"
    mkdir -p "$IMG_OUT"
    
    # Use standalone Python script (more reliable than heredoc)
    export RCC_ALPHA CLAHE_CLIP CLAHE_TILE DENOISE_GAUSS DENOISE_MEDIAN DENOISE_BILATERAL DENOISE_SHARPEN
    python3 src/ablation_study/fusion_enhance_images.py "$VAL_IMG_DIR" "$IMG_OUT" "$config_type"
    
    # Create labels symlink (remove directory if exists, then create symlink)
    if [[ -d "$DS_ROOT/labels" ]] && [[ ! -L "$DS_ROOT/labels" ]]; then
        rm -rf "$DS_ROOT/labels"
    fi
    ln -sf "$(cd "$(dirname "$VAL_LABELS")" && pwd)/$(basename "$VAL_LABELS")" "$DS_ROOT/labels"
    
    # Create YAML file
    cat > "$DS_ROOT/dataset.yaml" << EOF
path: $(cd "$DS_ROOT" && pwd)
train: images
val: images
test:

nc: 10
names: ['holothurian', 'echinus', 'scallop', 'starfish', 'fish', 'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish']
EOF
    
    echo "  ✓ Created: $DS_ROOT"
    echo "    Images: $(ls -1 "$IMG_OUT"/*.jpg 2>/dev/null | wc -l)"
    echo "    YAML: $DS_ROOT/dataset.yaml"
    echo ""
done

echo ""
echo "=============================================="
echo "RUNNING YOLO MAP EVALUATION"
echo "=============================================="

if [[ ! -f "$MODEL" ]]; then
    echo "[ERROR] Model not found: $MODEL"
    echo "Please update MODEL variable in the script"
    exit 1
fi

RESULTS_FILE="$MAP_DIR/fusion_map_results.txt"

{
    echo "=============================================="
    echo "FUSION MAP ABLATION RESULTS"
    echo "Date: $(date)"
    echo "Model: $MODEL"
    echo "=============================================="
    echo ""
    echo "Best Parameters Used:"
    echo "  RCC: alpha=$RCC_ALPHA (basic RCC, not WRCC)"
    echo "  CLAHE: clip=$CLAHE_CLIP, tile=$CLAHE_TILE"
    echo "  Denoise: gauss=$DENOISE_GAUSS, median=$DENOISE_MEDIAN, bilateral=$DENOISE_BILATERAL, sharpen=$DENOISE_SHARPEN"
    echo ""
} | tee "$RESULTS_FILE"

for config_name in raw rcc_only clahe_only denoise_only 2input_rcc_clahe 3input_rcc_clahe_denoise; do
    DS_ROOT="$MAP_DIR/$config_name"
    
    if [[ ! -f "$DS_ROOT/dataset.yaml" ]]; then
        echo "[SKIP] $config_name - no dataset.yaml"
        continue
    fi
    
    echo "" | tee -a "$RESULTS_FILE"
    echo "=== $config_name ===" | tee -a "$RESULTS_FILE"
    
    yolo val model="$MODEL" data="$DS_ROOT/dataset.yaml" \
        imgsz=640 conf=0.001 iou=0.7 \
        project="$MAP_DIR/runs" name="$config_name" \
        save=False plots=False \
        2>&1 | tee -a "$RESULTS_FILE"
done

echo ""
echo "=============================================="
echo "GENERATING SUMMARY TABLES"
echo "=============================================="

python3 << 'PYEOF'
import os
import csv
from pathlib import Path

map_dir = Path("./results/fusion_ablation/map_eval")
runs_dir = map_dir / "runs"

results = []

# Expected order for table
config_order = ['raw', 'rcc_only', 'clahe_only', 'denoise_only', '2input_rcc_clahe', '3input_rcc_clahe_denoise']
config_display = {
    'raw': 'Raw (baseline)',
    'rcc_only': 'RCC only (α=0.5)',
    'clahe_only': 'CLAHE only (clip=6, tile=32)',
    'denoise_only': 'Denoise only (3,3,3,0.1)',
    '2input_rcc_clahe': '2-Input Fusion: RCC + CLAHE',
    '3input_rcc_clahe_denoise': '3-Input Fusion: RCC + CLAHE + Denoise',
}

# Collect results
if runs_dir.exists():
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        results_csv = run_dir / "results.csv"
        if not results_csv.exists():
            continue
        
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                row = rows[-1]
                results.append({
                    'config': run_dir.name,
                    'display': config_display.get(run_dir.name, run_dir.name),
                    'mAP50': float(row.get('metrics/mAP50(B)', 0)),
                    'mAP50-95': float(row.get('metrics/mAP50-95(B)', 0)),
                    'precision': float(row.get('metrics/precision(B)', 0)),
                    'recall': float(row.get('metrics/recall(B)', 0)),
                })

if not results:
    print("No results found. Make sure YOLO evaluation completed successfully.")
    exit(0)

# Sort by config order
def sort_key(x):
    try:
        return config_order.index(x['config'])
    except ValueError:
        return 999

results.sort(key=sort_key)

# Find best
best_map = max(r['mAP50'] for r in results)

# Print summary table
print("\n" + "="*90)
print("FUSION ABLATION RESULTS")
print("="*90)
print(f"{'Method':<45} {'mAP50':>10} {'mAP50-95':>12} {'Precision':>10} {'Recall':>10}")
print("-"*90)

for r in results:
    marker = " ★" if abs(r['mAP50'] - best_map) < 0.0001 else ""
    print(f"{r['display']:<45} {r['mAP50']:>10.4f} {r['mAP50-95']:>12.4f} "
          f"{r['precision']:>10.4f} {r['recall']:>10.4f}{marker}")

print("-"*90)

# Find baseline and improvements
raw_result = next((r for r in results if r['config'] == 'raw'), None)
best_result = max(results, key=lambda x: x['mAP50'])

if raw_result:
    print(f"\nBaseline (raw): mAP50 = {raw_result['mAP50']:.4f}")
    print(f"Best method: {best_result['display']}")
    print(f"  mAP50 = {best_result['mAP50']:.4f} (Δ = {best_result['mAP50'] - raw_result['mAP50']:+.4f}, "
          f"{100*(best_result['mAP50'] - raw_result['mAP50'])/raw_result['mAP50']:+.1f}%)")

# Generate LaTeX table
latex = r"""\begin{table}[t]
\centering
\caption{Fusion ablation study comparing individual components and fusion methods on RUOD validation set. 
Best parameters: RCC ($\alpha$=0.5), CLAHE (clip=6.0, tile=32), Denoise (gaussian=3, median=3, bilateral=3, sharpen=0.1). 
The ★ indicates the best performing method.}
\label{tab:fusion_ablation}
\begin{tabular}{lcccc}
\toprule
Method & mAP$_{50}$ & mAP$_{50{-}95}$ & Precision & Recall \\
\midrule
"""

for r in results:
    name = r['display'].replace('_', r'\_').replace('α', r'$\alpha$')
    map50 = f"{r['mAP50']:.4f}"
    if abs(r['mAP50'] - best_map) < 0.0001:
        map50 = r"\textbf{" + map50 + r"}\,$^\star$"
    latex += (f"{name} & {map50} & {r['mAP50-95']:.4f} & "
             f"{r['precision']:.4f} & {r['recall']:.4f} \\\\\n")

latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

# Save files
output_path = map_dir / "fusion_ablation_table.tex"
with open(output_path, 'w') as f:
    f.write(latex)
print(f"\n✓ LaTeX table saved: {output_path}")

# Save CSV
csv_path = map_dir / "fusion_ablation_summary.csv"
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['config', 'display_name', 'mAP50', 'mAP50-95', 'precision', 'recall'])
    for r in results:
        w.writerow([r['config'], r['display'], f"{r['mAP50']:.4f}", 
                   f"{r['mAP50-95']:.4f}", f"{r['precision']:.4f}", f"{r['recall']:.4f}"])
print(f"✓ CSV saved: {csv_path}")

# Print LaTeX for easy copy-paste
print("\n" + "="*90)
print("LATEX TABLE (copy-paste ready)")
print("="*90)
print(latex)

PYEOF

echo ""
echo "=============================================="
echo "FUSION ABLATION COMPLETE"
echo "=============================================="
echo ""
echo "Results directory: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  Summary CSV:  $OUTPUT_DIR/map_eval/fusion_ablation_summary.csv"
echo "  LaTeX table:  $OUTPUT_DIR/map_eval/fusion_ablation_table.tex"
echo "  Full logs:    $OUTPUT_DIR/map_eval/fusion_map_results.txt"
echo ""
echo "Enhanced images saved in: $OUTPUT_DIR/map_eval/<config>/images/"
echo ""