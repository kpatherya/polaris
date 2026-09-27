#!/bin/bash
#
# RUN ALL EXPERIMENTS
#
# This script runs all three matching methods (Baseline A, Baseline B, Full Pipeline)
# on a curated set of test frames to enable systematic comparison.
#
# Test Frames:
#   Autumn: 0, 305, 864, 1136, 1726
#   Winter: 0, 309, 491, 797, 1109
#
# Each method finds top-5 matches in the opposite season dataset.
# Results are logged to CSV files for quantitative analysis.
#
# Usage:
#   bash experiments/run_all_experiments.sh

set -e  # Exit on error

echo "========================================================================"
echo "EXPERIMENTAL EVALUATION PIPELINE"
echo "========================================================================"
echo ""

# Configuration
MODEL_PATH="${MODEL_PATH:-checkpoints/llava-fastvithd_0.5b_stage2}"
DETECTIONS="${DETECTIONS:-data/detections.json}"
LEGACY_DETECTIONS="artifacts/archive/experiments_legacy/outputs/detections.json"
WINTER_DEPTH="${WINTER_DEPTH:-/Volumes/KAUSAR/rover_dataset/2024-01-13/realsense_D435i/depth}"
AUTUMN_DEPTH="${AUTUMN_DEPTH:-/Volumes/KAUSAR/rover_dataset/2024-04-11/realsense_D435i/depth}"
RESULTS_DIR="${RESULTS_DIR:-artifacts/runs/latest}"
BASELINE_A_LOG="$RESULTS_DIR/baseline_a_results.csv"
BASELINE_B_LOG="$RESULTS_DIR/baseline_b_results.csv"
FULL_PIPELINE_LOG="$RESULTS_DIR/full_pipeline_results.csv"
BASELINE_A_VIS="$RESULTS_DIR/visualizations/baseline_a"
BASELINE_B_VIS="$RESULTS_DIR/visualizations/baseline_b"
FULL_PIPELINE_VIS="$RESULTS_DIR/visualizations/full_pipeline"

mkdir -p "$RESULTS_DIR"

# Test frame indices
AUTUMN_FRAMES=(0 305 864 1136 1726)
WINTER_FRAMES=(0 309 491 797 1109)

# Check if detections.json exists
if [ ! -f "$DETECTIONS" ]; then
    if [ -f "$LEGACY_DETECTIONS" ]; then
        echo "Warning: $DETECTIONS not found; falling back to $LEGACY_DETECTIONS"
        DETECTIONS="$LEGACY_DETECTIONS"
    else
        echo "Error: detection cache not found."
        echo "Looked for: $DETECTIONS"
        echo "You can also set a custom path: DETECTIONS=/path/to/detections.json bash experiments/run_all_experiments.sh"
        exit 1
    fi
fi

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Detections: $DETECTIONS"
echo "  Results dir: $RESULTS_DIR"
echo "  Autumn test frames: ${AUTUMN_FRAMES[@]}"
echo "  Winter test frames: ${WINTER_FRAMES[@]}"
echo ""

# Stage 1: Baseline A (Embedding + Semantic)
echo "========================================================================"
echo "STAGE 1: BASELINE A (EMBEDDING + SEMANTIC MATCHING)"
echo "========================================================================"
echo ""

echo "Processing Autumn → Winter matches..."
for idx in "${AUTUMN_FRAMES[@]}"; do
    echo "  Running autumn_$(printf '%04d' $idx)..."
    uv run polaris match --strategy baseline-a \
        --detections "$DETECTIONS" \
        --model-path "$MODEL_PATH" \
        --autumn-idx $idx \
        --top-k 5 \
        --visualize \
        --visualization-dir "$BASELINE_A_VIS" \
        --log-file "$BASELINE_A_LOG"
done

echo ""
echo "Processing Winter → Autumn matches..."
for idx in "${WINTER_FRAMES[@]}"; do
    echo "  Running winter_$(printf '%04d' $idx)..."
    uv run polaris match --strategy baseline-a \
        --detections "$DETECTIONS" \
        --model-path "$MODEL_PATH" \
        --winter-idx $idx \
        --top-k 5 \
        --visualize \
        --visualization-dir "$BASELINE_A_VIS" \
        --log-file "$BASELINE_A_LOG"
done

echo ""
echo "✓ Stage 1 complete: Results logged to $BASELINE_A_LOG"
echo ""

# Stage 2: Baseline B (Geometric-Only)
echo "========================================================================"
echo "STAGE 2: BASELINE B (GEOMETRIC-ONLY MATCHING)"
echo "========================================================================"
echo ""

echo "Processing Autumn → Winter matches..."
for idx in "${AUTUMN_FRAMES[@]}"; do
    echo "  Running autumn_$(printf '%04d' $idx)..."
    uv run polaris match --strategy baseline-b \
        --detections "$DETECTIONS" \
        --autumn-idx $idx \
        --top-k 5 \
        --method orb \
        --min-inliers 4 \
        --visualize \
        --visualization-dir "$BASELINE_B_VIS" \
        --log-file "$BASELINE_B_LOG"
done

echo ""
echo "Processing Winter → Autumn matches..."
for idx in "${WINTER_FRAMES[@]}"; do
    echo "  Running winter_$(printf '%04d' $idx)..."
    uv run polaris match --strategy baseline-b \
        --detections "$DETECTIONS" \
        --winter-idx $idx \
        --top-k 5 \
        --method orb \
        --min-inliers 4 \
        --visualize \
        --visualization-dir "$BASELINE_B_VIS" \
        --log-file "$BASELINE_B_LOG"
done

echo ""
echo "✓ Stage 2 complete: Results logged to $BASELINE_B_LOG"
echo ""

# Stage 3: Full Pipeline (Multi-Modal Fusion)
echo "========================================================================"
echo "STAGE 3: FULL PIPELINE (MULTI-MODAL FUSION)"
echo "========================================================================"
echo ""

echo "Processing Autumn → Winter matches..."
for idx in "${AUTUMN_FRAMES[@]}"; do
    echo "  Running autumn_$(printf '%04d' $idx)..."
    uv run polaris match --strategy full-pipeline \
        --detections "$DETECTIONS" \
        --model-path "$MODEL_PATH" \
        --winter-depth "$WINTER_DEPTH" \
        --autumn-depth "$AUTUMN_DEPTH" \
        --autumn-idx $idx \
        --top-k 5 \
        --visualize \
        --visualization-dir "$FULL_PIPELINE_VIS" \
        --log-file "$FULL_PIPELINE_LOG"
done

echo ""
echo "Processing Winter → Autumn matches..."
for idx in "${WINTER_FRAMES[@]}"; do
    echo "  Running winter_$(printf '%04d' $idx)..."
    uv run polaris match --strategy full-pipeline \
        --detections "$DETECTIONS" \
        --model-path "$MODEL_PATH" \
        --winter-depth "$WINTER_DEPTH" \
        --autumn-depth "$AUTUMN_DEPTH" \
        --winter-idx $idx \
        --top-k 5 \
        --visualize \
        --visualization-dir "$FULL_PIPELINE_VIS" \
        --log-file "$FULL_PIPELINE_LOG"
done

echo ""
echo "✓ Stage 3 complete: Results logged to $FULL_PIPELINE_LOG"
echo ""

# Summary
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "========================================================================"
echo ""
echo "Results:"
echo "  Baseline A CSV:     $BASELINE_A_LOG"
echo "  Baseline B CSV:     $BASELINE_B_LOG"
echo "  Full Pipeline CSV:  $FULL_PIPELINE_LOG"
echo "  Visualizations:     $RESULTS_DIR/visualizations/"
echo ""
echo "Statistics:"
echo "  Test frames:        ${#AUTUMN_FRAMES[@]} autumn + ${#WINTER_FRAMES[@]} winter = $((${#AUTUMN_FRAMES[@]} + ${#WINTER_FRAMES[@]})) total"
echo "  Matches per frame:  5"
echo "  Total comparisons:  $(((${#AUTUMN_FRAMES[@]} + ${#WINTER_FRAMES[@]}) * 5)) matches logged per method"
echo ""
echo "Next steps:"
echo "  1. Analyze CSV files to compare methods quantitatively"
echo "  2. Review visualizations in $RESULTS_DIR/visualizations/"
echo "  3. Validate any challenging failure cases manually"
echo ""
