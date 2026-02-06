#!/bin/bash
# Run smll benchmarks and generate visualizations

set -e

# Resolve the directory this script lives in
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=================================================="
echo "SMLL Compression Benchmarks"
echo "=================================================="

# Default values
MODEL_REPO="${MODEL_REPO:-QuantFactory/SmolLM2-360M-GGUF}"
MODEL_FILE="${MODEL_FILE:-*Q4_0.gguf}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
VERIFY="${VERIFY:-false}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK=true
            shift
            ;;
        --verify)
            VERIFY=true
            shift
            ;;
        --model-repo)
            MODEL_REPO="$2"
            shift 2
            ;;
        --model-file)
            MODEL_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --quick         Run a quick subset of benchmarks"
            echo "  --verify        Verify decompression (slower)"
            echo "  --model-repo    Hugging Face model repo (default: $MODEL_REPO)"
            echo "  --model-file    Model filename pattern (default: $MODEL_FILE)"
            echo "  --output-dir    Output directory (default: $OUTPUT_DIR)"
            echo "  --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build verify flag
VERIFY_FLAG=""
if [ "$VERIFY" = true ]; then
    VERIFY_FLAG="--verify"
fi

echo ""
echo "Configuration:"
echo "  Model repo:  $MODEL_REPO"
echo "  Model file:  $MODEL_FILE"
echo "  Output dir:  $OUTPUT_DIR"
echo "  Verify:      $VERIFY"
echo ""

# Run benchmarks
if [ "$QUICK" = true ]; then
    echo "Running quick benchmarks (content types only)..."
    uv run --project "$REPO_ROOT" python "$SCRIPT_DIR/benchmarks.py" \
        --model-repo "$MODEL_REPO" \
        --model-file "$MODEL_FILE" \
        --benchmark content \
        $VERIFY_FLAG
else
    echo "Running all benchmarks..."
    uv run --project "$REPO_ROOT" python "$SCRIPT_DIR/benchmarks.py" \
        --model-repo "$MODEL_REPO" \
        --model-file "$MODEL_FILE" \
        --output "$OUTPUT_DIR/results.json" \
        $VERIFY_FLAG

    echo ""
    echo "Generating visualizations..."
    uv run --project "$REPO_ROOT" python "$SCRIPT_DIR/visualize_benchmarks.py" \
        "$OUTPUT_DIR/results.json" \
        --output-dir "$OUTPUT_DIR/charts"
fi

echo ""
echo "=================================================="
echo "Benchmarks complete!"
echo "=================================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
if [ "$QUICK" != true ]; then
    echo "  - JSON data:  $OUTPUT_DIR/results.json"
    echo "  - Charts:     $OUTPUT_DIR/charts/"
    echo "  - Summary:    $OUTPUT_DIR/charts/summary.md"
fi
