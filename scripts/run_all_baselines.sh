#!/usr/bin/env bash
# Run all baseline models against the benchmark query set.
set -euo pipefail

QUERIES="${1:-data/queries/sample.json}"
CORPUS="${2:-data/corpus_manifests/sample_manifest.json}"
OUTPUT="data/results"

echo "=== MomentBench Baseline Runs ==="
echo "Queries: $QUERIES"
echo "Corpus:  $CORPUS"
echo "Output:  $OUTPUT"
echo ""

CLOSED_MODELS="gemini openai claude mixpeek twelvelabs mixedbread jina-clip"
OSS_MODELS="qwen-vl e5-omni hf-vlm clip"

if [ "${OSS_ONLY:-}" = "1" ]; then
    MODELS="$OSS_MODELS"
elif [ "${CLOSED_ONLY:-}" = "1" ]; then
    MODELS="$CLOSED_MODELS"
else
    MODELS="$CLOSED_MODELS $OSS_MODELS"
fi

for model in $MODELS; do
    echo "--- Running $model ---"
    momentbench run "$model" -q "$QUERIES" -c "$CORPUS" -o "$OUTPUT" || \
        echo "WARN: $model failed, continuing..."
    echo ""
done

echo "=== Leaderboard ==="
momentbench leaderboard "$OUTPUT"
