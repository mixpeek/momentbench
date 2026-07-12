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

for model in gemini openai claude mixpeek; do
    echo "--- Running $model ---"
    momentbench run "$model" -q "$QUERIES" -c "$CORPUS" -o "$OUTPUT" || \
        echo "WARN: $model failed, continuing..."
    echo ""
done

echo "=== Leaderboard ==="
momentbench leaderboard "$OUTPUT"
