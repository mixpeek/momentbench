#!/usr/bin/env bash
# Ingest the corpus into Mixpeek from a manifest file.
set -euo pipefail

MANIFEST="${1:-data/corpus_manifests/sample_manifest.json}"

echo "=== MomentBench Corpus Ingestion ==="
echo "Manifest: $MANIFEST"
echo ""

momentbench ingest -m "$MANIFEST"
echo ""
echo "Running decomposition pipelines..."
momentbench decompose -m "$MANIFEST"
echo ""
echo "Generating attribution file..."
momentbench attribution -m "$MANIFEST" -o ATTRIBUTION.md
echo "Done."
