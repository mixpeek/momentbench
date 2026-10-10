# Runs the MomentBench harness. Mount the corpus and pass provider keys at run time:
#   docker build -t momentbench .
#   docker run --rm -e ANTHROPIC_API_KEY -v /path/to/videos:/corpus -v "$PWD/out:/out" \
#     momentbench agentic claude-opus-5-5 -q data/queries/sample.json \
#     -c data/corpus_manifests/sample_manifest.json --corpus-dir /corpus -o /out
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY momentbench ./momentbench
RUN pip install --no-cache-dir ".[models]"
COPY configs ./configs
COPY data ./data

ENTRYPOINT ["momentbench"]
CMD ["--help"]
