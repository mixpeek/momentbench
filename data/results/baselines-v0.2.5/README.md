# Transcript baselines on the 0.2.5 scored set

Prediction files and scored run files for four systems on the three domains: BM25 over
transcripts, BM25 that abstains when its best match scores under 10, MiniLM dense
retrieval over the same transcript cues, and the never-answer floor. None of them looks
at a frame. The BM25 threshold and the MiniLM baseline were contributed in
[#10](https://github.com/mixpeek/momentbench/pull/10) and
[#11](https://github.com/mixpeek/momentbench/pull/11); the numbers here were regenerated
by the maintainers from the commands below and match the contributed hit counts.

To cite these numbers, cite this directory at the commit that added it, and the
MomentBench release in [`CITATION.cff`](../../../CITATION.cff).

## Results

Positive-query Recall@1 is the leaderboard's headline number. Net score is +1 for a
correct moment or a correct no-match answer, -1 for a wrong moment and 0 for abstaining
on a query that has a match.

| Domain | System | Positive Recall@1 (hits) | No-answer accuracy | Net score |
|---|---|---:|---:|---:|
| Film | BM25 | 9.5% (10/105) | 0.0% | -0.85 |
| Film | BM25, abstain below 10 | 9.5% (10/105) | 96.0% | +0.07 |
| Film | MiniLM | 5.7% (6/105) | 0.0% | -0.91 |
| Film | Never answer | 0.0% (0/105) | 100.0% | +0.19 |
| Aerospace | BM25 | 17.3% (24/139) | 0.0% | -0.69 |
| Aerospace | BM25, abstain below 10 | 17.3% (24/139) | 80.0% | -0.06 |
| Aerospace | MiniLM | 12.9% (18/139) | 0.0% | -0.77 |
| Aerospace | Never answer | 0.0% (0/139) | 100.0% | +0.10 |
| Robotics | BM25 | 12.5% (5/40) | 0.0% | -0.80 |
| Robotics | BM25, abstain below 10 | 10.0% (4/40) | 77.8% | -0.12 |
| Robotics | MiniLM | 10.0% (4/40) | 0.0% | -0.84 |
| Robotics | Never answer | 0.0% (0/40) | 100.0% | +0.18 |

The no-answer labels were screened by model, not by people, so no-answer accuracy and
net score are exploratory ([`docs/labels.md`](../../../docs/labels.md)).

## Files

- `predictions/<system>_<domain>.json`: the prediction files, in the format of
  [`docs/prediction-files.md`](../../../docs/prediction-files.md).
- `runs/<domain>/predictions-*.json`: what `momentbench score-predictions` wrote for each.
  Run ids carry a timestamp, so a rerun gets new ids with the same scores.
- `requirements-dense.txt`: the packages installed for the run.
- `SHA256SUMS`: hashes of every file above.

## Reproducing

Generated on 2026-11-19 from commit `cd5e9ca` with Python 3.12.12 on macOS 26.2 (arm64),
on CPU. MiniLM is `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, pinned in `scripts/dense_predictions.py`,
with sentence-transformers 6.3.0, transformers 5.23.0 and torch 2.15.0. On an Intel
Mac, pip resolves torch 2.2.2, which recent transformers releases refuse; use an arm64
or Linux Python.

```bash
pip install -e ".[dense]"
for d in film aerospace robotics; do
  q=data/queries/scored/${d}_v0.2_scored.json
  python scripts/bm25_predictions.py -q $q -o bm25_$d.json
  python scripts/bm25_predictions.py -q $q -o bm25-abstain10_$d.json --abstain-below 10
  python scripts/dense_predictions.py -q $q -o dense_$d.json
  python -c "import json,sys; q=json.load(open(sys.argv[1])); json.dump({'system': 'never-answer', 'predictions': [{'query_id': x['query_id'], 'moments': []} for x in q]}, open(sys.argv[2], 'w'))" $q never-answer_$d.json
  for s in bm25 bm25-abstain10 dense never-answer; do
    momentbench score-predictions -p ${s}_$d.json -q $q -o runs/$d
  done
done
```

The BM25 and never-answer files are deterministic. MiniLM's ranking can differ in the
last digits of a similarity score across hardware; the hit counts above are what this
run produced.
