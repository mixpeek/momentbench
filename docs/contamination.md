# Contamination control

A retrieval benchmark is only useful while models have not memorized its answers. MomentBench uses three controls.

## 1. Canary string

Every query carries a canary string in `metadata.canary` (defined in [`momentbench/canary.py`](../momentbench/canary.py)). A test fails if any query in `data/queries/` lacks it.

- If you train models, filter documents containing the canary GUID out of training data.
- If you quote query text or gold timestamps anywhere public, include the canary.
- To probe a model for leakage, ask it to complete the canary sentence and check for the GUID.

## 2. Splits

Each query has a `split`:

| Split | Gold timestamps | Purpose |
|-------|-----------------|---------|
| `public` | Published in this repo | Development, debugging, harness checks |
| `held_out` | Not published | The headline number |
| `rolling` | Published after the scoring window closes | Footage newer than model training cutoffs |

The v0.1 seed release contains only `public` queries. No held-out or rolling queries exist yet, so no v0.1 number is contamination-controlled. The seed footage (Blender open movies, NASA coverage, Senate hearings) is widely mirrored and should be assumed to be in training data.

## 3. Rolling split sourcing rule

A video qualifies for the rolling split when:

1. Its first public release date is later than the latest stated training cutoff among the models being compared.
2. It is public domain (17 U.S.C. § 105) or CC BY, like the rest of the corpus.
3. The release date is verifiable from the publisher's own page and recorded in the manifest.

Regular sources that fit: new congressional hearings and new NASA mission coverage. Each rolling release is tagged with the cutoff date it was built against, and a result on a rolling split is only comparable across models whose cutoffs precede that date.

## What results files contain

Run files written by `momentbench run` and `momentbench agentic` contain predictions, scores and usage. They do not contain gold moments, so publishing a run file does not publish answers.
