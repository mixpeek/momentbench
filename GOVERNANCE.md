# Governance

MomentBench is created and maintained by Mixpeek, a company that sells video retrieval. Mixpeek's pipeline also appears on the leaderboard. This page states how that conflict of interest is handled.

## Rules that apply to every entrant, Mixpeek included

1. **Same harness.** Every leaderboard row comes from the public harness in this repo, and its run file records the commit, input hashes and a compatibility key. No entrant gets a private code path.
2. **Same data.** Every entrant is scored on the same query set and corpus release.
3. **Public run files.** Every row links to its run file, with per-query scores, usage and the harness version.
4. **Fixed configuration.** Budgets, trial counts and the IoU threshold are set per benchmark version and cannot be tuned per entrant.
5. **Failures count.** A refused, errored or unfinished trial scores zero and stays in the denominator.

## Rules that apply only to Mixpeek

1. **Labelled.** Mixpeek rows are marked as maintainer entries on the leaderboard.
2. **No early access.** Mixpeek's pipeline is not developed or tuned against held-out queries.
3. **Reproducible by others.** A Mixpeek row is published only with the configuration needed to rerun it.

## Changes to the benchmark

- Query sets, corpus releases and scoring rules are versioned. A change to any of them is a new version, and results are never compared across versions.
- Difficulty targets per tier were registered before baselines were run and are not adjusted after the fact.
- Proposed changes go through public pull requests.

## Disputes

Open a GitHub issue. Annotation disputes are resolved by a second annotator, and the outcome and reasoning are recorded on the issue. A corrected gold label ships in the next version.

## Held-out data

163 queries are held out of this repository. Mixpeek, as maintainer, holds their gold labels. A submission is scored on them with the same harness, budget and scoring as the public set, and held-out scores are reported next to public ones. Mixpeek's pipeline is held to rule 2 above. How to request a held-out evaluation is in [`SUBMISSION.md`](SUBMISSION.md).
