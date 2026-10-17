"""Split the published queries into scored and provisional sets from the label log.

    python scripts/label_sets.py

Reads `data/labels/adjudication.json` (written by scripts/adjudicate.py) and:

- records each query's decision in `metadata.adjudication` of the query files;
- writes `data/queries/scored/<domain>_v0.2_scored.json`, the queries both non-Claude
  judges accepted. A run on a scored file has its own compatibility key, because the
  key covers the selected queries;
- writes `docs/labels.md`, the public label log: method, counts and every provisional
  query with both judges' reasons.

A query whose annotation changed since it was judged is provisional until it is
judged again.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from momentbench.annotation import annotation_sha  # noqa: E402

DOMAINS = ("film", "aerospace", "robotics")
LOG = ROOT / "data/labels/adjudication.json"
TIER = {1: "single shot", 2: "dialogue and picture", 3: "no answer", 4: "multi-shot"}


def control_text(summary: dict) -> str:
    rows = summary.get("controls") or []
    if not rows:
        return ""
    used = rows[0]
    text = (f" **This check is a coarse screen, and passing it is weaker than proof of "
            f"absence.** In a control, the same scan was given true descriptions from the "
            f"videos it watched and found {used['found']} of {used['true_descriptions']} "
            f"({100 * used['recall']:.0f}%) near their gold spans ({used['config']}).")
    for r in rows[1:]:
        text += (f" A finer setting ({r['config']}) found {r['found']} of "
                 f"{r['true_descriptions']} ({100 * r['recall']:.0f}%) on {r['videos']} "
                 f"videos; {r.get('note', '')}.")
    return text + (" Some missed descriptions are themselves provisional labels, so part of "
                   "the gap may be label error. Human disputes remain the stronger check.")


def main() -> None:
    log = json.loads(LOG.read_text())
    entries, judges = log["queries"], log["summary"]["judges"]
    rows, counts = [], Counter()
    out_dir = ROOT / "data/queries/scored"
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in DOMAINS:
        path = ROOT / f"data/queries/{d}_v0.2.json"
        queries = json.loads(path.read_text())
        scored = []
        for q in queries:
            e = entries.get(q["query_id"])
            current = e is not None and e.get("annotation_sha") == annotation_sha(q)
            decision = e["decision"] if current else "provisional"
            absence = (e or {}).get("absence")
            if not q["gold_moments"] and decision == "scored" and not (
                    absence and absence.get("passed")
                    and absence.get("annotation_sha") == annotation_sha(q)):
                # A no-answer query also has to match nothing anywhere in the corpus.
                decision = "provisional"
            q["metadata"]["adjudication"] = {
                "decision": decision, "judges": judges,
                "config_sha": e["config_sha"] if e else None,
                "annotation_sha": e["annotation_sha"] if e else None}
            counts[(d, decision)] += 1
            counts[(d, int(q["tier"]), decision)] += 1
            if decision == "scored":
                scored.append(q)
            else:
                rows.append((d, q, e if current else None))
        path.write_text(json.dumps(queries, indent=2, ensure_ascii=False) + "\n")
        (out_dir / f"{d}_v0.2_scored.json").write_text(
            json.dumps(scored, indent=2, ensure_ascii=False) + "\n")

    def n(d, dec, tier=None):
        return counts[(d, tier, dec)] if tier else counts[(d, dec)]

    table = ["| Domain | Published | Scored | Provisional | No-answer scored |",
             "|---|---:|---:|---:|---:|"]
    for d in DOMAINS:
        table.append(f"| {d} | {n(d, 'scored') + n(d, 'provisional')} | {n(d, 'scored')} | "
                     f"{n(d, 'provisional')} | {n(d, 'scored', 3)} of "
                     f"{n(d, 'scored', 3) + n(d, 'provisional', 3)} |")
    tiers = ["| Tier | Published | Scored | Share scored |", "|---|---:|---:|---:|"]
    for t in sorted(TIER):
        s_ = sum(n(d, "scored", t) for d in DOMAINS)
        p_ = sum(n(d, "provisional", t) for d in DOMAINS)
        if s_ + p_:
            tiers.append(f"| {TIER[t]} | {s_ + p_} | {s_} | {100 * s_ / (s_ + p_):.0f}% |")
    table += ["", "By tier, across domains:", ""] + tiers
    total_s = sum(n(d, "scored") for d in DOMAINS)
    total_p = sum(n(d, "provisional") for d in DOMAINS)

    def cell(r):
        if not r:
            return "not judged on this annotation"
        bits = [r.get("verdict", "")]
        if "a_true" in r:
            bits.append(f"original true {r['a_true']}, altered true {r['b_true']}, "
                        f"detail visible {r['detail_visible']}")
        return "; ".join(bits) + (": " + r["reason"].replace("|", "/") if r.get("reason") else "")

    prov = ["| Query | Tier | Gemini | GPT |", "|---|---|---|---|"]
    for d, q, e in rows:
        g, o = (e or {}).get("gemini"), (e or {}).get("gpt")
        ab = (e or {}).get("absence") or {}
        note = ""
        if not q["gold_moments"] and ab.get("confirmed_matches"):
            m = ab["confirmed_matches"][0]
            note = (f" Both judges found a match elsewhere: {m['asset_id']} "
                    f"{m['t_start']:.0f}–{m['t_end']:.0f} s.")
        prov.append(f"| {q['query_id']} | {TIER.get(int(q['tier']), q['tier'])} | "
                    f"{cell(g)}{note} | {cell(o)} |")
    summary = log["summary"].get("absence")
    absence_text = (
        f"\n\n**Corpus-wide absence check** (`scripts/absence_check.py`). A no-answer query "
        f"must match nothing anywhere, which drafting only checked against three look-alike "
        f"shots. {judges[0]} watched all {summary['assets']} videos with every no-answer "
        f"description and listed any moment that fits; {judges[1]} checked each hit away "
        f"from the source shot on frames. {summary['hits_off_source']} hits, "
        f"{summary['confirmed']} confirmed by both; {summary['passed']} of "
        f"{summary['queries']} no-answer queries passed. A no-answer query is scored only "
        f"if it also passed this check.{control_text(summary)}") if summary else ""
    (ROOT / "docs/labels.md").write_text(f"""# Label log

*Generated by `scripts/label_sets.py` from `data/labels/adjudication.json`. Do not edit.*

Claude drafted, verified and reviewed MomentBench's labels, so Claude does not judge
them here. Two other model families judge every published query against its current
annotation (`scripts/adjudicate.py`):

- **{judges[0]}** watches the clip, sampled at 4 frames a second with its audio: the
  gold span plus 2 s on each side, or for a no-answer query the source shot it was
  altered from. Sampling means it does not see every frame.
- **{judges[1]}** reads a five-frame sheet of the same span (before, start, middle,
  end, after), with the speech inside it for dialogue queries.

For a no-answer query each judge answers three questions: is the original description
true of the source shot, is the altered one, and can the changed detail be seen or
heard? It passes only when the original is true, the altered text is false and the
detail is visible. A query is **scored** when both judges pass it and **provisional**
otherwise. Scored files are in `data/queries/scored/`, and a run on them carries its
own compatibility key. These are model judgements, not human review, and model
disagreement is not an annotation-error estimate: a query is provisional when either
judge objects, and some objections are judge errors (the frame judge sees five stills,
so it can miss what happens between them on multi-shot and fast-moving queries).
Selecting the scored set also changes coverage: multi-shot queries are excluded more
often than single-shot ones, as the tier table below shows. The scored set is frozen
for each release by the hash of its files. Human disputes are welcome
(open an issue).

{chr(10).join(table)}{absence_text}

In total {total_s} scored and {total_p} provisional, of {total_s + total_p}.

## Provisional queries and why

{chr(10).join(prov)}
""")
    print(f"scored {total_s}, provisional {total_p}")


if __name__ == "__main__":
    main()
