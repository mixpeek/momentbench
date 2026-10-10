"""Build a static page for human review of queries, and apply the reviewer's decisions.

    # 1. Build the page (frames are written next to it; open index.html in a browser)
    python scripts/review_queries.py build data/queries/film_v0.2.json \\
        --corpus-dir CORPUS/videos --out review/film

    # 2. The reviewer marks each query accept, fix or reject, adds notes, and clicks
    #    "Download decisions" to save decisions.json.

    # 3. Apply: accepted queries get metadata.human_review, rejected ones are removed,
    #    queries marked fix are kept but flagged with the note.
    python scripts/review_queries.py apply data/queries/film_v0.2.json decisions.json \\
        --reviewer "name or handle"

Each query shows five frames: one second before the gold span, its start, middle and
end, and one second after, so a reviewer can judge both the description and the
boundaries. A no-answer query shows its altered source shot and the change that was
made, and the reviewer confirms that nothing matches.
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import subprocess
from pathlib import Path

WIDTH = 256


def frame(video: Path, t: float, out: Path) -> str:
    if not out.exists():
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, t):.2f}", "-i", str(video),
             "-frames:v", "1", "-vf", f"scale={WIDTH}:-2", str(out)],
            check=False, capture_output=True,
        )
    return out.name if out.exists() else ""


def build(args: argparse.Namespace) -> None:
    queries = json.loads(Path(args.queries).read_text())
    corpus, out = Path(args.corpus_dir), Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    rows = []
    for q in queries:
        gold = q["gold_moments"][0] if q["gold_moments"] else None
        span = gold or q["metadata"].get("altered_from")
        if not span:
            continue
        video = next(corpus.glob(f"{span['asset_id']}.*"))
        a, b = span["t_in"], span["t_out"]
        times = [("before", a - 1.0), ("start", a + 0.2), ("middle", (a + b) / 2),
                 ("end", b - 0.2), ("after", b + 1.0)]
        cells = []
        for label, t in times:
            name = frame(video, t, out / "frames" / f"{q['query_id']}_{label}.jpg")
            cls = "ctx" if label in ("before", "after") else "in"
            img = f'<img src="frames/{name}">' if name else "<i>no frame</i>"
            cells.append(f'<figure class="{cls}">{img}<figcaption>{label} {t:.1f}s'
                         "</figcaption></figure>")
        if gold:
            what = (f"Gold: <b>{html.escape(span['asset_id'])}</b> {a:.2f} to {b:.2f} s. "
                    "Does the text match the frames inside the span, and only them?")
        else:
            change = html.escape(str(span.get("change", "")))
            what = (f"No-answer. Source shot {html.escape(span['asset_id'])} {a:.2f} to "
                    f"{b:.2f} s was altered: <i>{change}</i>. Accept only if nothing in the "
                    "corpus matches the text.")
        qid = html.escape(q["query_id"])
        rows.append(f"""
<section data-id="{qid}">
  <h3>{qid} <span class="tier">tier {q['tier']}</span></h3>
  <p class="text">{html.escape(q['text'])}</p>
  <p class="what">{what}</p>
  <div class="frames">{''.join(cells)}</div>
  <label><input type="radio" name="{qid}" value="accept"> accept</label>
  <label><input type="radio" name="{qid}" value="fix"> fix</label>
  <label><input type="radio" name="{qid}" value="reject"> reject</label>
  <input class="note" placeholder="note (required for fix or reject)">
</section>""")
    page = f"""<!doctype html><meta charset="utf-8"><title>MomentBench review</title>
<style>
body{{font:15px system-ui;margin:24px;max-width:1400px}}
section{{border-top:1px solid #ccc;padding:12px 0}}
.text{{font-size:17px}} .what{{color:#444}} .tier{{color:#888;font-weight:normal}}
.frames{{display:flex;gap:6px}} figure{{margin:0}} figcaption{{font-size:12px;color:#555}}
figure.ctx img{{opacity:.55}} .note{{width:420px;margin-left:12px}}
#bar{{position:sticky;top:0;background:#fff;padding:8px 0}}
</style>
<div id="bar"><b>{len(rows)} queries</b> from {html.escape(Path(args.queries).name)}.
Context frames (before, after) are faded. <span id="count"></span>
<button onclick="save()">Download decisions</button></div>
{''.join(rows)}
<script>
const key = "momentbench-review-{html.escape(Path(args.queries).name)}";
const state = JSON.parse(localStorage.getItem(key) || "{{}}");
for (const s of document.querySelectorAll("section")) {{
  const id = s.dataset.id, d = state[id] || {{}};
  if (d.decision) s.querySelector(`input[value="${{d.decision}}"]`).checked = true;
  s.querySelector(".note").value = d.note || "";
  s.addEventListener("change", () => {{
    const c = s.querySelector("input[type=radio]:checked");
    state[id] = {{decision: c ? c.value : null, note: s.querySelector(".note").value}};
    localStorage.setItem(key, JSON.stringify(state)); count();
  }});
}}
function count() {{
  const n = Object.values(state).filter(d => d.decision).length;
  document.getElementById("count").textContent = `${{n}} reviewed.`;
}}
function save() {{
  const blob = new Blob([JSON.stringify(state, null, 1)], {{type: "application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "decisions.json"; a.click();
}}
count();
</script>"""
    (out / "index.html").write_text(page)
    print(f"wrote {out / 'index.html'} with {len(rows)} queries")


def apply(args: argparse.Namespace) -> None:
    path = Path(args.queries)
    queries = json.loads(path.read_text())
    decisions = json.loads(Path(args.decisions).read_text())
    today = datetime.date.today().isoformat()
    kept, counts = [], {"accept": 0, "fix": 0, "reject": 0, "unreviewed": 0}
    for q in queries:
        d = decisions.get(q["query_id"]) or {}
        decision = d.get("decision")
        if decision not in ("accept", "fix", "reject"):
            counts["unreviewed"] += 1
            kept.append(q)
            continue
        if decision in ("fix", "reject") and not (d.get("note") or "").strip():
            raise SystemExit(f"{q['query_id']}: a {decision} decision needs a note")
        counts[decision] += 1
        if decision == "reject":
            continue
        q["metadata"]["human_review"] = {"reviewer": args.reviewer, "date": today,
                                         "decision": decision, "note": d.get("note", "")}
        if decision == "accept":
            q["metadata"]["source"] = q["metadata"]["source"].replace(
                "pending human review", f"human-reviewed by {args.reviewer} on {today}")
        kept.append(q)
    path.write_text(json.dumps(kept, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(counts))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("queries")
    b.add_argument("--corpus-dir", required=True)
    b.add_argument("--out", required=True)
    a = sub.add_parser("apply")
    a.add_argument("queries")
    a.add_argument("decisions")
    a.add_argument("--reviewer", required=True)
    args = p.parse_args()
    (build if args.cmd == "build" else apply)(args)


if __name__ == "__main__":
    main()
