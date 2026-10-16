"""Text-only controls over the shipped transcripts. No model, no frames, no cost.

`baseline-transcript` (baselines.py) runs inside the harness with the corpus's keyword
search. That search is weak, so failing it does not show a query needs the picture
(audit 3, finding 4). `bm25_control` is a stronger fixed lexical method over the same
cues: Okapi BM25 with k1=1.2, b=0.75, a fixed stopword list, and the top five cue
spans submitted in rank order. Gold is used only to score the result.

This is a diagnostic over the transcript files, outside the agentic loop: an agent
under the default budget cannot read the whole corpus, but a text index can.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from momentbench.agentic.environment import load_transcript
from momentbench.types import Moment, Query

K1, B, TOP_K = 1.2, 0.75, 5
STOPWORDS = frozenset("""a an and are as at be been but by for from has have he her his
in into is it its of on or she that the their them they this to was were which while
who with""".split())


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in STOPWORDS]


def load_transcripts(asset_ids: list[str], transcripts_dir: Path) -> dict[str, list[dict]]:
    out = {}
    for aid in asset_ids:
        for ext in (".transcript.json", ".vtt"):
            if (f := transcripts_dir / f"{aid}{ext}").exists():
                out[aid] = load_transcript(f)
                break
    return out


class BM25Index:
    def __init__(self, transcripts: dict[str, list[dict[str, Any]]]):
        self.cues = [(aid, s) for aid, segs in sorted(transcripts.items()) for s in segs
                     if s["t_out"] > s["t_in"]]
        self.docs = [Counter(tokens(s["text"])) for _, s in self.cues]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, text: str, top_k: int = TOP_K) -> list[Moment]:
        terms = set(tokens(text))
        scored = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            for t in terms & doc.keys():
                tf = doc[t]
                score += self.idf[t] * tf * (K1 + 1) / (
                    tf + K1 * (1 - B + B * self.lengths[i] / self.avg))
            if score > 0:
                scored.append((-score, i))
        scored.sort()
        return [Moment(asset_id=self.cues[i][0], t_in=self.cues[i][1]["t_in"],
                       t_out=self.cues[i][1]["t_out"]) for _, i in scored[:top_k]]


def bm25_control(queries: list[Query], transcripts: dict[str, list[dict[str, Any]]]
                 ) -> dict[str, bool]:
    """query_id -> hit@1 under the benchmark scorer, for the BM25 text-only control."""
    from momentbench.agentic.run import score_trial

    index = BM25Index(transcripts)
    out = {}
    for q in queries:
        moments = index.search(q.text)
        out[q.query_id] = score_trial(q, moments, "submitted")[0]
    return out
