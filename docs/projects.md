# Projects on MomentBench

MomentBench is built to be extended: a class project, a thesis chapter or a side
experiment can produce a validated result here. To list your project below, open a pull
request that adds one line to the list: title, what you found, names, institution, and a
link to your code or paper. A negative result counts.

## Projects

None listed yet.

## Start without a GPU or an API key

```bash
git clone https://github.com/mixpeek/momentbench.git && cd momentbench
pip install -e .
python scripts/bm25_predictions.py -q data/queries/scored/film_v0.2_scored.json -o bm25_film.json
momentbench score-predictions -p bm25_film.json -q data/queries/scored/film_v0.2_scored.json
```

That is the transcript baseline on the 130 film queries. It should print Recall@1 of
7.7%, net score -0.85 and no-answer accuracy 0%. You now have a scorer, a baseline and a
file format ([`prediction-files.md`](prediction-files.md)). The projects below start
there. Frame-based work needs the corpus: see [Getting the Corpus](../README.md#getting-the-corpus).

## Project ideas

The baseline numbers are from [`report.md`](report.md). Rows compare only within a group,
so compare prediction-file results with prediction-file results.

1. **Beat the transcript baseline.** BM25 over transcripts reaches positive-query
   Recall@1 of 9.5% on film, 17.3% on aerospace and 12.5% on robotics. Add dense
   retrieval, reranking or a hybrid, and report which queries move. Needs no video.
2. **Learn when to say nothing.** 49 of the 333 scored queries have no match. Abstaining
   on everything nets +0.19 on film, while BM25 nets -0.85 because it always answers. Add
   a confidence threshold, calibrate it on one domain and test it on another.
3. **Find the right seconds.** A hit needs temporal IoU of at least 0.5. Take the videos
   a system already finds and tighten its spans. Report how much Recall@1 comes from
   choosing the video and how much from the boundaries.
4. **Move a moment-retrieval model to a collection.** Models such as Moment-DETR are
   usually evaluated with the right video given. Run one on all 27 videos, convert its
   output with `scripts/convert_predictions.py`, and measure the drop. Needs a GPU.
5. **Test footage newer than training cutoffs.** Aerospace has 38 `rolling` queries, 30 of them
   scored, from footage published after current model training cutoffs. Compare them with the public
   queries and test whether the gap survives the small sample.
6. **Audit the labels.** Every label is a model judgement ([`labels.md`](labels.md)), so
   no person has reviewed them. Review a random sample by hand, report the error rate by
   tier, and open an issue for each label you dispute.
7. **Add a domain.** Propose a set of rights-cleared videos and queries that follow
   [`CONTRIBUTING.md`](../CONTRIBUTING.md#adding-queries). It would be scored in a future
   release; v0.2.5 stays frozen.

## Using MomentBench in a course

No course has used MomentBench yet, so tell us what breaks. It suits a course project
for these reasons:

- A team can have a score on the first day, on a laptop, at no cost.
- Each idea above fits a term and has a baseline to beat.
- `momentbench bundle` writes the JSON, LaTeX table and citation for the final report.
- A valid submission is checked automatically and published on
  [momentbench.com](https://momentbench.com), so students finish with a public result.

Instructors can open an issue with the term dates and team count, and we will suggest
projects and answer student questions there.

## Report what you find

Results go through [`bring-your-system.md`](bring-your-system.md). Mixpeek maintains
MomentBench and has an entry on the leaderboard under the same rules
([`GOVERNANCE.md`](../GOVERNANCE.md)), so results that make any entry look bad are welcome.
