# Next release candidates (v0.3)

Not part of v0.2. v0.2 is frozen at 27 assets so every published result was measured on the same corpus; these files join a release only when every model has been rerun on the larger corpus.

## Earth observation

- `earth_v0.3_manifest.json`: 6 public-domain NASA videos (hurricane from the ISS, wildfires, volcanic ash, the birth of a new island near Tonga, fire science, airglow), hash-pinned like the main manifest.
- `earth_v0.3_draft.json`: 28 published queries (6 more held out), tiers 1 to 4.

These queries were drafted and verified by **Gemini 3.1 Pro**, the reverse of the v0.2 sets, which Claude drafted. That lets the bias of the drafting model be measured: compare a model's score on the queries its own family drafted with its score on the other family's. Claude then cross-checked them against the same candidates: it agreed on 34 of 36 (94.4%) and the 2 it rejected were removed, the same agreement rate Gemini showed when checking Claude's v0.2 drafts.

Of 49 verified Gemini drafts, 13 were dropped because they hinge on on-screen titles or logos ("NASA HELIOPHYSICS" on black), which the drafting prompt rules out but Gemini wrote anyway; a 6-query spot check against frames had found 2 such queries and 0 wrong ones.
