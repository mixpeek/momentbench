# Transcripts

Machine transcripts of the spoken dialogue in the film corpus, one WebVTT file per asset, named `<asset_id>.vtt`. The agentic track's `search_transcript` and `read_transcript` tools read them, and so do the cross-modal (tier 2) queries.

- Produced with OpenAI Whisper `small`, English, on 2026-10-10, from the hash-verified files listed in `data/corpus_manifests/sample_manifest.json`.
- Not corrected by hand. Expect misheard words and cue boundaries that are a few seconds off. Agent 327's cues are especially coarse.
- Big Buck Bunny and Spring have no dialogue and no transcript.
- The dialogue belongs to the films and carries their licences: Blender Foundation, Creative Commons Attribution. See the manifest for each film's licence.

Every entrant gets the same transcripts, so differences between systems do not come from speech recognition.
