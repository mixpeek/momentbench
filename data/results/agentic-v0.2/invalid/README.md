# Invalid runs

Kept as evidence, never cited as results.

- `agentic-gpt-5.5-a958c130.json`: the OpenAI account went inactive at trial 121 of 185, and the remaining 63 trials record billing errors ("Your account is not active"). The harness now stops a run on such errors and saves only a `.partial.json` file (commits 2ae012a, bf2aa18).
- `agentic-claude-opus-5-5-1b3d03ba.json` (film, 26 errored trials) and `agentic-claude-sonnet-5-5-680da6ae.json` (aerospace, 198 errored trials): reruns on the 27-asset corpus that hit the Claude Code subscription session limit mid-run. A session limit now stops a run like a billing error.
