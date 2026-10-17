"""One way to build a Gemini client, for the agentic runner and the annotation tools.

An API key wins when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set. Otherwise, with
`MOMENTBENCH_VERTEX_PROJECT` set, the client goes through Vertex AI in that project
(`MOMENTBENCH_VERTEX_LOCATION`, default `global`). Credentials come from the access
token of `MOMENTBENCH_GCLOUD_ACCOUNT` when that is set (a service account that gcloud
already holds, so no key file is written), else from application default
credentials.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any


def make_client() -> Any:
    from google import genai

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return genai.Client(api_key=key)
    project = os.environ.get("MOMENTBENCH_VERTEX_PROJECT")
    if not project:
        raise RuntimeError("set GEMINI_API_KEY, or MOMENTBENCH_VERTEX_PROJECT for Vertex AI")
    location = os.environ.get("MOMENTBENCH_VERTEX_LOCATION", "global")
    credentials = None
    account = os.environ.get("MOMENTBENCH_GCLOUD_ACCOUNT")
    if account:
        from google.oauth2.credentials import Credentials

        token = subprocess.run(["gcloud", "auth", "print-access-token", "--account", account],
                               capture_output=True, text=True, check=True,
                               timeout=60).stdout.strip()
        credentials = Credentials(token)
    return genai.Client(vertexai=True, project=project, location=location,
                        credentials=credentials)
