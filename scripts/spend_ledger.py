"""Show or check the shared spend ledger (momentbench/spend.py).

    MOMENTBENCH_SPEND_LEDGER=~/.momentbench/spend.jsonl python scripts/spend_ledger.py
    MOMENTBENCH_SPEND_LEDGER=... python scripts/spend_ledger.py check 12.50
    MOMENTBENCH_SPEND_LEDGER=... python scripts/spend_ledger.py estimate gemini-3.5-flash 1e6 2e5

`status` (the default) prints the total, what is left under the cap and the spend by
item. `check USD` exits 1 when spending USD more would cross the cap, for use in
shell gates. `estimate MODEL IN OUT` prints what a call of that size would cost.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from momentbench import spend  # noqa: E402


def main(argv: list[str]) -> int:
    if not spend.active():
        print("MOMENTBENCH_SPEND_LEDGER is not set; no ledger is in force", file=sys.stderr)
        return 2
    cmd = argv[0] if argv else "status"
    if cmd == "status":
        print(json.dumps(spend.summary(), indent=2))
        return 0
    if cmd == "check":
        try:
            spend.reserve(float(argv[1]), "check")
        except spend.SpendCapError as e:
            print(e)
            return 1
        print(f"ok: ${float(argv[1]):.2f} fits; ${spend.spent_usd():.2f} of "
              f"${spend.cap_usd():.2f} spent")
        return 0
    if cmd == "estimate":
        print(f"${spend.token_cost_usd(argv[1], int(float(argv[2])), int(float(argv[3]))):.4f}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
