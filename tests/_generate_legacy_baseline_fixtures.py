"""One-off generator for the LEGACY variation-proposal prompt fixtures.

Run BEFORE moving self-report responsibilities into the shared layer so the
captured fixtures represent the current (flag-off) prompts. The snapshot test
then proves the refactored code still renders these byte-for-byte with the
self_report flag off.

    python -m tests._generate_legacy_baseline_fixtures
"""

from __future__ import annotations

from pathlib import Path

from tests.legacy_prompt_fixture_inputs import build_legacy_prompts, make_client

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "prompts_baseline_legacy"


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    client = make_client()
    for name, prompt in build_legacy_prompts(client).items():
        path = FIXTURE_DIR / f"{name}.txt"
        path.write_text(prompt, encoding="utf-8", newline="")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
