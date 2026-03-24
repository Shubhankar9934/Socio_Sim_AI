"""Minimal deterministic baseline snapshot for CI drift checks."""

from __future__ import annotations

import json
from pathlib import Path

from config.option_space import canonicalize_distribution
from core.rng import make_rng_pack


def build_snapshot() -> dict:
    pack = make_rng_pack("baseline:snapshot", base_seed=42)
    sample = [round(float(pack.np_rng.random()), 8) for _ in range(5)]
    canonical = canonicalize_distribution(
        "food_delivery_frequency",
        {"often": 0.2, "daily": 0.3, "rarely": 0.5},
    )
    return {
        "rng_sequence": sample,
        "canonical_distribution": canonical,
    }


def main() -> None:
    out = Path("tests/baseline_snapshot.json")
    out.write_text(json.dumps(build_snapshot(), indent=2), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()

