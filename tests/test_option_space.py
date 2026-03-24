import asyncio

from config.option_space import (
    canonicalize_distribution,
    canonicalize_option,
    canonicalize_options,
    validate_option_compatibility,
    validate_option_compatibility_hybrid,
)


def test_food_delivery_alias_mapping():
    assert canonicalize_option("food_delivery_frequency", "very often") == "multiple per day"
    assert canonicalize_option("food_delivery_frequency", "often") == "daily"


def test_distribution_merge_after_mapping():
    dist = {"daily": 0.3, "often": 0.2, "rarely": 0.5}
    out = canonicalize_distribution("food_delivery_frequency", dist)
    assert "daily" in out
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_options_deduplicate_after_canonicalization():
    opts = ["daily", "often", "rarely"]
    out = canonicalize_options("food_delivery_frequency", opts)
    assert out == ["daily", "rarely"]


def test_option_space_compatibility_accepts_semantic_aliases():
    compatible, normalized, warnings = validate_option_compatibility(
        "generic_frequency",
        ["Never", "Rarely", "Sometimes", "Often", "Very Often"],
        ["never", "rarely", "sometimes", "often", "very often"],
    )
    assert compatible is True
    assert normalized == ["never", "rarely", "sometimes", "often", "very often"]
    assert warnings == []


def test_option_space_compatibility_rejects_incompatible_scale():
    compatible, normalized, warnings = validate_option_compatibility(
        "generic_frequency",
        ["very unlikely", "unlikely", "neutral", "likely", "very likely"],
        ["never", "rarely", "sometimes", "often", "very often"],
    )
    assert compatible is False
    assert normalized == ["very unlikely", "unlikely", "neutral", "likely", "very likely"]
    assert any(w.startswith("missing_options:") for w in warnings)
    assert any(w.startswith("unexpected_options:") for w in warnings)


class _FakeClient:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0

    async def chat(self, *_args, **_kwargs):
        self.calls += 1
        return self.payload


def test_hybrid_option_mapping_can_learn_aliases(monkeypatch, tmp_path):
    cache_path = tmp_path / "option_space_cache.json"
    monkeypatch.setattr("config.option_space._HYBRID_ALIAS_CACHE_PATH", cache_path)
    monkeypatch.setattr("config.option_space._HYBRID_ALIAS_CACHE", {})
    fake = _FakeClient(
        """
{
  "not at all": "never",
  "once in a while": "rarely",
  "somewhat often": "sometimes",
  "pretty often": "often",
  "all the time": "very often"
}
""".strip()
    )
    monkeypatch.setattr("config.option_space.get_llm_client", lambda: fake)

    compatible, normalized, warnings = asyncio.run(
        validate_option_compatibility_hybrid(
            "generic_frequency",
            ["not at all", "once in a while", "somewhat often", "pretty often", "all the time"],
            ["never", "rarely", "sometimes", "often", "very often"],
        )
    )

    assert fake.calls == 1
    assert compatible is True
    assert normalized == ["never", "rarely", "sometimes", "often", "very often"]
    assert "hybrid_alias_mapping_applied" in warnings
    assert canonicalize_option("generic_frequency", "once in a while") == "rarely"

