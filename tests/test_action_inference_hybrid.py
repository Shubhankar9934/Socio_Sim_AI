import asyncio

from discovery.action_inference import ActionModelBuilder


class _FakeClient:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0

    async def chat(self, *_args, **_kwargs):
        self.calls += 1
        return self.payload


def _llm_payload() -> str:
    return """
{
  "interaction_mode": "survey",
  "question_type": "likelihood",
  "scale_type": "likelihood",
  "topic": "general",
  "domain": "technology",
  "location_related": false,
  "question_model_key_candidate": "tech_adoption_likelihood",
  "persona_anchor_allowed": false,
  "action_type_candidate": "adopt",
  "target_candidate": "product",
  "intensity_scale_candidate": "binary",
  "normalization_candidates": [],
  "confidence": 0.94,
  "reason": "hybrid_test"
}
""".strip()


def test_action_inference_uses_hybrid_understanding(monkeypatch, tmp_path):
    cache_path = tmp_path / "action_template_cache.json"
    monkeypatch.setattr("discovery.action_inference._CACHE_PATH", cache_path)
    monkeypatch.setattr("discovery.action_inference._ACTION_CACHE", {})
    fake = _FakeClient(_llm_payload())
    monkeypatch.setattr("agents.intent_router.get_llm_client", lambda: fake)

    builder = ActionModelBuilder()
    result = asyncio.run(builder.infer_action_type_result("Would you try a new fintech wallet next month?"))

    assert fake.calls == 1
    assert result.template.action_type == "adopt"
    assert result.template.target == "product"
    assert result.template.intensity_scale == "binary"
    assert result.source in {"hybrid", "rule_llm_agree"}
