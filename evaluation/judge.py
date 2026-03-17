"""
LLM-as-judge: score responses for realism, persona consistency, cultural plausibility.
"""

import json
import re
from typing import Any, Dict, List, Optional

from config.settings import get_settings
from llm.client import get_llm_client
from llm.prompts import build_judge_prompt, JUDGE_SYSTEM
from population.personas import Persona


async def judge_response(
    persona: Persona,
    question: str,
    response: str,
    model: Optional[str] = None,
) -> Dict[str, int]:
    """
    Returns {realism: 1-5, persona_consistency: 1-5, cultural_plausibility: 1-5}.
    """
    settings = get_settings()
    model = model or settings.openai_judge_model
    client = get_llm_client()
    summary = persona.to_compressed_summary()
    prompt = build_judge_prompt(summary, question, response)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    raw = await client.chat(messages, model=model, temperature=0.2, max_tokens=128)
    # Parse JSON from response
    try:
        match = re.search(r"\{[^{}]*\}", raw)
        if match:
            data = json.loads(match.group())
            return {
                "realism": int(data.get("realism", 3)),
                "persona_consistency": int(data.get("persona_consistency", 3)),
                "cultural_plausibility": int(data.get("cultural_plausibility", 3)),
            }
    except (json.JSONDecodeError, ValueError):
        pass
    return {"realism": 3, "persona_consistency": 3, "cultural_plausibility": 3}


async def judge_responses_batch(
    personas: List[Persona],
    questions: List[str],
    responses: List[str],
    sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Judge a batch (personas[i], questions[i], responses[i]).
    If sample_size set, only judge a random sample to save cost.
    """
    import random
    n = len(personas)
    if n != len(questions) or n != len(responses):
        return {"error": "Length mismatch", "scores": []}
    indices = list(range(n))
    if sample_size and n > sample_size:
        indices = random.sample(indices, sample_size)
    scores = []
    for i in indices:
        s = await judge_response(personas[i], questions[i], responses[i])
        s["agent_id"] = personas[i].agent_id
        scores.append(s)
    avg = {
        "realism": sum(s["realism"] for s in scores) / len(scores) if scores else 0,
        "persona_consistency": sum(s["persona_consistency"] for s in scores) / len(scores) if scores else 0,
        "cultural_plausibility": sum(s["cultural_plausibility"] for s in scores) / len(scores) if scores else 0,
    }
    return {"scores": scores, "average": avg, "n_judged": len(scores)}
