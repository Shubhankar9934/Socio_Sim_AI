"""
Regenerate the 500-agent survey dataset with all realism upgrades:
  - Population segmentation (multimodal behavioral clusters)
  - Structured cross-question memory
  - Persistent narrative style profiles
  - Pre-survey social warmup diffusion

Run: python regenerate_survey.py [N]
Default N=500. Output: survey_response.json
"""

import asyncio
import json
import sys
import time
import uuid


async def regenerate(n: int = 500):
    from agents.state import AgentState
    from population.synthesis import generate_population
    from simulation.orchestrator import run_survey
    from social.network import build_social_network

    print(f"Generating {n} personas (Bayesian synthesis + segments + styles)...")
    personas = generate_population(n=n, method="bayesian", seed=42)

    # Report segment distribution
    from collections import Counter
    seg_counts = Counter(p.meta.population_segment for p in personas)
    print(f"\nPopulation segments:")
    for seg, count in seg_counts.most_common():
        pct = count / n * 100
        bar = "#" * (count // 5)
        print(f"  {seg:<25} {count:>3} ({pct:.1f}%) {bar}")

    # Report narrative style distribution
    verb_counts = Counter(p.personal_anchors.narrative_style.verbosity for p in personas)
    print(f"\nNarrative verbosity distribution:")
    for v, count in verb_counts.most_common():
        print(f"  {v:<10} {count:>3} ({count/n*100:.1f}%)")

    # Build agents with persistent state (needed for memory + social warmup)
    agents = []
    for p in personas:
        state = AgentState.from_persona(p)
        agents.append({
            "persona": p,
            "state": state,
            "social_trait_fraction": 0.0,
            "location_quality": 0.5,
        })

    # Build social graph and store it globally for the orchestrator's warmup
    print(f"\nBuilding social network ({n} agents, Barabasi-Albert)...")
    social_graph = build_social_network(personas, seed=42)
    import api.state as app_state
    app_state.social_graph = social_graph

    # Compute social_trait_fraction from the social graph
    from social.influence import fraction_friends_with_trait
    from world.districts import location_quality_for_satisfaction

    trait_by_agent = {}
    for a in agents:
        p = a["persona"]
        if p.lifestyle.primary_service_preference >= 0.5:
            trait_by_agent[p.agent_id] = True
        else:
            trait_by_agent[p.agent_id] = False

    for a in agents:
        p = a["persona"]
        frac = fraction_friends_with_trait(social_graph, p.agent_id, trait_by_agent)
        a["social_trait_fraction"] = frac
        a["location_quality"] = location_quality_for_satisfaction(p.location)
        a["state"].set_social_trait_fraction(frac)

    question = "How often do you order food delivery?"
    question_id = str(uuid.uuid4())

    print(f"\nRunning survey: '{question}'")
    print(f"Agents: {n}, with social warmup + memory + style profiles...")
    t0 = time.time()

    # Pass think_fn=None so the orchestrator's default_think handles:
    #   - persistent state cache (cross-question memory)
    #   - world environment injection (neighbor latent means)
    #   - LLM reasoning with style profiles
    responses = await run_survey(
        agents,
        question=question,
        question_id=question_id,
        think_fn=None,
        use_archetypes=False,
        max_concurrent=20,
    )

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

    survey_id = str(uuid.uuid4())
    output = {
        "survey_id": survey_id,
        "question": question,
        "responses": responses,
        "n_total": len(responses),
    }

    with open("survey_response.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    errors = sum(1 for r in responses if r.get("error"))
    vague = sum(1 for r in responses if r.get("answer", "") and len(r["answer"]) < 30)
    print(f"\nResults: {len(responses)} responses, {errors} errors, {vague} short/vague answers")
    print(f"Output: survey_response.json")

    # Answer distribution analysis
    options = Counter(r.get("sampled_option", "unknown") for r in responses)
    print(f"\nAnswer distribution:")
    for opt, count in options.most_common():
        bar = "#" * (count // 3)
        print(f"  {opt:<20} {count:>3} ({count/len(responses)*100:.1f}%) {bar}")

    # Segment x Answer cross-tab
    seg_answers = {}
    for a, r in zip(agents, responses):
        seg = a["persona"].meta.population_segment or "unknown"
        ans = r.get("sampled_option", "unknown")
        seg_answers.setdefault(seg, Counter())[ans] += 1

    print(f"\nSegment x Answer cross-tabulation:")
    all_opts = ["rarely", "1-2 per week", "3-4 per week", "daily", "multiple per day"]
    header = f"  {'Segment':<25}" + "".join(f"{o:<18}" for o in all_opts)
    print(header)
    print("  " + "-" * (25 + 18 * len(all_opts)))
    for seg in sorted(seg_answers.keys()):
        counts = seg_answers[seg]
        total = sum(counts.values())
        row = f"  {seg:<25}"
        for opt in all_opts:
            c = counts.get(opt, 0)
            row += f"{c:>3} ({c/total*100:4.0f}%)       "
        print(row)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    asyncio.run(regenerate(n))
