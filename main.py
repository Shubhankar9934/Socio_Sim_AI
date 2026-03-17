"""
JADU – CLI entry point. Run API server or one-off commands.
"""

import asyncio
import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        import uvicorn
        uvicorn.run(
            "api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "generate":
        # One-off: generate population and print realism score
        from population.synthesis import generate_population
        from population.validator import validate_population
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        personas = generate_population(n=n, method="bayesian")
        passed, score, per_attr = validate_population(personas)
        print(f"Generated {len(personas)} agents. Realism score: {score:.3f}, passed: {passed}")
        for k, v in per_attr.items():
            print(f"  {k}: {v:.3f}")
    else:
        print("Usage: python main.py run          # Start API server on :8000")
        print("       python main.py generate [N] # Generate N agents (default 100) and validate")
        sys.exit(1)


if __name__ == "__main__":
    main()
