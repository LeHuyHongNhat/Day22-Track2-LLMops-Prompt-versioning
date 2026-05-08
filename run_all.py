"""
run_all.py — Orchestrator for Day 22 Lab
=========================================
Run all steps sequentially: python run_all.py
Run a single step:         python run_all.py --step 3
"""

import sys
import subprocess
from pathlib import Path

STEPS = {
    1: "01_langsmith_rag_pipeline.py",
    2: "02_prompt_hub_ab_routing.py",
    3: "03_ragas_evaluation.py",
    4: "04_guardrails_validator.py",
}

STEP_DESCRIPTIONS = {
    1: "LangSmith RAG Pipeline",
    2: "Prompt Hub & A/B Routing",
    3: "RAGAS Evaluation (~20-30 min)",
    4: "Guardrails AI Validators",
}


def run_step(step_num: int):
    """Execute a single lab step as a subprocess."""
    script = Path(__file__).parent / STEPS[step_num]
    desc = STEP_DESCRIPTIONS[step_num]

    print(f"\n{'=' * 60}")
    print(f"  Running Step {step_num}: {desc}")
    print(f"{'=' * 60}")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=Path(__file__).parent,
    )

    if result.returncode != 0:
        print(f"\n❌ Step {step_num} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n✅ Step {step_num} complete!")


def main():
    args = sys.argv[1:]

    if "--step" in args:
        idx = args.index("--step")
        step = int(args[idx + 1])
        if step not in STEPS:
            print(f"Invalid step: {step}. Choose from {list(STEPS.keys())}")
            sys.exit(1)
        run_step(step)
    else:
        print("=" * 60)
        print("  Day 22 Lab — Full Pipeline")
        print("=" * 60)

        for step_num in sorted(STEPS):
            run_step(step_num)

        print("\n" + "=" * 60)
        print("  All steps complete! Check evidence/ folder.")
        print("=" * 60)


if __name__ == "__main__":
    main()
