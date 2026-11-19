"""
Runs the golden eval set against the live multi-agent graph and checks
structural properties of each run (which agents were used, whether the
itinerary has the expected shape) rather than exact-string matching, since
LLM output varies between runs.

Usage:
    python -m eval.run_eval                 # print report
    python -m eval.run_eval --fail-under 0.8 # exit 1 if pass rate < 0.8 (for CI)
"""

import argparse
import sys
import uuid

from agent.agentic_workflow import GraphBuilder
from eval.test_cases import TEST_CASES


def check_case(case: dict, final_state: dict) -> tuple[bool, list[str]]:
    checks = case["checks"]
    problems = []

    draft = final_state.get("itinerary_draft") or {}
    days = draft.get("daily_itinerary") or []
    counts = final_state.get("agent_call_counts") or {}

    if len(days) < checks.get("min_days_covered", 0):
        problems.append(f"expected >= {checks['min_days_covered']} days, got {len(days)}")

    for agent in checks.get("must_call_agents", []):
        if counts.get(agent, 0) < 1:
            problems.append(f"expected {agent} to be called at least once, call counts={counts}")

    if checks.get("itinerary_has_cost_fields"):
        if not draft.get("total_estimated_cost"):
            problems.append("expected total_estimated_cost to be populated")

    return (len(problems) == 0, problems)


def run_eval(model_provider: str = "groq"):
    builder = GraphBuilder(model_provider=model_provider)
    graph = builder.build_graph()

    results = []
    for case in TEST_CASES:
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        try:
            final_state = graph.invoke(
                {
                    "messages": [case["query"]],
                    "next_agent": "",
                    "user_profile": {},
                    "itinerary_draft": None,
                    "critique_feedback": None,
                    "agent_call_counts": {},
                },
                config=config,
            )
            passed, problems = check_case(case, final_state)
        except Exception as e:  # noqa: BLE001
            passed, problems = False, [f"exception: {e}"]

        results.append({"id": case["id"], "passed": passed, "problems": problems})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-under", type=float, default=None)
    parser.add_argument("--model-provider", default="groq")
    args = parser.parse_args()

    results = run_eval(model_provider=args.model_provider)

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)
    pass_rate = passed_count / total if total else 0.0

    print(f"\n{'PASS' if pass_rate == 1.0 else 'PARTIAL' if pass_rate > 0 else 'FAIL'} "
          f"- {passed_count}/{total} cases passed ({pass_rate:.0%})\n")
    for r in results:
        status = "✅" if r["passed"] else "❌"
        print(f"{status} {r['id']}")
        for p in r["problems"]:
            print(f"    - {p}")

    if args.fail_under is not None and pass_rate < args.fail_under:
        print(f"\nPass rate {pass_rate:.0%} below required {args.fail_under:.0%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
