"""
Compare RAG agent vs Rule-based agent on golden dataset.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json

from src.rag.agent import F1StrategyAgent
from src.rag.engine import F1StrategyRAG


def main():
    print("=" * 60)
    print("A/B TEST: RAG vs RULE-BASED AGENT")
    print("=" * 60)

    # Initialize both agents
    print("\nInitializing RAG agent...")
    rag_agent = F1StrategyRAG()

    print("Initializing rule-based agent...")
    rule_agent = F1StrategyAgent()

    # Load golden scenarios
    golden_files = sorted(Path("datasets/golden").glob("*.json"))

    rag_correct = 0
    rule_correct = 0
    total = 0

    comparisons = []

    print("\n" + "=" * 60)
    print("Testing on golden dataset...")
    print("=" * 60)

    for dataset_file in golden_files:
        with open(dataset_file) as f:
            data = json.load(f)

        track_name = data["race"]["track"]
        print(f"\n{track_name}:")

        for scenario in data["scenarios"]:
            total += 1
            scenario_id = scenario["id"]
            golden_decision = scenario["golden_truth"]["decision"]

            # Test Rule-based (returns tuple: decision, reasoning, confidence, risk)
            try:
                rule_result = rule_agent.generate_strategy(scenario)
                # Handle different return formats
                if isinstance(rule_result, tuple):
                    rule_decision = rule_result[0]
                elif isinstance(rule_result, dict):
                    rule_decision = rule_result.get("decision")
                else:
                    rule_decision = rule_result

                rule_match = rule_decision == golden_decision
                if rule_match:
                    rule_correct += 1
            except Exception as e:
                print(f"  ⚠️  Rule-based failed on {scenario_id}: {e}")
                rule_decision = "ERROR"
                rule_match = False

            # Test RAG
            rag_match = False
            rag_decision = None
            rag_confidence = 0.0
            try:
                rag_result = rag_agent.generate_strategy(scenario)
                rag_decision = rag_result["decision"]
                rag_confidence = rag_result["confidence"]
                rag_match = rag_decision == golden_decision
                if rag_match:
                    rag_correct += 1
            except Exception as e:
                print(f"  ⚠️  RAG failed on {scenario_id}: {e}")
                rag_decision = "ERROR"

            # Track comparison
            comparisons.append(
                {
                    "id": scenario_id,
                    "golden": golden_decision,
                    "rule": rule_decision,
                    "rag": rag_decision,
                    "rule_match": rule_match,
                    "rag_match": rag_match,
                    "rag_confidence": rag_confidence,
                }
            )

            # Show results
            rule_icon = "✓" if rule_match else "✗"
            rag_icon = "✓" if rag_match else "✗"
            print(
                f"  {scenario_id[:30]:30} | Golden: {golden_decision:8} | Rule: {rule_icon} | RAG: {rag_icon} ({rag_confidence:.0%})"
            )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total scenarios: {total}")
    print(f"\nRule-Based Agent: {rule_correct}/{total} ({rule_correct/total*100:.1f}%)")
    print(f"RAG Agent:        {rag_correct}/{total} ({rag_correct/total*100:.1f}%)")
    print(f"\nDifference: {abs(rag_correct - rule_correct)} scenarios")

    if rag_correct > rule_correct:
        print("\n🎉 RAG outperforms rule-based!")
    elif rule_correct > rag_correct:
        print("\n⚠️  Rule-based still better (RAG needs tuning)")
        print("\nScenarios where RAG disagreed:")
        for comp in comparisons:
            if comp["rule_match"] and not comp["rag_match"]:
                print(
                    f"  - {comp['id']}: Golden={comp['golden']}, RAG={comp['rag']} ({comp['rag_confidence']:.0%})"
                )
    else:
        print("\n🤝 Tie!")

    # Show where they differ
    disagreements = [
        c
        for c in comparisons
        if c["rule"] != c["rag"] and c["rag"] != "ERROR" and c["rule"] != "ERROR"
    ]
    if disagreements:
        print(f"\n" + "=" * 60)
        print(f"DISAGREEMENTS ({len(disagreements)} scenarios)")
        print("=" * 60)
        for comp in disagreements:
            print(f"\n{comp['id']}:")
            print(f"  Golden: {comp['golden']}")
            print(f"  Rule:   {comp['rule']} {'✓' if comp['rule_match'] else '✗'}")
            print(
                f"  RAG:    {comp['rag']} {'✓' if comp['rag_match'] else '✗'} ({comp['rag_confidence']:.0%})"
            )


if __name__ == "__main__":
    main()
