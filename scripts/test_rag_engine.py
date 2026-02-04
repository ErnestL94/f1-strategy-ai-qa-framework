"""
Test the RAG engine manually.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.engine import F1StrategyRAG


def main():
    print("=" * 60)
    print("TESTING RAG ENGINE")
    print("=" * 60)

    # Initialize RAG
    print("\nInitializing RAG engine with Ollama...")
    rag = F1StrategyRAG()

    # Test scenario (similar to Piastri Silverstone Lap 30)
    scenario = {
        "lap": 30,
        "driver": "VER",
        "position": 2,
        "tires": {"compound": "MEDIUM", "age_laps": 30, "condition": "high_wear"},
        "weather": {"condition": "dry"},
        "race_state": "racing",
        "gaps": {"to_p1": 5.0, "to_p3": 10.0},
    }

    print("\n" + "=" * 60)
    print("TEST SCENARIO")
    print("=" * 60)
    print(f"Lap: {scenario['lap']}")
    print(f"Position: P{scenario['position']}")
    print(f"Tires: {scenario['tires']['compound']} ({scenario['tires']['age_laps']} laps old)")
    print(f"Condition: {scenario['tires']['condition']}")
    print(f"Gaps: {scenario['gaps']}")

    print("\n" + "=" * 60)
    print("Generating recommendation...")
    print("=" * 60)

    try:
        result = rag.generate_strategy(scenario)

        print("\n" + "=" * 60)
        print("RAG RECOMMENDATION")
        print("=" * 60)
        print(f"Decision: {result['decision']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"\nReasoning:")
        print(f"  {result['reasoning']}")

        print(f"\n" + "=" * 60)
        print(f"RETRIEVED {result['num_retrieved']} SIMILAR SCENARIOS")
        print("=" * 60)
        for i, retrieved in enumerate(result["retrieved_scenarios"][:3], 1):
            print(f"\n{i}. {retrieved['id']}")
            print(f"   Similarity: {retrieved['similarity']:.3f}")
            print(f"   Track: {retrieved['metadata']['track']}")
            print(
                f"   Lap {retrieved['metadata']['lap']}, "
                f"{retrieved['metadata']['tire_compound']} "
                f"({retrieved['metadata']['tire_age']} laps old)"
            )

        print("\n" + "=" * 60)
        print("✓ RAG ENGINE WORKING!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("  1. Is Ollama running? (ollama serve)")
        print("  2. Is the model pulled? (ollama pull llama3.2:3b)")
        print("  3. Is the vector DB built? (python scripts/ingest_golden_scenarios.py)")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
