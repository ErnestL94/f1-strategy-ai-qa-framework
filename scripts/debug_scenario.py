"""
Deep dive into a specific scenario to understand RAG behavior.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json

from src.rag.agent import F1StrategyAgent
from src.rag.engine_v2 import F1StrategyRAG  # ← Using v2!


def load_scenario(scenario_id: str):
    """Load a scenario by ID from golden datasets"""
    golden_dir = Path("datasets/golden")

    if not golden_dir.exists():
        print(f"❌ Golden dataset directory not found: {golden_dir}")
        return None, None

    dataset_files = list(golden_dir.glob("*.json"))

    if not dataset_files:
        print(f"❌ No dataset files found in {golden_dir}")
        return None, None

    for dataset_file in dataset_files:
        with open(dataset_file) as f:
            data = json.load(f)

        race_info = data.get("race", {})

        for s in data["scenarios"]:
            if s["id"] == scenario_id:
                # CRITICAL: Add race info to scenario if missing
                if "race" not in s or not s["race"]:
                    s["race"] = race_info
                return s, race_info

    return None, None


def analyze_scenario(scenario_id: str):
    """Deep analysis of a specific scenario"""

    print("=" * 80)
    print(f"DEEP DIVE: {scenario_id}")
    print("=" * 80)

    # Load scenario
    scenario, race_info = load_scenario(scenario_id)

    if not scenario:
        print(f"\n❌ Scenario '{scenario_id}' not found")
        print("\nAvailable scenarios:")

        golden_dir = Path("datasets/golden")
        dataset_files = list(golden_dir.glob("*.json"))

        for dataset_file in sorted(dataset_files):
            with open(dataset_file) as f:
                data = json.load(f)
            print(f"\n{data['race']['track']}:")
            for s in data["scenarios"]:
                print(f"  - {s['id']}")
        return

    # Display scenario details
    print("\n" + "=" * 80)
    print("SCENARIO DETAILS")
    print("=" * 80)

    track = scenario.get("race", {}).get("track", "Unknown")
    year = scenario.get("race", {}).get("year", "Unknown")
    print(f"Track: {track} ({year})")
    print(f"Lap: {scenario.get('lap')}")
    print(f"Driver: {scenario.get('driver')}")
    print(f"Position: {scenario.get('position')}")

    tires = scenario.get("tires", {})
    print(f"\nTires:")
    print(f"  Compound: {tires.get('compound')}")
    print(f"  Age: {tires.get('age_laps')} laps")
    print(f"  Condition: {tires.get('condition')}")

    print(f"\nWeather: {scenario.get('weather', {}).get('condition')}")
    print(f"Race State: {scenario.get('race_state', 'racing')}")
    print(f"Gaps: {scenario.get('gaps', {})}")

    print(f"\nContext:")
    context = scenario.get("context", {})
    print(f"  Description: {context.get('description', 'N/A')}")
    print(f"  Strategy: {context.get('strategy_type', 'N/A')}")

    golden = scenario.get("golden_truth", {})
    print(f"\n✓ GOLDEN TRUTH:")
    print(f"  Decision: {golden.get('decision')}")
    print(f"  Reasoning: {golden.get('reasoning', 'N/A')}")
    print(f"  Risk: {golden.get('risk_level', 'N/A')}")

    # Initialize agents
    print("\n" + "=" * 80)
    print("INITIALIZING AGENTS...")
    print("=" * 80)

    rag_agent = F1StrategyRAG()
    rule_agent = F1StrategyAgent()

    # Test Rule-based
    print("\n" + "=" * 80)
    print("RULE-BASED AGENT")
    print("=" * 80)

    rule_result = rule_agent.generate_strategy(scenario)
    if isinstance(rule_result, tuple):
        rule_decision = rule_result[0]
        rule_reasoning = rule_result[1] if len(rule_result) > 1 else "N/A"
    else:
        rule_decision = rule_result.get("decision")
        rule_reasoning = rule_result.get("reasoning", "N/A")

    rule_match = rule_decision == golden.get("decision")
    print(f"Decision: {rule_decision} {'✓' if rule_match else '✗'}")
    print(f"Reasoning: {rule_reasoning[:200] if rule_reasoning != 'N/A' else 'N/A'}...")

    # Test RAG with full details
    print("\n" + "=" * 80)
    print("RAG AGENT V2 - RETRIEVAL")
    print("=" * 80)

    # Get retrieved scenarios
    retrieved = rag_agent.vectordb.search(scenario, k=5)

    print(f"\nRetrieved {len(retrieved)} similar scenarios:\n")
    for i, ret in enumerate(retrieved, 1):
        ret_scenario = ret["scenario"]
        ret_golden = ret_scenario.get("golden_truth", {})

        print(f"{i}. {ret['id']}")
        print(f"   Similarity: {ret['similarity']:.3f}")
        print(f"   Track: {ret['metadata'].get('track', 'Unknown')}")
        print(
            f"   Lap {ret['metadata'].get('lap')}, "
            f"{ret['metadata'].get('tire_compound')} "
            f"({ret['metadata'].get('tire_age')} laps)"
        )
        print(f"   Decision: {ret_golden.get('decision', 'UNKNOWN')}")
        print(f"   Reasoning: {ret_golden.get('reasoning', 'N/A')[:100]}...")
        print()

    # Count retrieved decisions
    retrieved_decisions = [r["scenario"].get("golden_truth", {}).get("decision") for r in retrieved]
    box_count = retrieved_decisions.count("BOX")
    stay_count = retrieved_decisions.count("STAY_OUT")

    print(f"Retrieved scenario bias:")
    print(f"  BOX: {box_count}/5 ({box_count/5*100:.0f}%)")
    print(f"  STAY_OUT: {stay_count}/5 ({stay_count/5*100:.0f}%)")

    # Calculate average similarity
    avg_sim = sum(r["similarity"] for r in retrieved) / len(retrieved) if retrieved else 0
    print(f"  Average similarity: {avg_sim:.3f}")

    # Test RAG decision
    print("\n" + "=" * 80)
    print("RAG AGENT V2 - DECISION")
    print("=" * 80)

    # Add debug info about what the prompt will receive
    lap = scenario.get("lap", 0)
    track_name = scenario.get("race", {}).get("track", None)
    race_length = rag_agent.estimate_race_length(lap, track_name)
    race_phase = rag_agent.get_race_phase(lap, race_length)
    laps_remaining = race_length - lap

    print(f"\nPrompt will receive:")
    print(f"  Track: {track_name}")
    print(f"  Lap: {lap} of ~{race_length}")
    print(f"  Race Phase: {race_phase}")
    print(f"  Laps Remaining: ~{laps_remaining}")
    print(f"  Retrieved consensus: {stay_count}x STAY_OUT, {box_count}x BOX")

    try:
        rag_result = rag_agent.generate_strategy(scenario, k=5)
        rag_decision = rag_result["decision"]
        rag_confidence = rag_result["confidence"]
        rag_reasoning = rag_result["reasoning"]
        rag_risk = rag_result["risk_level"]

        rag_match = rag_decision == golden.get("decision")

        print(f"\nDecision: {rag_decision} {'✓' if rag_match else '✗'}")
        print(f"Confidence: {rag_confidence:.0%}")
        print(f"Risk Level: {rag_risk}")
        print(f"\nReasoning:")
        print(f"  {rag_reasoning}")

    except Exception as e:
        print(f"\n❌ RAG failed: {e}")
        import traceback

        traceback.print_exc()
        rag_match = False
        rag_decision = "ERROR"

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    if not rag_match:
        print("\n❌ RAG V2 FAILED - Why?")

        print(f"\n1. Race Phase Detection:")
        print(f"   → Detected: {race_phase}")
        print(f"   → Laps remaining: {laps_remaining}")
        if laps_remaining <= 3:
            print(f"   ⚠️  SHOULD TRIGGER FINAL_LAPS RULE!")

        print(f"\n2. Retrieved Bias:")
        print(f"   → {box_count}/5 scenarios recommend BOX")
        print(f"   → {stay_count}/5 scenarios recommend STAY_OUT")
        if stay_count >= 3:
            print(f"   → Consensus for STAY_OUT - LLM should follow!")

        print(f"\n3. Similarity Quality:")
        print(f"   → Average similarity: {avg_sim:.3f}")
        if avg_sim < 0.5:
            print(f"   → Low similarity - retrieved scenarios not very relevant")

        print(f"\n4. Specific Context Issues:")
        if tires.get("age_laps", 0) < 5:
            print(f"   → Fresh tires (age={tires.get('age_laps')})")
            print(f"   → SHOULD TRIGGER FRESH TIRE RULE")

        if laps_remaining <= 3:
            print(f"   → Final laps (remaining={laps_remaining})")
            print(f"   → SHOULD TRIGGER FINAL LAPS RULE")

        if "restart" in scenario.get("context", {}).get("description", "").lower():
            print(f"   → Safety Car restart scenario")

        print(f"\n5. LLM Understanding:")
        print(f"   → Did it mention the critical rules in reasoning?")
        print(f"   → Did it mention retrieved consensus?")
        if rag_decision != "ERROR":
            if "final" not in rag_reasoning.lower() and laps_remaining <= 3:
                print(f"   ⚠️  Did NOT mention final laps!")
            if "retrieved" not in rag_reasoning.lower() and "scenario" not in rag_reasoning.lower():
                print(f"   ⚠️  Did NOT mention retrieved scenarios!")

    else:
        print("\n✅ RAG V2 SUCCEEDED!")
        print(f"\n✓ Correctly identified: {rag_decision}")
        print(f"✓ Race phase: {race_phase}")
        print(f"✓ Followed retrieved consensus: {stay_count}x STAY_OUT, {box_count}x BOX")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_scenario.py <scenario_id>")
        print("\nAvailable scenarios:")

        golden_dir = Path("datasets/golden")
        if not golden_dir.exists():
            print(f"❌ Golden dataset directory not found: {golden_dir}")
            return

        dataset_files = list(golden_dir.glob("*.json"))
        if not dataset_files:
            print(f"❌ No dataset files found in {golden_dir}")
            return

        for dataset_file in sorted(dataset_files):
            with open(dataset_file) as f:
                data = json.load(f)
            print(f"\n{data['race']['track']}:")
            for s in data["scenarios"]:
                print(f"  - {s['id']}")
        return

    scenario_id = sys.argv[1]
    analyze_scenario(scenario_id)


if __name__ == "__main__":
    main()
