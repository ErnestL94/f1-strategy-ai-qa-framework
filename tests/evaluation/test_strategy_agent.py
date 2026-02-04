"""
Evaluation tests for the F1 Strategy Agent.
Tests AI recommendations against golden dataset.
"""

import pytest


class TestStrategyAgentBaseline:
    """Test the strategy agent against golden dataset"""

    def test_agent_exists(self, agent):
        """Strategy agent can be instantiated"""
        assert agent is not None

    def test_agent_has_generate_strategy_method(self, agent):
        """Agent has the core generate_strategy method"""
        assert hasattr(agent, "generate_strategy")
        assert callable(agent.generate_strategy)

    def test_norris_lap1_recommendation(self, agent, golden_dataset):
        """Agent recommends STAY_OUT for Norris lap 1 (original test)"""
        scenario = next(
            s for s in golden_dataset["scenarios"] if s["id"] == "silverstone_2023_norris_lap1"
        )

        recommendation = agent.generate_strategy(scenario)
        expected = scenario["golden_truth"]

        assert (
            recommendation["decision"] == expected["decision"]
        ), f"Expected {expected['decision']}, got {recommendation['decision']}"

        assert (
            recommendation["confidence_score"] >= 0.80
        ), f"Confidence too low: {recommendation['confidence_score']}"

        assert "rationale" in recommendation
        assert len(recommendation["rationale"]) > 10

    @pytest.mark.parametrize(
        "scenario_id",
        [
            # Silverstone 2023
            "silverstone_2023_norris_lap1",
            "silverstone_2023_russell_lap28",
            "silverstone_2023_piastri_lap30",
            "silverstone_2023_norris_lap34_restart",
            "silverstone_2023_hamilton_lap34_sc",
            # Spa 2023
            "spa_2023_verstappen_lap14",
            "spa_2023_norris_lap17",
            "spa_2023_russell_lap22",
            "spa_2023_leclerc_lap24",
            "spa_2023_hamilton_lap42",
            # Singapore 2023
            "singapore_2023_leclerc_lap1",
            "singapore_2023_sainz_lap20",
            "singapore_2023_russell_lap44",
            "singapore_2023_sainz_lap60",
            "singapore_2023_russell_lap62",
        ],
    )
    def test_all_golden_scenarios(self, agent, all_golden_datasets, scenario_id):
        """Test agent against ALL golden scenarios across all tracks"""
        # Find the scenario across all datasets
        scenario = None
        for dataset_name, dataset in all_golden_datasets:
            for s in dataset["scenarios"]:
                if s["id"] == scenario_id:
                    scenario = s
                    break
            if scenario:
                break

        assert scenario is not None, f"Scenario {scenario_id} not found"

        # Get recommendation
        recommendation = agent.generate_strategy(scenario)
        expected = scenario["golden_truth"]

        # Check decision matches
        assert (
            recommendation["decision"] == expected["decision"]
        ), f"[{scenario_id}] Expected {expected['decision']}, got {recommendation['decision']}"

        # Print for debugging
        print(
            f"\n✓ {scenario_id}: {recommendation['decision']} (confidence: {recommendation['confidence_score']:.2f})"
        )
