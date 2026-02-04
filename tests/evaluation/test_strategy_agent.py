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
        assert hasattr(agent, 'generate_strategy')
        assert callable(agent.generate_strategy)
    
    def test_norris_lap1_recommendation(self, agent, golden_dataset):
        """Agent recommends STAY_OUT for Norris lap 1"""
        # Get the scenario
        scenario = next(
            s for s in golden_dataset["scenarios"]
            if s["id"] == "silverstone_2023_norris_lap1"
        )
        
        # Get AI recommendation
        recommendation = agent.generate_strategy(scenario)
        
        # Check against golden truth
        expected = scenario["golden_truth"]
        
        # Core assertions
        assert recommendation["decision"] == expected["decision"], \
            f"Expected {expected['decision']}, got {recommendation['decision']}"
        
        assert recommendation["confidence_score"] >= 0.80, \
            f"Confidence too low: {recommendation['confidence_score']}"
        
        # Should have rationale
        assert "rationale" in recommendation
        assert len(recommendation["rationale"]) > 10
    
    @pytest.mark.parametrize("scenario_id", [
        "silverstone_2023_norris_lap1",
        "silverstone_2023_russell_lap28",
        "silverstone_2023_piastri_lap30",
        "silverstone_2023_norris_lap34_restart",
        "silverstone_2023_hamilton_lap34_sc"
    ])
    def test_all_silverstone_scenarios(self, agent, golden_dataset, scenario_id):
        """Test agent against all Silverstone scenarios"""
        # Get scenario
        scenario = next(
            s for s in golden_dataset["scenarios"]
            if s["id"] == scenario_id
        )
        
        # Get recommendation
        recommendation = agent.generate_strategy(scenario)
        expected = scenario["golden_truth"]
        
        # Check decision matches
        assert recommendation["decision"] == expected["decision"], \
            f"[{scenario_id}] Expected {expected['decision']}, got {recommendation['decision']}"
        
        # Print for debugging
        print(f"\n✓ {scenario_id}: {recommendation['decision']} (confidence: {recommendation['confidence_score']:.2f})")