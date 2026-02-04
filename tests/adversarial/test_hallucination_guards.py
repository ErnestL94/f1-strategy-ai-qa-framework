"""
Adversarial tests to detect hallucinations and breaking points.
These tests are designed to FAIL and expose weaknesses.
"""
import json
import pytest
from pathlib import Path
from src.rag.agent import F1StrategyAgent


class TestHallucinationGuards:
    """Test that agent rejects invalid/impossible scenarios"""
    
    @pytest.fixture
    def agent(self):
        return F1StrategyAgent()
    
    @pytest.fixture
    def adversarial_scenarios(self, project_root):
        """Load adversarial test scenarios"""
        dataset_path = project_root / "datasets" / "adversarial" / "impossible_scenarios.json"
        
        if not dataset_path.exists():
            pytest.skip(f"Adversarial dataset not found at {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            return json.load(f)
    
    def test_rejects_fictional_compound(self, agent, adversarial_scenarios):
        """Agent should reject HYPERSOFT (doesn't exist anymore)"""
        scenario = next(
            s for s in adversarial_scenarios["scenarios"]
            if s["id"] == "adv_fictional_compound"
        )
        
        # Agent should either reject or flag this
        with pytest.raises(ValueError, match="invalid.*compound|HYPERSOFT"):
            agent.generate_strategy(scenario)
    
    def test_rejects_negative_tire_age(self, agent, adversarial_scenarios):
        """Agent should reject negative tire age"""
        scenario = next(
            s for s in adversarial_scenarios["scenarios"]
            if s["id"] == "adv_negative_tire_age"
        )
        
        with pytest.raises(ValueError, match="negative|invalid.*age"):
            agent.generate_strategy(scenario)
    
    def test_rejects_impossible_tire_age(self, agent, adversarial_scenarios):
        """Agent should reject tire age > race laps"""
        scenario = next(
            s for s in adversarial_scenarios["scenarios"]
            if s["id"] == "adv_impossible_tire_age"
        )
        
        with pytest.raises(ValueError, match="(?i)tire age.*exceeds|impossible"):
            agent.generate_strategy(scenario)
    
    def test_rejects_future_lap(self, agent, adversarial_scenarios):
        """Agent should reject lap > race distance"""
        scenario = next(
            s for s in adversarial_scenarios["scenarios"]
            if s["id"] == "adv_future_lap"
        )
        
        with pytest.raises(ValueError):
            agent.generate_strategy(scenario)
    
    def test_handles_slicks_in_heavy_rain(self, agent, adversarial_scenarios):
        """Agent MUST recommend BOX for slicks in heavy rain"""
        scenario = next(
            s for s in adversarial_scenarios["scenarios"]
            if s["id"] == "adv_slicks_in_monsoon"
        )
        
        recommendation = agent.generate_strategy(scenario)
        expected = scenario["expected_behavior"]
        
        assert recommendation["decision"] == "BOX", \
            "Must recommend pitting for slicks in heavy rain"
        
        assert recommendation["confidence_score"] >= expected["confidence_minimum"], \
            f"Should be very confident about rain change"
    
    def test_handles_wets_on_dry_track(self, agent, adversarial_scenarios):
        """Agent should recommend change from WETs on dry track"""
        scenario = next(
            s for s in adversarial_scenarios["scenarios"]
            if s["id"] == "adv_wets_on_dry_track"
        )
        
        recommendation = agent.generate_strategy(scenario)
        expected = scenario["expected_behavior"]
        
        assert recommendation["decision"] == "BOX", \
            "Must recommend pitting for WETs on dry track"
        
        assert recommendation["confidence_score"] >= expected["confidence_minimum"]


class TestConfidenceCalibration:
    """Test that confidence scores are meaningful"""
    
    @pytest.fixture
    def agent(self):
        return F1StrategyAgent()
    
    def test_high_confidence_for_clear_decisions(self, agent):
        """Clear-cut scenarios should have high confidence"""
        # Fresh tires at start = obvious STAY_OUT
        scenario = {
            "lap": 1,
            "tires": {"compound": "MEDIUM", "age_laps": 0},
            "weather": {"condition": "dry"}
        }
        
        recommendation = agent.generate_strategy(scenario)
        assert recommendation["confidence_score"] >= 0.85, \
            "Fresh tires should have high confidence"
    
    def test_low_confidence_for_ambiguous_decisions(self, agent):
        """Ambiguous scenarios should have lower confidence"""
        # Mid-stint, no gaps data, stable weather = uncertain
        scenario = {
            "lap": 25,
            "tires": {"compound": "MEDIUM", "age_laps": 20},
            "weather": {"condition": "dry"}
        }
        
        recommendation = agent.generate_strategy(scenario)
        assert recommendation["confidence_score"] < 0.80, \
            "Ambiguous situations should have lower confidence"