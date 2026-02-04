"""
Tests for RAG engine with Ollama.
"""

import pytest
import requests

from src.rag.engine import F1StrategyRAG


class TestF1StrategyRAG:
    """Test RAG engine functionality"""

    @pytest.fixture
    def rag(self):
        """Initialize RAG engine (requires Ollama running)"""
        # Check if Ollama is running
        try:
            requests.get("http://localhost:11434/api/tags", timeout=2)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pytest.skip("Ollama not running. Start with: ollama serve")

        return F1StrategyRAG()

    def test_engine_initialization(self, rag):
        """RAG engine can be initialized"""
        assert rag is not None
        assert rag.vectordb is not None
        assert rag.model is not None

    def test_generate_strategy_basic(self, rag):
        """Can generate strategy for a basic scenario"""
        scenario = {
            "lap": 30,
            "driver": "VER",
            "position": 2,
            "tires": {"compound": "MEDIUM", "age_laps": 30, "condition": "high_wear"},
            "weather": {"condition": "dry"},
            "race_state": "racing",
            "gaps": {"to_p1": 5.0, "to_p3": 10.0},
        }

        result = rag.generate_strategy(scenario)

        # Check structure
        assert "decision" in result
        assert "confidence" in result
        assert "reasoning" in result
        assert "risk_level" in result
        assert "retrieved_scenarios" in result

        # Check values
        assert result["decision"] in ["BOX", "STAY_OUT"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        assert len(result["reasoning"]) > 0
        assert len(result["retrieved_scenarios"]) > 0

    def test_parse_llm_response_valid_json(self, rag):
        """Can parse valid LLM JSON response"""
        response = """{"decision": "BOX", "confidence": 0.85, "reasoning": "Tire degradation suggests pit stop is optimal.", "risk_level": "LOW"}"""

        parsed = rag.parse_llm_response(response)

        assert parsed["decision"] == "BOX"
        assert parsed["confidence"] == 0.85
        assert "degradation" in parsed["reasoning"]
        assert parsed["risk_level"] == "LOW"
