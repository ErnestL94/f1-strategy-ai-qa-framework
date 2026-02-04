"""
Tests for scenario embedding with hybrid approach.
"""

import numpy as np
import pytest

from src.rag.embeddings import ScenarioEmbedder


class TestScenarioEmbedder:
    """Test hybrid embedding generation"""

    @pytest.fixture
    def embedder(self):
        """Initialize embedder once for all tests"""
        return ScenarioEmbedder()

    def test_embedder_initialization(self, embedder):
        """Embedder initializes with correct dimensions"""
        assert embedder is not None
        assert embedder.text_dim == 384
        assert embedder.numerical_dim == 11
        assert embedder.embedding_dim == 395

    def test_scenario_to_text(self, embedder):
        """Scenario converts to readable text"""
        scenario = {
            "lap": 30,
            "position": 2,
            "tires": {"compound": "MEDIUM", "age_laps": 30},
            "weather": {"condition": "dry"},
        }

        text = embedder.scenario_to_text(scenario)

        assert "lap 30" in text.lower()
        assert "medium" in text.lower()
        assert "30 laps" in text.lower()
        assert "dry" in text.lower()

    def test_extract_numerical_features(self, embedder):
        """Numerical features are extracted and normalized"""
        scenario = {
            "lap": 30,
            "position": 2,
            "tires": {"compound": "MEDIUM", "age_laps": 30, "condition": "high_wear"},
            "weather": {"condition": "dry"},
            "gaps": {"to_p2": 5.0, "to_p4": 10.0},
        }

        features = embedder.extract_numerical_features(scenario)

        assert len(features) == 11
        assert np.all(features >= 0)  # All normalized to >= 0
        assert np.all(features <= 1)  # All normalized to <= 1

        # Check specific values
        assert features[0] > 0.5  # Tire age 30/50 = 0.6
        assert features[1] == 0.25  # MEDIUM = 1/4
        assert features[2] > 0.4  # Lap 30/70 ≈ 0.43

    def test_embed_single_scenario(self, embedder):
        """Can embed a single scenario with hybrid approach"""
        scenario = {"lap": 30, "tires": {"compound": "MEDIUM", "age_laps": 30}}

        embedding = embedder.embed(scenario)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 395  # 384 text + 11 numerical
        assert not np.isnan(embedding).any()

    def test_similar_scenarios_have_similar_embeddings(self, embedder):
        """Similar scenarios should have high cosine similarity"""
        scenario_a = {
            "lap": 30,
            "position": 2,
            "tires": {"compound": "MEDIUM", "age_laps": 30},
            "weather": {"condition": "dry"},
        }

        scenario_b = {
            "lap": 28,
            "position": 3,
            "tires": {"compound": "MEDIUM", "age_laps": 28},
            "weather": {"condition": "dry"},
        }

        scenario_c = {
            "lap": 1,
            "position": 1,
            "tires": {"compound": "SOFT", "age_laps": 0},
            "weather": {"condition": "dry"},
        }

        emb_a = embedder.embed(scenario_a)
        emb_b = embedder.embed(scenario_b)
        emb_c = embedder.embed(scenario_c)

        # Cosine similarity
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_ab = cosine_similarity(emb_a, emb_b)
        sim_ac = cosine_similarity(emb_a, emb_c)

        # A and B should be more similar than A and C
        assert sim_ab > sim_ac
        assert sim_ab > 0.85  # Very high similarity with hybrid

    def test_batch_embedding(self, embedder):
        """Can embed multiple scenarios efficiently"""
        scenarios = [
            {"lap": 30, "tires": {"compound": "MEDIUM", "age_laps": 30}},
            {"lap": 1, "tires": {"compound": "SOFT", "age_laps": 0}},
            {"lap": 50, "tires": {"compound": "HARD", "age_laps": 40}},
        ]

        embeddings = embedder.embed_batch(scenarios)

        assert embeddings.shape == (3, 395)
        assert not np.isnan(embeddings).any()

    def test_numerical_features_distinguish_scenarios(self, embedder):
        """Numerical features help distinguish similar text scenarios"""
        # Both are "old tires" but different ages
        scenario_a = {"lap": 30, "tires": {"compound": "MEDIUM", "age_laps": 30}}

        scenario_b = {"lap": 30, "tires": {"compound": "MEDIUM", "age_laps": 40}}

        features_a = embedder.extract_numerical_features(scenario_a)
        features_b = embedder.extract_numerical_features(scenario_b)

        # Tire age feature should be different
        assert features_a[0] < features_b[0]  # 30 laps < 40 laps

    def test_safety_car_types_distinguished(self, embedder):
        """SC and VSC should be distinguished in features"""
        scenario_sc = {
            "lap": 20,
            "race_state": "safety_car",
            "tires": {"compound": "MEDIUM", "age_laps": 20},
        }

        scenario_vsc = {
            "lap": 20,
            "race_state": "virtual_safety_car",
            "tires": {"compound": "MEDIUM", "age_laps": 20},
        }

        scenario_normal = {
            "lap": 20,
            "race_state": "racing",
            "tires": {"compound": "MEDIUM", "age_laps": 20},
        }

        features_sc = embedder.extract_numerical_features(scenario_sc)
        features_vsc = embedder.extract_numerical_features(scenario_vsc)
        features_normal = embedder.extract_numerical_features(scenario_normal)

        # Feature 8: SC flag
        assert features_sc[8] == 1.0  # SC active
        assert features_vsc[8] == 0.0  # SC not active
        assert features_normal[8] == 0.0

        # Feature 9: VSC flag
        assert features_sc[9] == 0.0  # VSC not active
        assert features_vsc[9] == 1.0  # VSC active
        assert features_normal[9] == 0.0

    def test_vsc_abbreviation_detected(self, embedder):
        """VSC abbreviation should be recognized"""
        scenario_vsc_abbr = {
            "lap": 44,
            "race_state": "vsc",
            "tires": {"compound": "HARD", "age_laps": 24},
        }

        features = embedder.extract_numerical_features(scenario_vsc_abbr)

        # Should detect VSC
        assert features[9] == 1.0  # VSC flag
        assert features[8] == 0.0  # Not full SC

    def test_handles_dict_position(self, embedder):
        """Can handle position as dict or int"""
        # Position as int
        scenario_int = {"lap": 30, "position": 3, "tires": {"compound": "MEDIUM", "age_laps": 30}}

        # Position as dict
        scenario_dict = {
            "lap": 30,
            "position": {"current": 3, "start": 2},
            "tires": {"compound": "MEDIUM", "age_laps": 30},
        }

        features_int = embedder.extract_numerical_features(scenario_int)
        features_dict = embedder.extract_numerical_features(scenario_dict)

        # Both should work
        assert len(features_int) == 11
        assert len(features_dict) == 11

        # Position feature should be the same
        assert features_int[3] == features_dict[3]
