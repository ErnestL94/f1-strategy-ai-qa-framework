"""
Tests for vector database functionality.
"""

import json
from pathlib import Path

import pytest

from src.rag.vectordb import VectorDatabase


class TestVectorDatabase:
    """Test vector database operations"""

    @pytest.fixture
    def db(self, tmp_path):
        """Create temporary database for testing"""
        db = VectorDatabase(
            collection_name="test_scenarios", persist_directory=str(tmp_path / "vectordb")
        )
        yield db
        # Cleanup
        db.clear()

    @pytest.fixture
    def sample_scenario(self):
        """Sample scenario for testing"""
        return {
            "id": "test_scenario_1",
            "lap": 30,
            "driver": "VER",
            "position": 2,
            "tires": {"compound": "MEDIUM", "age_laps": 30},
            "weather": {"condition": "dry"},
            "race": {"track": "Test Track"},
        }

    def test_database_initialization(self, db):
        """Database can be initialized"""
        assert db is not None
        assert db.collection is not None
        assert db.embedder is not None

    def test_ingest_single_scenario(self, db, sample_scenario):
        """Can ingest a single scenario"""
        scenario_id = sample_scenario["id"]

        db.ingest_scenario(sample_scenario, scenario_id)

        # Verify it was added
        assert db.collection.count() == 1

        # Retrieve it
        retrieved = db.get_by_id(scenario_id)
        assert retrieved is not None
        assert retrieved["lap"] == 30
        assert retrieved["driver"] == "VER"

    def test_ingest_from_directory(self, db):
        """Can ingest scenarios from golden dataset directory"""
        golden_dir = Path("datasets/golden")

        if not golden_dir.exists():
            pytest.skip("Golden dataset directory not found")

        count = db.ingest_from_directory(golden_dir)

        assert count > 0
        assert db.collection.count() == count

        # Verify stats
        stats = db.stats()
        assert stats["total_scenarios"] == count
        assert len(stats["tracks"]) > 0

    def test_search_similar_scenarios(self, db, sample_scenario):
        """Can search for similar scenarios"""
        # Ingest some test scenarios
        db.ingest_scenario(sample_scenario, "test_1")

        similar_scenario = {
            "lap": 28,
            "driver": "HAM",
            "position": 3,
            "tires": {"compound": "MEDIUM", "age_laps": 28},
            "weather": {"condition": "dry"},
        }
        db.ingest_scenario(similar_scenario, "test_2")

        different_scenario = {
            "lap": 1,
            "driver": "LEC",
            "position": 1,
            "tires": {"compound": "SOFT", "age_laps": 0},
            "weather": {"condition": "dry"},
        }
        db.ingest_scenario(different_scenario, "test_3")

        # Search using first scenario
        results = db.search(sample_scenario, k=2)

        assert len(results) == 2
        assert results[0]["id"] == "test_1"  # Should find itself first
        assert results[0]["similarity"] > 0.95  # Very high similarity to itself
        assert results[1]["id"] == "test_2"  # Similar scenario second

    def test_search_returns_similarity_scores(self, db, sample_scenario):
        """Search results include similarity scores"""
        db.ingest_scenario(sample_scenario, "test_1")

        results = db.search(sample_scenario, k=1)

        assert len(results) == 1
        assert "similarity" in results[0]
        assert 0.0 <= results[0]["similarity"] <= 1.0

    def test_clear_database(self, db, sample_scenario):
        """Can clear all scenarios"""
        db.ingest_scenario(sample_scenario, "test_1")
        assert db.collection.count() == 1

        db.clear()
        assert db.collection.count() == 0

    def test_stats(self, db):
        """Can get database statistics"""
        stats = db.stats()

        assert "total_scenarios" in stats
        assert "tracks" in stats
        assert "embedding_dim" in stats
        assert stats["embedding_dim"] == 395
