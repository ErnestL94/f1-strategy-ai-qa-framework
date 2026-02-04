"""
Vector database for F1 scenario similarity search.
Uses ChromaDB for storage and retrieval.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import numpy as np
from chromadb.config import Settings

from src.rag.embeddings import ScenarioEmbedder


class VectorDatabase:
    """
    Vector database for storing and retrieving F1 racing scenarios.
    Uses ChromaDB for persistent storage and semantic search.
    """

    def __init__(
        self, collection_name: str = "f1_scenarios", persist_directory: str = "data/vectordb"
    ):
        """
        Initialize vector database.
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Where to store the database
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize embedder
        print("Initializing embedder...")
        self.embedder = ScenarioEmbedder()

        # Initialize ChromaDB client
        print(f"Initializing ChromaDB (persist: {persist_directory})...")
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"description": "F1 racing strategy scenarios"}
        )

        print(f"✓ VectorDB ready. Collection: {collection_name}")
        print(f"  Current scenarios: {self.collection.count()}")

    def ingest_scenario(self, scenario: Dict, scenario_id: str) -> None:
        """
        Add a single scenario to the database.

        Args:
            scenario: Scenario dict
            scenario_id: Unique ID for the scenario
        """
        # Generate embedding
        embedding = self.embedder.embed(scenario)

        # Store in ChromaDB
        self.collection.add(
            embeddings=[embedding.tolist()],
            documents=[json.dumps(scenario)],  # Store full scenario as JSON
            ids=[scenario_id],
            metadatas=[
                {
                    "track": scenario.get("race", {}).get("track", "unknown"),
                    "lap": scenario.get("lap", 0),
                    "driver": scenario.get("driver", "unknown"),
                    "tire_compound": scenario.get("tires", {}).get("compound", "unknown"),
                    "tire_age": scenario.get("tires", {}).get("age_laps", 0),
                }
            ],
        )

    def ingest_from_file(self, filepath: Path) -> int:
        """
        Ingest all scenarios from a golden dataset file.
        Args:
            filepath: Path to JSON file
        Returns:
            Number of scenarios ingested
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        scenarios = data.get("scenarios", [])
        track_name = data.get("race", {}).get("track", filepath.stem)

        print(f"  Ingesting {len(scenarios)} scenarios from {track_name}...")

        for scenario in scenarios:
            scenario_id = scenario.get("id")
            if not scenario_id:
                print(f"    ⚠️  Skipping scenario without ID")
                continue

            # Add race info to scenario for context
            scenario["race"] = data.get("race", {})

            self.ingest_scenario(scenario, scenario_id)

        return len(scenarios)

    def ingest_from_directory(self, directory: Path) -> int:
        """
        Ingest all scenarios from all JSON files in a directory.
        Args:
            directory: Path to directory with golden dataset files
        Returns:
            Total number of scenarios ingested
        """
        directory = Path(directory)
        json_files = sorted(directory.glob("*.json"))

        if not json_files:
            print(f"⚠️  No JSON files found in {directory}")
            return 0

        print(f"Found {len(json_files)} dataset files")

        total_scenarios = 0
        for json_file in json_files:
            count = self.ingest_from_file(json_file)
            total_scenarios += count

        print(f"✓ Total scenarios ingested: {total_scenarios}")
        print(f"  Database now contains: {self.collection.count()} scenarios")

        return total_scenarios

    def search(
        self, query_scenario: Dict, k: int = 5, filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar scenarios.
        Args:
            query_scenario: Scenario to search for
            k: Number of results to return
            filter_metadata: Optional filters (e.g., {"track": "Silverstone"})
        Returns:
            List of similar scenarios with similarity scores
        """
        # Generate query embedding
        query_embedding = self.embedder.embed(query_scenario)

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=k, where=filter_metadata
        )

        # Parse results
        similar_scenarios = []
        for i in range(len(results["ids"][0])):
            scenario_id = results["ids"][0][i]
            distance = results["distances"][0][i]
            document = results["documents"][0][i]
            metadata = results["metadatas"][0][i]

            # Convert distance to similarity score (cosine similarity)
            # ChromaDB returns L2 distance, convert to similarity
            similarity = 1 - (distance / 2)  # Approximate conversion

            similar_scenarios.append(
                {
                    "id": scenario_id,
                    "similarity": similarity,
                    "scenario": json.loads(document),
                    "metadata": metadata,
                }
            )

        return similar_scenarios

    def get_by_id(self, scenario_id: str) -> Optional[Dict]:
        """
        Retrieve a specific scenario by ID.
        Args:
            scenario_id: Scenario ID
        Returns:
            Scenario dict or None if not found
        """
        results = self.collection.get(ids=[scenario_id])

        if not results["ids"]:
            return None

        return json.loads(results["documents"][0])

    def clear(self) -> None:
        """Clear all scenarios from the database."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name, metadata={"description": "F1 racing strategy scenarios"}
        )
        print("✓ Database cleared")

    def stats(self) -> Dict:
        """
        Get database statistics.
        Returns:
            Dict with stats (count, tracks, etc.)
        """
        count = self.collection.count()

        # Get all metadata to analyze
        if count > 0:
            all_data = self.collection.get()
            tracks = set(m.get("track") for m in all_data["metadatas"])

            return {
                "total_scenarios": count,
                "tracks": sorted(tracks),
                "embedding_dim": self.embedder.embedding_dim,
            }

        return {"total_scenarios": 0, "tracks": [], "embedding_dim": self.embedder.embedding_dim}
