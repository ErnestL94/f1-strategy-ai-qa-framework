"""
Embedding utilities for F1 scenarios.
Converts scenarios to vectors for semantic search.

Uses hybrid approach: text embeddings + numerical features
"""

from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


class ScenarioEmbedder:
    """
    Converts F1 racing scenarios into embedding vectors.

    Uses hybrid approach:
    - Sentence-transformers for semantic understanding (384 dims)
    - Numerical features for precise values (11 dims)
    - Total: 395-dimensional embeddings
    """

    # Feature normalization constants
    MAX_TIRE_AGE = 50  # laps
    MAX_LAP = 70  # typical F1 race
    MAX_POSITION = 20  # grid size
    MAX_GAP = 60  # seconds

    # Tire compound encoding
    COMPOUND_ENCODING = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}

    # Weather encoding
    WEATHER_ENCODING = {
        "dry": 0,
        "damp": 1,
        "drizzle": 2,
        "light_rain": 3,
        "rain": 4,
        "heavy_rain": 5,
        "wet": 6,
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedder with sentence-transformer model.
        Args:
            model_name: HuggingFace model name (default: fast, good quality)
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.text_dim = self.model.get_sentence_embedding_dimension()
        self.numerical_dim = 11  # Number of numerical features
        self.embedding_dim = self.text_dim + self.numerical_dim
        print(f"✓ Model loaded")
        print(f"  Text embedding: {self.text_dim} dims")
        print(f"  Numerical features: {self.numerical_dim} dims")
        print(f"  Total: {self.embedding_dim} dims")

    def scenario_to_text(self, scenario: Dict) -> str:
        """
        Convert scenario dict to natural language text for semantic embedding.
        Args:
            scenario: Race scenario dict
        Returns:
            Natural language description
        """
        tires = scenario.get("tires", {})
        compound = tires.get("compound", "UNKNOWN")
        age = tires.get("age_laps", 0)
        condition = tires.get("condition", "")

        lap = scenario.get("lap", 0)
        position = scenario.get("position", 0)

        weather = scenario.get("weather", {})
        weather_condition = weather.get("condition", "unknown")

        # Build semantic description
        text = f"""
        Race situation at lap {lap}.
        Currently in position {position}.
        Running on {compound} compound tires that are {age} laps old.
        Tire condition: {condition}.
        Track conditions: {weather_condition} weather.
        """

        # Add contextual information
        if "context" in scenario:
            context = scenario["context"]
            if "description" in context:
                text += f"\n{context['description']}"

        # Add race state if available
        race_state = scenario.get("race_state", "")
        if race_state:
            text += f"\nRace state: {race_state}"

        return text.strip()

    def extract_numerical_features(self, scenario: Dict) -> np.ndarray:
        """
        Extract normalized numerical features from scenario.

        Features (11 total):
        0. Tire age (normalized 0-1)
        1. Tire compound (categorical 0-4)
        2. Lap number (normalized 0-1)
        3. Position (normalized 0-1)
        4. Gap ahead (normalized 0-1, 0 if unknown)
        5. Gap behind (normalized 0-1, 0 if unknown)
        6. Weather severity (categorical 0-6)
        7. Race phase (early=0, mid=0.5, late=1)
        8. Is Safety Car (binary 0-1)
        9. Is Virtual Safety Car (binary 0-1)
        10. Tire condition severity (normalized 0-1)

        Args:
            scenario: Race scenario dict

        Returns:
            Numpy array of 11 normalized features
        """
        features = []

        # Feature 0: Tire age (normalized)
        tires = scenario.get("tires", {})
        tire_age = tires.get("age_laps", 0)
        features.append(min(tire_age / self.MAX_TIRE_AGE, 1.0))

        # Feature 1: Tire compound (categorical)
        compound = tires.get("compound", "MEDIUM")
        features.append(self.COMPOUND_ENCODING.get(compound, 1) / 4.0)

        # Feature 2: Lap number (normalized)
        lap = scenario.get("lap", 0)
        features.append(min(lap / self.MAX_LAP, 1.0))

        # Feature 3: Position (normalized) - HANDLE BOTH INT AND DICT
        position_raw = scenario.get("position", 10)
        if isinstance(position_raw, dict):
            # Position is a dict like {"current": 3, "start": 2}
            position = position_raw.get("current", 10)
        else:
            # Position is an int
            position = position_raw
        features.append(min(position / self.MAX_POSITION, 1.0))

        # Features 4-5: Gaps (normalized)
        gaps = scenario.get("gaps", {})
        gap_ahead = gaps.get("to_p2", 0) or gaps.get("to_p1", 0) or gaps.get("to_p3", 0) or 0
        gap_behind = gaps.get("to_p4", 0) or gaps.get("to_p5", 0) or 0
        features.append(min(gap_ahead / self.MAX_GAP, 1.0))
        features.append(min(gap_behind / self.MAX_GAP, 1.0))

        # Feature 6: Weather severity
        weather = scenario.get("weather", {})
        condition = weather.get("condition", "dry").lower()
        weather_code = self.WEATHER_ENCODING.get(condition, 0)
        features.append(weather_code / 6.0)

        # Feature 7: Race phase
        race_phase = 0.0
        if lap > 0:
            if lap < 15:
                race_phase = 0.0  # Early
            elif lap < 50:
                race_phase = 0.5  # Mid
            else:
                race_phase = 1.0  # Late
        features.append(race_phase)

        # Features 8-9: Safety Car detection (order matters!)
        race_state = scenario.get("race_state", "").lower()

        # Detect both first
        is_vsc = 0.0
        is_sc = 0.0

        # Check for Virtual Safety Car
        if race_state in ["virtual_safety_car", "vsc"]:
            is_vsc = 1.0
        elif "virtual" in race_state and "safety" in race_state:
            is_vsc = 1.0

        # Check for Full Safety Car (but NOT if it's a VSC)
        if is_vsc == 0.0:  # Only if NOT VSC
            if race_state in ["safety_car", "sc"]:
                is_sc = 1.0
            elif "safety" in race_state and "car" in race_state and "virtual" not in race_state:
                is_sc = 1.0

        # Feature 8: Full SC
        features.append(is_sc)

        # Feature 9: VSC
        features.append(is_vsc)

        # Feature 10: Tire condition severity
        tire_condition = tires.get("condition", "optimal").lower()
        condition_map = {
            "optimal": 0.0,
            "good": 0.2,
            "mid_stint": 0.4,
            "end_of_optimal_window": 0.6,
            "high_wear": 0.8,
            "critical_wear": 1.0,
        }
        features.append(condition_map.get(tire_condition, 0.4))

        return np.array(features, dtype=np.float32)

    def embed(self, scenario: Dict) -> np.ndarray:
        """
        Create hybrid embedding vector for scenario.
        Args:
            scenario: Race scenario dict
        Returns:
            Hybrid embedding vector (text + numerical features)
        """
        # Text embedding (semantic understanding)
        text = self.scenario_to_text(scenario)
        text_embedding = self.model.encode(text, convert_to_numpy=True)

        # Numerical features (precise values)
        numerical_features = self.extract_numerical_features(scenario)

        # Concatenate into single embedding
        hybrid_embedding = np.concatenate([text_embedding, numerical_features])

        return hybrid_embedding

    def embed_batch(self, scenarios: List[Dict]) -> np.ndarray:
        """
        Embed multiple scenarios efficiently.
        Args:
            scenarios: List of scenario dicts
        Returns:
            Array of embeddings (num_scenarios x embedding_dim)
        """
        # Batch text embeddings (efficient)
        texts = [self.scenario_to_text(s) for s in scenarios]
        text_embeddings = self.model.encode(texts, convert_to_numpy=True)

        # Extract numerical features for each
        numerical_features_list = [self.extract_numerical_features(s) for s in scenarios]
        numerical_features = np.array(numerical_features_list)

        # Concatenate
        hybrid_embeddings = np.concatenate([text_embeddings, numerical_features], axis=1)

        return hybrid_embeddings
