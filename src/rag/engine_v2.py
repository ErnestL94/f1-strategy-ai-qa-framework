"""
F1 Strategy RAG Engine - Version 2 (Improved)
Combines vector retrieval with LLM reasoning using Ollama.

Improvements over v1:
- Race phase awareness (final laps detection)
- Explicit retrieved scenario analysis
- Critical rules section in prompt
- Lower temperature (0.1 vs 0.3)
- Better context weighting
"""

import json
import os
from typing import Dict, List, Optional

import requests

from src.rag.vectordb import VectorDatabase


class F1StrategyRAG:
    """
    RAG-based F1 strategy recommendation engine - Version 2.

    Combines:
    - Vector DB retrieval (similar historical scenarios)
    - Local LLM reasoning (Ollama for decision-making)
    - Improved prompt engineering
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        ollama_url: str = "http://localhost:11434",
        vectordb_path: str = "data/vectordb",
    ):
        """
        Initialize RAG engine with Ollama.

        Args:
            model: Ollama model to use (e.g., "llama3.2:3b", "mistral:7b")
            ollama_url: Ollama API endpoint
            vectordb_path: Path to vector database
        """
        print("Initializing F1 Strategy RAG Engine v2 (Ollama)...")

        # Initialize vector database
        print("  Loading vector database...")
        self.vectordb = VectorDatabase(persist_directory=vectordb_path)

        # Configure Ollama
        print(f"  Connecting to Ollama ({ollama_url})...")
        self.ollama_url = ollama_url
        self.model = model

        # Verify Ollama is running and model is available
        try:
            response = requests.get(f"{ollama_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]

            if model not in model_names:
                print(f"  ⚠️  Model '{model}' not found. Available models:")
                for m in model_names:
                    print(f"      - {m}")
                print(f"  Run: ollama pull {model}")
                raise ValueError(f"Model {model} not available in Ollama")

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {ollama_url}. "
                "Make sure Ollama is running: ollama serve"
            )

        print(f"✓ RAG Engine v2 ready")
        print(f"  Model: {model}")
        print(f"  Scenarios indexed: {self.vectordb.collection.count()}")

    def estimate_race_length(self, lap: int, track: str = None) -> int:
        """
        Estimate race length based on lap and track.

        Args:
            lap: Current lap
            track: Track name (optional)

        Returns:
            Estimated total race laps
        """
        # Track-specific lengths (if we know them)
        track_lengths = {
            "Monaco": 78,
            "Marina Bay Street Circuit": 62,
            "Marina Bay": 62,
            "Singapore": 62,
            "Spa": 44,
            "Silverstone": 52,
            "Monza": 53,
        }

        if track and track in track_lengths:
            return track_lengths[track]

        # Default estimation: most F1 races are 50-70 laps
        # If we're at lap 60+, assume ~62-70 lap race
        if lap >= 60:
            return 62
        elif lap >= 50:
            return 60
        else:
            return 55  # Conservative estimate

    def get_race_phase(self, lap: int, race_length: int) -> str:
        """
        Determine race phase.

        Args:
            lap: Current lap
            race_length: Total race laps

        Returns:
            Race phase string
        """
        laps_remaining = race_length - lap

        if lap <= 1:
            return "START"
        elif laps_remaining <= 3:
            return "FINAL_LAPS"
        elif laps_remaining <= 10:
            return "LATE_RACE"
        elif lap <= 15:
            return "EARLY_RACE"
        else:
            return "MID_RACE"

    def build_prompt(self, scenario: Dict, retrieved_scenarios: List[Dict]) -> str:
        """
        Build improved prompt for LLM with retrieved context.

        Args:
            scenario: Current racing scenario
            retrieved_scenarios: Similar historical scenarios

        Returns:
            Formatted prompt string
        """
        # Extract scenario details
        lap = scenario.get("lap", 0)
        driver = scenario.get("driver", "Unknown")
        position = scenario.get("position", 0)
        if isinstance(position, dict):
            position = position.get("current", 0)

        tires = scenario.get("tires", {})
        compound = tires.get("compound", "UNKNOWN")
        age = tires.get("age_laps", 0)
        condition = tires.get("condition", "unknown")

        weather = scenario.get("weather", {})
        weather_condition = weather.get("condition", "dry")

        gaps = scenario.get("gaps", {})
        race_state = scenario.get("race_state", "racing")

        # Estimate race phase
        track = scenario.get("race", {}).get("track", None)
        race_length = self.estimate_race_length(lap, track)
        race_phase = self.get_race_phase(lap, race_length)
        laps_remaining = race_length - lap

        # Analyze retrieved scenarios
        retrieved_decisions = []
        for ret in retrieved_scenarios:
            ret_scenario = ret["scenario"]
            ret_decision = ret_scenario.get("golden_truth", {}).get("decision", "UNKNOWN")
            retrieved_decisions.append(ret_decision)

        box_count = retrieved_decisions.count("BOX")
        stay_count = retrieved_decisions.count("STAY_OUT")

        # Build context from retrieved scenarios
        context_sections = []
        for i, retrieved in enumerate(retrieved_scenarios[:5], 1):
            ret_scenario = retrieved["scenario"]
            similarity = retrieved["similarity"]

            ret_decision = ret_scenario.get("golden_truth", {}).get("decision", "UNKNOWN")
            ret_reasoning = ret_scenario.get("golden_truth", {}).get("reasoning", "")

            context_sections.append(f"""Historical Scenario {i} (Similarity: {similarity:.2f}):
- Race: {ret_scenario.get('race', {}).get('track', 'Unknown')} {ret_scenario.get('race', {}).get('year', '')}
- Lap {ret_scenario.get('lap')}, Position {ret_scenario.get('position')}
- Tires: {ret_scenario.get('tires', {}).get('compound')} ({ret_scenario.get('tires', {}).get('age_laps')} laps old)
- Decision: {ret_decision}
- Reasoning: {ret_reasoning[:150] if ret_reasoning else 'N/A'}""")

        context = "\n\n".join(context_sections)

        # Calculate average similarity
        avg_similarity = sum(r["similarity"] for r in retrieved_scenarios) / len(
            retrieved_scenarios
        )

        # Build full prompt
        prompt = f"""You are an expert F1 race strategist. Analyze the current racing situation and recommend whether the driver should BOX (pit stop) or STAY_OUT (remain on track).

CURRENT SITUATION:
Lap: {lap} of ~{race_length} (estimated)
Race Phase: {race_phase}
Laps Remaining: ~{laps_remaining}
Driver: {driver}
Position: P{position}
Tires: {compound} compound, {age} laps old, condition: {condition}
Weather: {weather_condition}
Race State: {race_state}
Gaps: {gaps}

SIMILAR HISTORICAL SCENARIOS:
{context}

ANALYSIS OF RETRIEVED SCENARIOS:
- Total scenarios analyzed: {len(retrieved_scenarios)}
- Average similarity: {avg_similarity:.2f}
- Recommendations: {stay_count} × STAY_OUT, {box_count} × BOX
- Pattern: {"STRONG consensus for " + ("STAY_OUT" if stay_count >= 4 else "BOX" if box_count >= 4 else "MIXED") if len(retrieved_scenarios) >= 4 else "Insufficient pattern"}

================================================================================
DECISION TREE - FOLLOW THIS EXACTLY:
================================================================================

YOUR CURRENT VALUES:
- laps_remaining = {laps_remaining}
- race_state = "{race_state}"
- tire_age = {age} laps
- retrieved consensus = {stay_count} STAY_OUT, {box_count} BOX

STEP 1: Is laps_remaining ≤ 3?
  → YES: Output STAY_OUT immediately (RULE 1: Final laps).
     Reasoning: "Applying RULE 1: Race ends in {laps_remaining} laps. Cannot recover from pit stop."
     Stop here.
  → NO: Go to STEP 2

STEP 2: Does race_state contain "safety" OR "vsc" OR "virtual"?
  Check: Does the string "{race_state}" contain any of these words?
  → YES: Is tire_age >= 10 laps?
    → YES: Output BOX immediately (RULE 2: Safety Car free pit).
       Reasoning: "Applying RULE 2: {race_state} active. {age}-lap tires. Free pit window."
       Stop here.
    → NO: Go to STEP 3
  → NO: Go to STEP 3

STEP 3: Is tire_age ≤ 5 laps?
  → YES: Output STAY_OUT immediately (RULE 3: Fresh tires).
     Reasoning: "Applying RULE 3: Tires only {age} laps old, plenty of life."
     Stop here.
  → NO: Go to STEP 4

STEP 4: Do 4+ retrieved scenarios agree (≥80% consensus)?
  Check: Is {stay_count} >= 4 OR {box_count} >= 4?
  → YES: Follow the consensus (RULE 4: Historical pattern).
     If {stay_count} >= 4: Output STAY_OUT
     If {box_count} >= 4: Output BOX
     Reasoning: "Applying RULE 4: {stay_count or box_count} of 5 similar scenarios agree."
     Stop here.
  → NO: Go to STEP 5

STEP 5: Use full strategic analysis (RULE 5)
  Consider:
  - Tire degradation vs laps remaining
  - Gaps to competitors
  - Retrieved scenarios (even if <80% consensus)
  - Race context and strategy
  Make your expert judgment and explain reasoning clearly.

CRITICAL: Your reasoning MUST start with "Applying RULE X: ..." to show which rule you followed.

Respond ONLY with valid JSON in this exact format (no markdown, no code blocks):
{{"decision": "BOX", "confidence": 0.85, "reasoning": "Applying RULE 1: Race ends in 2 laps. Cannot recover from pit stop. Staying out.", "risk_level": "LOW"}}

Rules for JSON response:
- decision: Must be exactly "BOX" or "STAY_OUT"
- confidence: Number between 0.0 and 1.0
- reasoning: Must start with "Applying RULE X: ..." then 1-2 sentences
- risk_level: Must be "LOW", "MEDIUM", or "HIGH"

Respond with ONLY the JSON object, nothing else."""

        return prompt

    def query_ollama(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Query Ollama API.

        Args:
            prompt: Prompt to send
            temperature: Sampling temperature (lower = more deterministic)

        Returns:
            Model response text
        """
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "seed": 42,
                "stream": False,
            },
        )
        response.raise_for_status()

        return response.json()["response"]

    def parse_llm_response(self, response_text: str) -> Dict:
        """
        Parse LLM response into structured format.

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed response dict
        """
        import re

        # Clean the response
        response_text = response_text.strip()

        # Try to extract JSON from response
        # LLM might wrap it in markdown code blocks or add extra text
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError(f"Could not extract JSON from response: {response_text[:200]}")

        # Parse JSON
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}\nResponse: {json_str[:200]}")

        # Validate required fields
        required_fields = ["decision", "confidence", "reasoning", "risk_level"]
        for field in required_fields:
            if field not in parsed:
                raise ValueError(f"Missing required field: {field}")

        # Validate decision
        if parsed["decision"] not in ["BOX", "STAY_OUT"]:
            raise ValueError(f"Invalid decision: {parsed['decision']}")

        # Validate confidence
        if not (0.0 <= parsed["confidence"] <= 1.0):
            raise ValueError(f"Invalid confidence: {parsed['confidence']}")

        # Validate risk level
        if parsed["risk_level"] not in ["LOW", "MEDIUM", "HIGH"]:
            raise ValueError(f"Invalid risk level: {parsed['risk_level']}")

        return parsed

    def generate_strategy(self, scenario: Dict, k: int = 5, temperature: float = 0.1) -> Dict:
        """
        Generate strategy recommendation using RAG.

        Args:
            scenario: Current racing scenario
            k: Number of similar scenarios to retrieve
            temperature: LLM temperature (lower = more deterministic)

        Returns:
            Strategy recommendation with decision, confidence, reasoning
        """
        # Step 1: Retrieve similar scenarios
        retrieved = self.vectordb.search(scenario, k=k)

        if not retrieved:
            raise ValueError("No similar scenarios found in database")

        # Step 2: Build improved prompt with context
        prompt = self.build_prompt(scenario, retrieved)

        # Step 3: Query Ollama
        response_text = self.query_ollama(prompt, temperature)

        # Step 4: Parse response
        parsed = self.parse_llm_response(response_text)

        # Step 5: Add retrieved scenarios to response
        parsed["retrieved_scenarios"] = retrieved
        parsed["num_retrieved"] = len(retrieved)

        return parsed
