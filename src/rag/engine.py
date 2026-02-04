"""
F1 Strategy RAG Engine
Combines vector retrieval with LLM reasoning using Ollama.
"""

import json
import os
from typing import Dict, List, Optional

import requests

from src.rag.vectordb import VectorDatabase


class F1StrategyRAG:
    """
    RAG-based F1 strategy recommendation engine.
    Combines:
    - Vector DB retrieval (similar historical scenarios)
    - Local LLM reasoning (Ollama for decision-making)
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
        print("Initializing F1 Strategy RAG Engine (Ollama)...")

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

        print(f"✓ RAG Engine ready")
        print(f"  Model: {model}")
        print(f"  Scenarios indexed: {self.vectordb.collection.count()}")

    def build_prompt(self, scenario: Dict, retrieved_scenarios: List[Dict]) -> str:
        """
        Build prompt for LLM with retrieved context.
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

        # Build context from retrieved scenarios
        context_sections = []
        for i, retrieved in enumerate(retrieved_scenarios[:5], 1):
            ret_scenario = retrieved["scenario"]
            similarity = retrieved["similarity"]

            ret_decision = ret_scenario.get("golden_truth", {}).get("decision", "UNKNOWN")
            ret_reasoning = ret_scenario.get("golden_truth", {}).get("reasoning", "")

            context_sections.append(
                f"""
                Historical Scenario {i} (Similarity: {similarity:.2f}):
                - Race: {ret_scenario.get('race', {}).get('track', 'Unknown')} {ret_scenario.get('race', {}).get('year', '')}
                - Lap {ret_scenario.get('lap')}, Position {ret_scenario.get('position')}
                - Tires: {ret_scenario.get('tires', {}).get('compound')} ({ret_scenario.get('tires', {}).get('age_laps')} laps old)
                - Decision: {ret_decision}
                - Reasoning: {ret_reasoning[:150]}
                """
            )
        context = "\n".join(context_sections)

        # Build full prompt
        prompt = f"""
                You are an expert F1 race strategist. Analyze the current racing situation and recommend whether the driver should BOX (pit stop) or STAY_OUT (remain on track).
                CURRENT SITUATION:
                Lap: {lap}
                Driver: {driver}
                Position: P{position}
                Tires: {compound} compound, {age} laps old, condition: {condition}
                Weather: {weather_condition}
                Race State: {race_state}
                Gaps: {gaps}

                SIMILAR HISTORICAL SCENARIOS:
                {context}

                Based on the current situation and similar historical scenarios, provide your recommendation.

                Respond ONLY with valid JSON in this exact format (no markdown, no code blocks):
                {{"decision": "BOX", "confidence": 0.85, "reasoning": "Your explanation here", "risk_level": "MEDIUM"}}

                Rules:
                - decision: Must be exactly "BOX" or "STAY_OUT"
                - confidence: Number between 0.0 and 1.0
                - reasoning: 2-3 sentences explaining your decision
                - risk_level: Must be "LOW", "MEDIUM", or "HIGH"
                - Consider tire degradation, gaps to competitors, and race phase
                - Safety Car/VSC periods offer advantageous pit windows

                Respond with ONLY the JSON object, nothing else.
                """

        return prompt

    def query_ollama(self, prompt: str, temperature: float = 0.3) -> str:
        """
        Query Ollama API.
        Args:
            prompt: Prompt to send
            temperature: Sampling temperature
        Returns:
            Model response text
        """
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
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

    def generate_strategy(self, scenario: Dict, k: int = 5, temperature: float = 0.3) -> Dict:
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

        # Step 2: Build prompt with context
        prompt = self.build_prompt(scenario, retrieved)

        # Step 3: Query Ollama
        response_text = self.query_ollama(prompt, temperature)

        # Step 4: Parse response
        parsed = self.parse_llm_response(response_text)

        # Step 5: Add retrieved scenarios to response
        parsed["retrieved_scenarios"] = retrieved
        parsed["num_retrieved"] = len(retrieved)

        return parsed
