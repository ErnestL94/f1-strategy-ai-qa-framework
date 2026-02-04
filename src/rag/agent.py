"""
F1 Strategy Agent - Core decision engine
"""

class F1StrategyAgent:
    """
    Generates race strategy recommendations.
    Version 0.2.0: Rule-based decision logic
    """
    
    def __init__(self):
        """Initialize the strategy agent"""
        self.version = "v0.2.0-rules"
    
    def generate_strategy(self, scenario: dict) -> dict:
        """
        Generate strategy recommendation for a given race scenario.
        Args:
            scenario: Race scenario dict with lap, driver, tires, weather, etc.
        Returns:
            Strategy recommendation with decision, rationale, confidence, risk
        """
        # Extract scenario data
        lap = scenario["lap"]
        tires = scenario["tires"]
        tire_age = tires["age_laps"]
        tire_compound = tires["compound"]
        weather = scenario["weather"]
        
        # Decision logic
        decision, rationale, confidence, risk = self._make_decision(
            lap, tire_age, tire_compound, weather, scenario
        )
        
        return {
            "decision": decision,
            "compound": tire_compound,
            "rationale": rationale,
            "confidence_score": confidence,
            "risk_level": risk,
            "model_version": self.version
        }
    
    def _make_decision(
        self, 
        lap: int, 
        tire_age: int, 
        tire_compound: str, 
        weather: dict,
        scenario: dict
    ) -> tuple:
        """
        Core decision logic based on rules.
        Returns:
            (decision, rationale, confidence, risk)
        """
        
        # Rule 1: Fresh tires at start = STAY OUT
        if tire_age == 0:
            return (
                "STAY_OUT",
                f"Fresh {tire_compound} tires optimal for opening stint. Track position critical at race start.",
                0.90,
                "LOW"
            )
        
        # Rule 2: Very old tires (>30 laps) = BOX
        if tire_age >= 30:
            return (
                "BOX",
                f"{tire_compound} tires at {tire_age} laps - past optimal window. Degradation critical.",
                0.85,
                "MEDIUM"
            )
        
        # Rule 3: Weather change = BOX (if dry to wet or wet to dry)
        if self._is_weather_changing(weather):
            return (
                "BOX",
                f"Weather changing to {weather['condition']}. Tire compound change required.",
                0.80,
                "MEDIUM"
            )
        
        # Rule 4: Mid-stint (10-25 laps) = STAY OUT (default strategy)
        if 10 <= tire_age <= 25:
            return (
                "STAY_OUT",
                f"{tire_compound} tires at {tire_age} laps - within optimal window. Maintain strategy.",
                0.75,
                "LOW"
            )
        
        # Rule 5: Approaching end of window (26-29 laps) = Contextual
        if 26 <= tire_age <= 29:
            # Check if there's a safe gap to pit
            if "gaps" in scenario and scenario["gaps"].get("to_p4", 0) > 10:
                return (
                    "BOX",
                    f"{tire_compound} approaching end of optimal window. Safe gap allows pit stop.",
                    0.70,
                    "MEDIUM"
                )
            else:
                return (
                    "STAY_OUT",
                    f"{tire_compound} near end of window but no safe gap to pit. Extend stint.",
                    0.65,
                    "MEDIUM"
                )
        
        # Default: STAY OUT with medium confidence
        return (
            "STAY_OUT",
            f"Standard strategy for {tire_compound} at {tire_age} laps.",
            0.70,
            "MEDIUM"
        )
    
    def _is_weather_changing(self, weather: dict) -> bool:
        """Detect if weather is changing"""
        condition = weather.get("condition", "").lower()
        
        # Weather keywords that suggest change
        changing_keywords = ["changing", "mixed", "drizzle", "damp"]
        
        return any(keyword in condition for keyword in changing_keywords)