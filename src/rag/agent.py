"""
F1 Strategy Agent - Core decision engine.
"""
from src.validators.input_validator import InputValidator, ScenarioValidationError


class F1StrategyAgent:
    """
    Generates race strategy recommendations.
    Version 0.3.2: Added SC, VSC, final laps logic.
    """
    
    def __init__(self):
        """Initialize the strategy agent"""
        self.version = "v0.3.2-sc-vsc-final-laps"
        self.validator = InputValidator()
    
    def generate_strategy(self, scenario: dict) -> dict:
        """
        Generate strategy recommendation for a given race scenario.
        Args:
            scenario: Race scenario dict with lap, driver, tires, weather, etc.
        Returns:
            Strategy recommendation with decision, rationale, confidence, risk
        Raises:
            ScenarioValidationError: If scenario data is invalid
        """
        # STEP 1: Validate inputs
        is_valid, errors = self.validator.validate_scenario(scenario)
        if not is_valid:
            raise ScenarioValidationError(
                f"Invalid scenario data:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        
        # Extract scenario data
        lap = scenario["lap"]
        tires = scenario["tires"]
        tire_age = tires["age_laps"]
        tire_compound = tires["compound"]
        weather = scenario["weather"]
        weather_condition = weather.get("condition", "unknown")
        
        # STEP 2: Check weather/tire compatibility
        is_compatible, warning = self.validator.validate_weather_tire_compatibility(
            tire_compound, weather_condition
        )
        
        if not is_compatible:
            # Dangerous situation - override normal logic
            return self._handle_dangerous_situation(
                tire_compound, weather_condition, warning, scenario
            )
        
        # STEP 3: Normal decision logic
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
    
    def _handle_dangerous_situation(
        self,
        current_compound: str,
        weather_condition: str,
        warning: str,
        scenario: dict
    ) -> dict:
        """
        Handle dangerous weather/tire mismatches.
        These situations override normal strategy.
        """
        # Slicks in wet = Must pit for inters/wets
        if current_compound in ["SOFT", "MEDIUM", "HARD"]:
            if "heavy" in weather_condition.lower():
                recommended_compound = "WET"
            else:
                recommended_compound = "INTERMEDIATE"
            
            return {
                "decision": "BOX",
                "compound": recommended_compound,
                "rationale": f"{warning} Immediate tire change required for safety.",
                "confidence_score": 0.98,
                "risk_level": "HIGH",
                "model_version": self.version,
                "warning": warning
            }
        
        # Wets/inters on dry = Must pit for slicks
        if current_compound in ["INTERMEDIATE", "WET"]:
            return {
                "decision": "BOX",
                "compound": "MEDIUM",  # Safe default for dry
                "rationale": f"{warning} Tire overheating risk. Change to slicks.",
                "confidence_score": 0.95,
                "risk_level": "MEDIUM",
                "model_version": self.version,
                "warning": warning
            }
        
        # Fallback (shouldn't reach here)
        return {
            "decision": "BOX",
            "compound": "MEDIUM",
            "rationale": warning,
            "confidence_score": 0.50,
            "risk_level": "HIGH",
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
        # RULE -1: Final laps protection (within 3 laps of finish)
        # Must come FIRST - never pit in final laps unless emergency
        # Estimate race end: if lap > 58 (typical F1 race is 50-70 laps)
        if lap >= 58:  # Heuristic for "final laps" without knowing total_laps
            # Exception: Emergency weather/tire situations handled by other rules
            # For now, just don't pit for normal degradation
            if tire_age < 30 or tire_age >= 30:  # Covers all cases
                return (
                    "STAY_OUT",
                    f"Final laps of race (L{lap}). Position priority over tire condition. No time to benefit from pit stop.",
                    0.85,
                    "MEDIUM"
                )
        
        # RULE 0A: Safety Car opportunity
        race_state = scenario.get("race_state", "").lower()
        if "safety" in race_state or "sc" in race_state:
            # Safety Car or VSC - exploit for "free" pit stop
            if tire_age >= 15:  # Tires past early stint
                return (
                    "BOX",
                    f"Safety Car/VSC opportunity. {tire_compound} at {tire_age} laps ready for change. Pit time loss negated.",
                    0.90,
                    "LOW"
                )
        
        # RULE 0B: Free pit stop opportunity (late race, huge gaps)
        if "gaps" in scenario and lap > 40 and lap < 58:
            gaps = scenario["gaps"]
            
            # Try multiple gap keys (to_p1, to_p2, to_p3, etc.)
            gap_ahead = (
                gaps.get("to_p1") or 
                gaps.get("to_p2") or 
                gaps.get("to_p3") or 
                0
            )
            gap_behind = (
                gaps.get("to_p4") or 
                gaps.get("to_p5") or 
                gaps.get("to_p6") or 
                0
            )
            
            # Position secured with huge gaps (>14s is safe for F1 pit stop)
            if gap_ahead > 14 and gap_behind > 14:
                return (
                    "BOX",
                    f"Free pit stop opportunity. P{scenario.get('position', 'X')} secured with {gap_ahead:.1f}s ahead, {gap_behind:.1f}s behind. Attack fastest lap point with fresh tires.",
                    0.90,
                    "LOW"
                )
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