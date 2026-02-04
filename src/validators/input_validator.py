"""
Input validation for race scenarios.
Prevents hallucinations and catches impossible data.
"""
from typing import Dict, List, Tuple


class ScenarioValidationError(ValueError):
    """Raised when scenario data is invalid"""
    pass


class InputValidator:
    """
    Validates race scenario inputs before processing.
    Prevents the agent from making decisions on impossible/dangerous data.
    """
    
    # FIA regulation: Valid tire compounds (2023+)
    VALID_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
    
    # Reasonable bounds
    MAX_TIRE_AGE = 100  # No tire lasts 100 laps
    MAX_RACE_LAPS = 80  # Longest F1 races are ~70 laps
    
    @staticmethod
    def validate_scenario(scenario: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a race scenario.
        Args:
            scenario: Race scenario dict
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate tire compound
        if "tires" in scenario:
            compound = scenario["tires"].get("compound")
            if compound and compound not in InputValidator.VALID_COMPOUNDS:
                errors.append(
                    f"Invalid tire compound: '{compound}'. "
                    f"Valid compounds: {InputValidator.VALID_COMPOUNDS}"
                )
            
            # Validate tire age
            tire_age = scenario["tires"].get("age_laps")
            if tire_age is not None:
                if tire_age < 0:
                    errors.append(f"Tire age cannot be negative: {tire_age}")
                elif tire_age > InputValidator.MAX_TIRE_AGE:
                    errors.append(
                        f"Tire age ({tire_age} laps) exceeds maximum realistic age "
                        f"({InputValidator.MAX_TIRE_AGE} laps)"
                    )
        
        # Validate lap number
        if "lap" in scenario:
            lap = scenario["lap"]
            if lap < 1:
                errors.append(f"Lap number must be positive: {lap}")
            elif lap > InputValidator.MAX_RACE_LAPS:
                errors.append(
                    f"Lap {lap} exceeds maximum race distance "
                    f"({InputValidator.MAX_RACE_LAPS})"
                )
        
        return (len(errors) == 0, errors)
    
    @staticmethod
    def validate_weather_tire_compatibility(
        tire_compound: str,
        weather_condition: str
    ) -> Tuple[bool, str]:
        """
        Check if tire compound is appropriate for weather.
        Returns:
            (is_compatible, warning_message)
        """
        condition_lower = weather_condition.lower()
        
        # Slicks in wet conditions
        if tire_compound in ["SOFT", "MEDIUM", "HARD"]:
            dangerous_conditions = ["heavy_rain", "rain", "wet"]
            if any(cond in condition_lower for cond in dangerous_conditions):
                return (
                    False,
                    f"DANGER: Slick tires ({tire_compound}) in {weather_condition} "
                    f"conditions. Must change to INTERMEDIATE or WET."
                )
        
        # Wet tires in dry conditions
        if tire_compound in ["INTERMEDIATE", "WET"]:
            if "dry" in condition_lower:
                severity = "CRITICAL" if tire_compound == "WET" else "WARNING"
                return (
                    False,
                    f"{severity}: {tire_compound} tires will overheat on dry track. "
                    f"Must change to slick compound."
                )
        
        return (True, "")