"""
JSON Schema validation for golden datasets.
Ensures consistency across all test data.
"""

from typing import Dict, List

GOLDEN_DATASET_SCHEMA = {
    "required_fields": {
        "race": ["name", "track", "date", "total_laps", "regulation_era"],
        "scenario": [
            "id",
            "name",
            "lap",
            "driver",
            "position",
            "tires",
            "weather",
            "context",
            "golden_truth",
            "test_criteria",
        ],
        "golden_truth": ["decision", "rationale", "confidence_level", "risk_level"],
        "tires": ["compound", "age_laps"],
    },
    "valid_values": {
        "decision": ["BOX", "STAY_OUT"],
        "confidence_level": ["LOW", "MEDIUM", "HIGH"],
        "risk_level": ["LOW", "MEDIUM", "HIGH"],
        "compound": ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"],
        "weather_condition": [
            "dry",
            "wet",
            "damp",
            "drizzle",
            "rain",
            "heavy_rain",
            "mixed",
            "changing",
        ],
    },
}


def validate_golden_dataset(dataset: Dict) -> List[str]:
    """
    Validate a golden dataset against schema.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check race metadata
    if "race" not in dataset:
        errors.append("Missing 'race' metadata")
        return errors

    for field in GOLDEN_DATASET_SCHEMA["required_fields"]["race"]:
        if field not in dataset["race"]:
            errors.append(f"Race metadata missing field: {field}")

    # Check scenarios
    if "scenarios" not in dataset:
        errors.append("Missing 'scenarios' array")
        return errors

    for i, scenario in enumerate(dataset["scenarios"]):
        scenario_id = scenario.get("id", f"scenario_{i}")

        # Required scenario fields
        for field in GOLDEN_DATASET_SCHEMA["required_fields"]["scenario"]:
            if field not in scenario:
                errors.append(f"[{scenario_id}] Missing field: {field}")

        # Validate golden truth
        if "golden_truth" in scenario:
            truth = scenario["golden_truth"]
            for field in GOLDEN_DATASET_SCHEMA["required_fields"]["golden_truth"]:
                if field not in truth:
                    errors.append(f"[{scenario_id}] Golden truth missing: {field}")

            # Validate decision value
            if "decision" in truth:
                if truth["decision"] not in GOLDEN_DATASET_SCHEMA["valid_values"]["decision"]:
                    errors.append(f"[{scenario_id}] Invalid decision: {truth['decision']}")

        # Validate tire data
        if "tires" in scenario:
            tires = scenario["tires"]
            for field in GOLDEN_DATASET_SCHEMA["required_fields"]["tires"]:
                if field not in tires:
                    errors.append(f"[{scenario_id}] Tires missing field: {field}")

            # Validate compound
            if "compound" in tires:
                if tires["compound"] not in GOLDEN_DATASET_SCHEMA["valid_values"]["compound"]:
                    errors.append(f"[{scenario_id}] Invalid compound: {tires['compound']}")

    return errors
