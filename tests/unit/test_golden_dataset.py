"""
Unit tests for golden dataset validation.
Tests work across ALL golden datasets using parameterisation. 
"""

import json
import pytest
from pathlib import Path
from src.validators.schema_validator import validate_golden_dataset

def get_all_golden_datasets():
    """Discover all golden dataset files"""
    golden_dir = Path("datasets/golden")
    dataset_files = list(golden_dir.glob("*_scenarios.json"))
    return [(f.stem, f) for f in dataset_files]


@pytest.fixture(params=get_all_golden_datasets(), ids=lambda x: x[0])
def golden_dataset(request):
    """
    Parameterised fixture - run tests against all golden datasets.
    This will automatically test new datasets as you add them.
    """
    dataset_name, dataset_path = request.param
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        data ['_dataset_name'] = dataset_name
    return data


class TestGoldenDatasetStructure:
    """Tests to validate structure of golden datasets."""

    def test_dataset_loads_successfully(self, golden_dataset):
        assert golden_dataset is not None
        assert isinstance(golden_dataset, dict)
        print(f"Testing: {golden_dataset['_dataset_name']}")

    def test_dataset_has_race_metadata(self, golden_dataset):
        assert "race" in golden_dataset
        race = golden_dataset["race"]

        # Required fields for any race
        required_fields = ["name", "track", "total_laps", "regulation_era"]
        for field in required_fields:
            assert field in race, f"Missing metadata: {field}"

        # Validate data types
        assert isinstance(race["name"], str)
        assert isinstance(race["track"], str)
        assert isinstance(race["total_laps"], int)
        assert race["total_laps"] > 0

    def test_dataset_has_scenarios(self, golden_dataset):
        assert "scenarios" in golden_dataset
        scenarios = golden_dataset["scenarios"]
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0, "No scenarios defined in dataset, must have at least 1 scenario"

    def test_scenario_required_fields(self, golden_dataset):
        required_fields = [
            "id", "name", "lap", "driver", "position", "tires", "weather", "context", "golden_truth", "test_criteria"
        ]
        for scenario in golden_dataset["scenarios"]:
            for field in required_fields:
                assert field in scenario, f"Scenario '{scenario.get('id', 'unknown')}' missing required field: {field}"

    def test_golden_truth_structure(self, golden_dataset):
        for scenario in golden_dataset["scenarios"]:
            truth = scenario["golden_truth"]

            # Decision
            assert "decision" in truth
            assert truth["decision"] in ["BOX", "STAY_OUT"], \
                f"Invalid decision: {truth['decision']}"

            # Rationale
            assert "rationale" in truth
            assert isinstance(truth["rationale"], str)
            assert len(truth["rationale"]) > 10, "Rationale too short"

            # Confidence Level
            assert "confidence_level" in truth
            confidence = truth["confidence_level"] in ["HIGH", "MEDIUM", "LOW"], \
                f"Invalid confidence level: {truth['confidence_level']}"

            # Risk Level
            assert "risk_level" in truth
            risk = truth["risk_level"] in ["HIGH", "MEDIUM", "LOW"], \
                f"Invalid risk level: {truth['risk_level']}"

    def test_tire_compounds_valid(self, golden_dataset):
        valid_compounds = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

        for scenario in golden_dataset["scenarios"]:
            tire_compound = scenario["tires"]["compound"]
            assert tire_compound in valid_compounds, \
                f"Invalid tire compound: {tire_compound} in scenario {scenario['id']}"

    def test_tire_age_is_valid(self, golden_dataset):
        for scenario in golden_dataset["scenarios"]:
            tire_age = scenario["tires"]["age_laps"]
            assert isinstance(tire_age, int)
            assert tire_age >= 0, f"Negative tire age in scenario {scenario['id']}"
            assert tire_age <=100, f"Unrealistic tire age in scenario {scenario['id']}"

    def test_lap_number_valid(self, golden_dataset):
        total_laps = golden_dataset["race"]["total_laps"]

        for scenario in golden_dataset["scenarios"]:
            lap_number = scenario["lap"]
            assert isinstance(lap_number, int)
            assert 1 <= lap_number <= total_laps, \
                f"Lap number {lap_number} out of range in scenario {scenario['id']}"

    def test_scenario_ids_unique(self, golden_dataset):
        scenario_ids = [scenario["id"] for scenario in golden_dataset["scenarios"]]
        assert len(scenario_ids) == len(set(scenario_ids)), "Duplicate scenario IDs found"

    #TODO: Test scenario IDs follow naming convention
    def test_scenario_id_naming_convention(self, golden_dataset):
        import re
        pattern = r'^[a-z_]+_\d{4}_[a-z]+_lap\d+.*$'

        for scenario in golden_dataset["scenarios"]:
            scenario_id = scenario["id"]
            assert re.match(pattern, scenario_id), \
                f"Scenario ID '{scenario_id}' does not follow naming convention"


class TestScenarioExtraction:
    """Tests to validate scenario extraction logic."""

    def test_filter_scenarios_by_decision(self, golden_dataset):
        scenarios = golden_dataset["scenarios"]

        box_scenarios = [
            s for s in scenarios if s["golden_truth"]["decision"] == "BOX"
        ]

        stay_out_scenarios = [
            s for s in scenarios if s["golden_truth"]["decision"] == "STAY_OUT"
        ]

        # Should have at least one of each type for good test coverage
        total = len(box_scenarios) + len(stay_out_scenarios)
        assert total == len(scenarios), "Some scenarios have invalid decision types"

        # Should have both types for diversity, however don't fail if a dataset only has one type
        if len(scenarios) >= 3:
            assert len(box_scenarios) > 0 or len(stay_out_scenarios) > 0
    
    def test_weather_conditions_valid(self, golden_dataset):
        valid_conditions = ["dry", "wet", "damp", "drizzle", "rain", "heavy_rain", "mixed", "changing"]

        for scenario in golden_dataset["scenarios"]:
            condition = scenario["weather"]["condition"]
            assert condition in valid_conditions, \
                f"Invalid weather condition: {condition} in scenario {scenario['id']}"


class TestDataConsistency:
    """Tests that apply across multiple datasaets"""

    def test_all_datasets_load(self):
        """All golden datasets can be loaded"""
        golden_dir = Path("datasets/golden")
        dataset_files = list(golden_dir.glob("*_scenarios.json"))
        
        assert len(dataset_files) > 0, "No golden datasets found"
        
        for dataset_file in dataset_files:
            with open(dataset_file, 'r') as f:
                data = json.load(f)
                assert "scenarios" in data
                print(f"✓ {dataset_file.stem}: {len(data['scenarios'])} scenarios")


class TestSchemaValidation:
    """Use the schema validator"""
    
    def test_dataset_passes_schema_validation(self, golden_dataset):
        """Dataset conforms to schema"""
        errors = validate_golden_dataset(golden_dataset)
        
        assert len(errors) == 0, f"Schema validation errors:\n" + "\n".join(errors)