"""
Unit tests for golden dataset validation.
Tests work across ALL golden datasets.
"""

import re

import pytest


class TestGoldenDatasetStructure:
    """Validate golden dataset structure"""

    def test_dataset_loads_successfully(self, golden_dataset):
        """Single dataset loads successfully"""
        assert golden_dataset is not None
        assert isinstance(golden_dataset, dict)

    def test_dataset_has_race_metadata(self, golden_dataset):
        """Dataset contains required race information"""
        assert "race" in golden_dataset
        race = golden_dataset["race"]

        required_fields = ["name", "track", "total_laps", "regulation_era"]
        for field in required_fields:
            assert field in race, f"Race metadata missing: {field}"

        assert isinstance(race["name"], str)
        assert isinstance(race["track"], str)
        assert isinstance(race["total_laps"], int)
        assert race["total_laps"] > 0

    def test_dataset_has_scenarios(self, golden_dataset):
        """Dataset contains test scenarios"""
        assert "scenarios" in golden_dataset
        scenarios = golden_dataset["scenarios"]

        assert isinstance(scenarios, list)
        assert len(scenarios) > 0, "Dataset must have at least 1 scenario"

    def test_scenario_required_fields(self, golden_dataset):
        """Every scenario has the minimum required fields"""
        required_fields = [
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
        ]

        for scenario in golden_dataset["scenarios"]:
            for field in required_fields:
                assert (
                    field in scenario
                ), f"Scenario '{scenario.get('id', 'unknown')}' missing required field: {field}"

    def test_golden_truth_structure(self, golden_dataset):
        """Golden truth has decision, rationale, confidence, risk"""
        for scenario in golden_dataset["scenarios"]:
            truth = scenario["golden_truth"]

            assert "decision" in truth
            assert truth["decision"] in ["BOX", "STAY_OUT"]

            assert "rationale" in truth
            assert isinstance(truth["rationale"], str)
            assert len(truth["rationale"]) > 10

            assert "confidence_level" in truth
            assert truth["confidence_level"] in ["LOW", "MEDIUM", "HIGH"]

            assert "risk_level" in truth
            assert truth["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    def test_tire_compounds_valid(self, golden_dataset):
        """Tire compounds match FIA regulations"""
        valid_compounds = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

        for scenario in golden_dataset["scenarios"]:
            tire_compound = scenario["tires"]["compound"]
            assert (
                tire_compound in valid_compounds
            ), f"Invalid compound: {tire_compound} in scenario {scenario['id']}"

    def test_tire_age_valid(self, golden_dataset):
        """Tire age is non-negative and reasonable"""
        for scenario in golden_dataset["scenarios"]:
            tire_age = scenario["tires"]["age_laps"]
            assert isinstance(tire_age, int)
            assert tire_age >= 0, f"Negative tire age in {scenario['id']}"
            assert tire_age <= 100, f"Unrealistic tire age in {scenario['id']}"

    def test_lap_number_valid(self, golden_dataset):
        """Lap number is within race bounds"""
        total_laps = golden_dataset["race"]["total_laps"]

        for scenario in golden_dataset["scenarios"]:
            lap = scenario["lap"]
            assert isinstance(lap, int)
            assert (
                1 <= lap <= total_laps
            ), f"Lap {lap} outside race bounds (1-{total_laps}) in {scenario['id']}"

    def test_scenario_ids_unique(self, golden_dataset):
        """No duplicate scenario IDs within a dataset"""
        scenario_ids = [s["id"] for s in golden_dataset["scenarios"]]
        assert len(scenario_ids) == len(set(scenario_ids)), "Duplicate scenario IDs found"

    def test_scenario_ids_naming_convention(self, golden_dataset):
        """Scenario IDs follow pattern: track_year_driver_lapX"""
        pattern = r"^[a-z_]+_\d{4}_[a-z]+_lap\d+.*$"

        for scenario in golden_dataset["scenarios"]:
            scenario_id = scenario["id"]
            assert re.match(
                pattern, scenario_id
            ), f"Scenario ID doesn't follow convention: {scenario_id}"

    def test_weather_conditions_valid(self, golden_dataset):
        """Weather conditions use valid values"""
        valid_conditions = [
            "dry",
            "wet",
            "damp",
            "drizzle",
            "rain",
            "heavy_rain",
            "mixed",
            "changing",
        ]

        for scenario in golden_dataset["scenarios"]:
            condition = scenario["weather"]["condition"]
            assert (
                condition in valid_conditions
            ), f"Invalid weather condition: {condition} in {scenario['id']}"


class TestCrossDatasetConsistency:
    """Tests that apply across multiple datasets"""

    def test_all_datasets_load(self, all_golden_datasets):
        """All golden datasets can be loaded"""
        assert len(all_golden_datasets) > 0, "No golden datasets found"

        for dataset_name, dataset in all_golden_datasets:
            assert "scenarios" in dataset
            print(f"\n✓ {dataset_name}: {len(dataset['scenarios'])} scenarios")

    def test_all_datasets_have_unique_ids(self, all_golden_datasets):
        """Scenario IDs are unique across ALL datasets"""
        all_ids = []

        for dataset_name, dataset in all_golden_datasets:
            scenario_ids = [s["id"] for s in dataset["scenarios"]]
            all_ids.extend(scenario_ids)

        # Check for duplicates across datasets
        duplicates = [id for id in all_ids if all_ids.count(id) > 1]
        assert len(duplicates) == 0, f"Duplicate IDs across datasets: {set(duplicates)}"


class TestSchemaValidation:
    """Validate against schema"""

    def test_dataset_passes_schema_validation(self, golden_dataset):
        """Dataset conforms to schema"""
        from src.validators.schema_validator import validate_golden_dataset

        errors = validate_golden_dataset(golden_dataset)

        assert len(errors) == 0, f"Schema validation errors:\n" + "\n".join(errors)
