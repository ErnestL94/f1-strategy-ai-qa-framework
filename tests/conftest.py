"""
Shared pytest fixtures for all tests.

conftest.py is automatically discovered by pytest and makes fixtures
available to all test files without imports.
"""
import json
import pytest
from pathlib import Path


@pytest.fixture
def project_root(pytestconfig):
    """Get the project root directory"""
    return pytestconfig.rootpath


@pytest.fixture
def golden_dataset_path(project_root):
    """Path to Silverstone 2023 golden dataset"""
    return project_root / "datasets" / "golden" / "silverstone_2023_scenarios.json"


@pytest.fixture
def golden_dataset(golden_dataset_path):
    """
    Load Silverstone 2023 golden dataset.
    
    This fixture is available to ALL test files automatically.
    """
    assert golden_dataset_path.exists(), f"Dataset not found at {golden_dataset_path}"
    
    with open(golden_dataset_path, 'r') as f:
        data = json.load(f)
    
    return data


@pytest.fixture
def all_golden_datasets(project_root):
    """
    Discover and load ALL golden datasets.
    
    Returns list of (dataset_name, dataset_data) tuples.
    """
    golden_dir = project_root / "datasets" / "golden"
    dataset_files = list(golden_dir.glob("*_scenarios.json"))
    
    datasets = []
    for dataset_file in dataset_files:
        with open(dataset_file, 'r') as f:
            data = json.load(f)
            data['_dataset_name'] = dataset_file.stem
            datasets.append((dataset_file.stem, data))
    
    return datasets


@pytest.fixture
def agent():
    """
    Initialize F1 Strategy Agent.
    
    Available to all tests that need the agent.
    """
    from src.rag.agent import F1StrategyAgent
    return F1StrategyAgent()