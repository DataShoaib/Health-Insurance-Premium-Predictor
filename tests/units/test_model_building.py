import pytest
import pandas as pd
import numpy as np
import os
import json
from sklearn.pipeline import Pipeline

from src.model.model_building import (
    load_data,
    model_training,
    save_model,
    save_run_info
)


# ==========================================================
# FIXTURE: Sample Training Data (No external dependency)
# ==========================================================
@pytest.fixture
def sample_training_data():

    df = pd.DataFrame({
        "age": [22.0, 28.0, 38.0],
        "gender": ["male", "female", "female"],
        "bmi": [34.1, 21.8, 34.8],
        "bloodpressure": [108, 81, 132],
        "diabetic": ["No", "No", "No"],
        "children": [0, 0, 2],
        "smoker": ["Yes", "Yes", "Yes"],
        "region": ["northeast", "southwest", "southwest"],
        "bmi_group": ["obese", "normal", "obese"],
        "high_bmi_risk": [1, 0, 1],
        "high_bp_risk": [1, 0, 1],
        "age_risk": ["Low", "Low", "Medium"],
        "smoker_risk": [1, 1, 1],
        "risk_score": [2, 0, 2],
        "claim": [15000, 8000, 20000]
    })

    X = df.drop(columns=["claim"])
    y = df["claim"]

    return X, y

# ==========================================================
# TEST: load_data - File Missing Case
# ==========================================================
def test_load_data_file_missing():
    with pytest.raises(Exception):
        load_data("fake_x.csv", "fake_y.csv")


# ==========================================================
# TEST: load_data - Successful Load (using tmp_path)
# ==========================================================
def test_load_data_success(tmp_path):
    """
    Uses pytest tmp_path fixture to create temporary files.
    No real project files touched.
    """
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.DataFrame({"target": [10, 20, 30]})

    x_path = tmp_path / "x.csv"
    y_path = tmp_path / "y.csv"

    X.to_csv(x_path, index=False)
    y.to_csv(y_path, index=False)

    x_loaded, y_loaded = load_data(str(x_path), str(y_path))

    assert isinstance(x_loaded, pd.DataFrame)
    assert isinstance(y_loaded, pd.DataFrame)
    assert len(x_loaded) == 3
    assert len(y_loaded) == 3


# ==========================================================
# TEST: model_training
# ==========================================================
def test_model_training(sample_training_data):
    """
    Model should train successfully using in-memory fixture data.
    """
    X, y = sample_training_data

    model = model_training(X, y)

    assert isinstance(model, Pipeline)
    assert "model" in model.named_steps

    # Check prediction works
    preds = model.predict(X)
    assert len(preds) == len(X)


# ==========================================================
# TEST: save_model
# ==========================================================
def test_save_model(tmp_path, sample_training_data):
    """
    Model should be saved inside temporary directory.
    """
    X, y = sample_training_data
    model = model_training(X, y)

    save_model(model, str(tmp_path))

    saved_file = os.path.join(tmp_path, "model.pkl")
    assert os.path.exists(saved_file)


# ==========================================================
# TEST: save_run_info
# ==========================================================
def test_save_run_info(tmp_path):
    """
    JSON file should be created correctly with proper content.
    """
    run_id = "test_run_123"
    model_path = "model"
    file_path = tmp_path / "run_info.json"

    save_run_info(run_id, model_path, str(file_path))

    assert os.path.exists(file_path)

    with open(file_path) as f:
        data = json.load(f)

    assert data["run_id"] == run_id
    assert data["model_path"] == model_path