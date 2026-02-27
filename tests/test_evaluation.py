import pytest
import pandas as pd
import numpy as np
import os
import json
import pickle

from src.evaluation.evaluation import (
    load_data,
    load_model,
    model_evaluation,
    save_metrics
)


# ==========================================================
# DUMMY MODEL (No sklearn pipeline, no real training)
# ==========================================================
class DummyModel:
    def predict(self, X):
        # Always return constant prediction
        return np.ones(len(X)) * 100


# ==========================================================
# FIXTURE: Sample Test Data (In-memory only)
# ==========================================================
@pytest.fixture
def sample_test_data():
    X = pd.DataFrame({
        "age": [25, 30, 45, 50],
        "bmi": [22.1, 28.5, 30.0, 26.3],
        "children": [0, 1, 2, 3]
    })

    y = pd.Series([100, 100, 100, 100])

    return X, y


# ==========================================================
# TEST: load_data - Missing File
# ==========================================================
def test_load_data_file_missing():
    with pytest.raises(Exception):
        load_data("fake_x.csv", "fake_y.csv")


# ==========================================================
# TEST: load_data - Success (tmp_path used)
# ==========================================================
def test_load_data_success(tmp_path):
    X = pd.DataFrame({"a": [1, 2]})
    y = pd.DataFrame({"target": [10, 20]})

    x_path = tmp_path / "x_test.csv"
    y_path = tmp_path / "y_test.csv"

    X.to_csv(x_path, index=False)
    y.to_csv(y_path, index=False)

    x_loaded, y_loaded = load_data(str(x_path), str(y_path))

    assert isinstance(x_loaded, pd.DataFrame)
    assert isinstance(y_loaded, pd.DataFrame)
    assert len(x_loaded) == 2


# ==========================================================
# TEST: load_model
# ==========================================================
def test_load_model(tmp_path):
    dummy_model = DummyModel()

    model_path = tmp_path / "model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(dummy_model, f)

    loaded_model = load_model(str(model_path))

    assert hasattr(loaded_model, "predict")


# ==========================================================
# TEST: model_evaluation (NO real pipeline)
# ==========================================================
def test_model_evaluation(sample_test_data):
    X, y = sample_test_data

    dummy_model = DummyModel()

    metrics = model_evaluation(dummy_model, X, y)

    assert isinstance(metrics, dict)
    assert "mae" in metrics
    assert "mse" in metrics
    assert "r2_scr" in metrics


# ==========================================================
# TEST: save_metrics
# ==========================================================
def test_save_metrics(tmp_path):
    metrics = {
        "mae": 0.0,
        "mse": 0.0,
        "r2_scr": 1.0
    }

    save_path = tmp_path / "metrics.json"

    save_metrics(metrics, str(save_path))

    assert os.path.exists(save_path)

    with open(save_path) as f:
        saved_data = json.load(f)

    assert saved_data["mae"] == 0.0