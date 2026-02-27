import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import load_cleaned_data,feature_Construction,spliting_data,save_split_data,pipeline_preprocessor
    


# Fixture: reusable sample CSV

@pytest.fixture
def sample_cleaned_csv(tmp_path):
    file = tmp_path / "cleaned.csv"
    df = pd.DataFrame({
        "Id": [1,2,3,4,5],
        "age": [39.0, 24.0, None, None, None],
        "gender": ["male","male","male","male","male"],
        "bmi": [23.2,30.1,33.3,33.7,34.1],
        "bloodpressure": [91,87,82,80,100],
        "diabetic": ["Yes","No","Yes","No","No"],
        "children": [0,0,0,0,0],
        "smoker": ["No","No","No","No","No"],
        "region": ["southeast","southeast","southeast","northwest","northwest"],
        "claim": [1121.87,1131.51,1135.94,1136.4,1137.01]
    })
    df.to_csv(file, index=False)
    return file


# Happy path tests

def test_load_cleaned_data(sample_cleaned_csv):
    df = load_cleaned_data(sample_cleaned_csv)
    assert df.shape[0] == 5
    assert "bmi" in df.columns

def test_feature_construction(sample_cleaned_csv):
    df = load_cleaned_data(sample_cleaned_csv)
    df_feat = feature_Construction(df)
    # Check new columns
    for col in ["bmi_group", "high_bmi_risk", "high_bp_risk", "age_risk", "smoker_risk", "risk_score"]:
        assert col in df_feat.columns
    # Check bmi_group labels
    assert df_feat["bmi_group"].dtype.name == "category"

def test_spliting_data(sample_cleaned_csv):
    df = load_cleaned_data(sample_cleaned_csv)
    df_feat = feature_Construction(df)
    x_train, x_test, y_train, y_test = spliting_data(df_feat)
    assert x_train.shape[0] + x_test.shape[0] == df_feat.shape[0]
    assert "claim" not in x_train.columns
    assert y_train.name == "claim"

def test_save_split_data(tmp_path, sample_cleaned_csv):
    df = load_cleaned_data(sample_cleaned_csv)
    df_feat = feature_Construction(df)
    x_train, x_test, y_train, y_test = spliting_data(df_feat)
    save_split_data(x_train, x_test, y_train, y_test, tmp_path)
    
    processed_dir = tmp_path / "proccessed"
    assert (processed_dir / "x_train.csv").exists()
    assert (processed_dir / "y_test.csv").exists()

def test_pipeline_preprocessor_creation():
    preprocessor = pipeline_preprocessor()
    # Check it's ColumnTransformer
    from sklearn.compose import ColumnTransformer
    assert isinstance(preprocessor, ColumnTransformer)
    # Check expected transformers exist
    names = [name for name, _, _ in preprocessor.transformers]
    for expected in ["num", "ohe", "ordinal", "binary"]:
        assert expected in names
        

# Exception tests


def test_load_cleaned_data_file_missing():
    with pytest.raises(Exception):
        load_cleaned_data("non_existent_file.csv")

def test_feature_construction_invalid_data():
    # Passing empty DF should raise exception
    empty_df = pd.DataFrame()
    with pytest.raises(Exception):
        feature_Construction(empty_df)