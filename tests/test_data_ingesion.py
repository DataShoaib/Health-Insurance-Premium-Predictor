import pytest
import pandas as pd
from src.data.data_ingesion import data_load, data_cleaning

@pytest.fixture
def sample_csv(tmp_path):
    file = tmp_path / "sample.csv"
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

def test_data_load(sample_csv):
    df = data_load(sample_csv)
    for col in ["age", "gender", "bmi", "bloodpressure", "diabetic", "children", "smoker", "region", "claim"]:
        assert col in df.columns
    assert df.shape[0] > 0

def test_data_cleaning(sample_csv):
    df = data_load(sample_csv)
    cleaned_df = data_cleaning(df)
    assert cleaned_df.isnull().sum().sum() == 0
    assert cleaned_df.duplicated().sum() == 0
    assert "Id" not in cleaned_df.columns

def test_data_load_file_missing():
    import pytest
    with pytest.raises(Exception):
        data_load("non_existent.csv")