import pytest
import pandas as pd
from src.data.data_ingesion import data_load, data_cleaning

@pytest.fixture
def sample_csv(tmp_path):
    file = tmp_path / "sample.csv"
    df = pd.DataFrame({
        "Id": [1, 2, 3],
        "age": [25, 30, None],
        "gender": ["M", "F", "M"],
        "bmi": [22.5, 27.3, 24.0],
        "bloodpressure": [120, 130, 125],
        "diabetic": [0, 1, 0],
        "children": [0, 2, 1],
        "smoker": [0, 1, 0],
        "region": ["NE", "SW", "NW"],
        "claim": [0, 1, 0]
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