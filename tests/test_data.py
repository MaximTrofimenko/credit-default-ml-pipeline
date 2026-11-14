import pandas as pd
import pandera.errors
import pytest

from src.data.features import add_features
from src.data.validation import validate_data


@pytest.fixture
def sample():
    return pd.read_csv("UCI_Credit_Card.csv", nrows=3)


def test_valid_data(sample):
    result = validate_data(sample)
    assert len(result) == 3


def test_bad_age_is_rejected(sample):
    sample.loc[0, "AGE"] = 5
    with pytest.raises(pandera.errors.SchemaError):
        validate_data(sample)


def test_missing_value_is_rejected(sample):
    sample.loc[0, "LIMIT_BAL"] = None
    with pytest.raises(pandera.errors.SchemaError):
        validate_data(sample)


def test_features(sample):
    result = add_features(sample)
    assert result.loc[0, "PAY_TOTAL"] == sample.loc[0, "PAY_AMT1":"PAY_AMT6"].sum()
    assert result.loc[0, "DELAY_COUNT"] == 2
