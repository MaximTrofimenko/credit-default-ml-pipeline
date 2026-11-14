"""Простые проверки входного CSV."""

import argparse

import pandas as pd
import pandera.pandas as pa
from pandera import Check

from src.data.features import BILL_COLUMNS, PAY_AMOUNT_COLUMNS, PAY_COLUMNS

TARGET = "default.payment.next.month"


def make_schema():
    columns = {
        "ID": pa.Column(int, Check.greater_than(0), unique=True, nullable=False),
        "LIMIT_BAL": pa.Column(float, Check.greater_than(0), nullable=False),
        "SEX": pa.Column(int, Check.isin([1, 2]), nullable=False),
        "EDUCATION": pa.Column(int, Check.in_range(0, 6), nullable=False),
        "MARRIAGE": pa.Column(int, Check.in_range(0, 3), nullable=False),
        "AGE": pa.Column(int, Check.in_range(18, 100), nullable=False),
        TARGET: pa.Column(int, Check.isin([0, 1]), nullable=False),
    }
    for name in PAY_COLUMNS:
        columns[name] = pa.Column(int, Check.in_range(-2, 9), nullable=False)
    for name in BILL_COLUMNS:
        columns[name] = pa.Column(float, nullable=False)
    for name in PAY_AMOUNT_COLUMNS:
        columns[name] = pa.Column(float, Check.ge(0), nullable=False)
    return pa.DataFrameSchema(columns, strict=True, coerce=True)


def validate_data(data):
    return make_schema().validate(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    data = pd.read_csv(args.csv_path)
    validate_data(data)
    print(f"Проверка пройдена: {len(data)} строк")


if __name__ == "__main__":
    main()
