"""Короткий отчёт о датасете до обучения."""

import pandas as pd

from src.data.validation import TARGET, validate_data


def main():
    data = validate_data(pd.read_csv("UCI_Credit_Card.csv"))
    print("Размер:", data.shape)
    print("Пропуски:", data.isna().sum().sum())
    print("Дубли ID:", data.ID.duplicated().sum())
    print("Классы:")
    print(data[TARGET].value_counts(normalize=True).round(3))
    print("Возраст и кредитный лимит:")
    print(data[["AGE", "LIMIT_BAL"]].describe().round(1))


if __name__ == "__main__":
    main()
