"""Проверка, создание признаков и разделение на train/test."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.features import add_features
from src.data.validation import TARGET, validate_data


def prepare(input_path, output_dir):
    data = pd.read_csv(input_path)
    data = validate_data(data)
    data = add_features(data)
    data = data.drop(columns=["ID"])
    train, test = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data[TARGET]
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    print(f"Обучающая выборка: {len(train)}, тестовая: {len(test)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="UCI_Credit_Card.csv")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    prepare(args.input, args.output_dir)


if __name__ == "__main__":
    main()
