"""Отправляет новые строки в API и сравнивает распределения признаков."""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from src.data.validation import TARGET


def psi(reference, current):
    cuts = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    edges = np.concatenate(([-np.inf], cuts[1:-1], [np.inf]))
    old_counts = np.histogram(reference, bins=edges)[0]
    new_counts = np.histogram(current, bins=edges)[0]
    old_share = (old_counts + 0.001) / (len(reference) + 0.001 * len(old_counts))
    new_share = (new_counts + 0.001) / (len(current) + 0.001 * len(new_counts))
    return float(np.sum((new_share - old_share) * np.log(new_share / old_share)))


def send_to_api(row, api_url):
    body = json.dumps(row).encode("utf-8")
    request = Request(api_url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def monitor(input_path, reference_path, api_url, limit):
    current = pd.read_csv(input_path)
    reference = pd.read_csv(reference_path)
    features = [name for name in current.columns if name not in ["ID", TARGET]]
    probabilities = []
    for _, row in current.head(limit).iterrows():
        result = send_to_api(row[features].to_dict(), api_url)
        probabilities.append(result["default_probability"])
    results = {
        "requests": len(probabilities),
        "mean_probability": float(np.mean(probabilities)),
    }
    for column in ["LIMIT_BAL", "AGE", "PAY_0"]:
        results[f"psi_{column}"] = psi(reference[column], current[column])
    results["drift_detected"] = any(
        results[f"psi_{name}"] > 0.2 for name in ["LIMIT_BAL", "AGE", "PAY_0"]
    )
    Path("reports").mkdir(exist_ok=True)
    with open("reports/drift.json", "w") as file:
        json.dump(results, file, indent=2)
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", default="data/processed/train.csv")
    parser.add_argument("--api-url", default="http://localhost:8000/predict")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    monitor(args.input, args.reference, args.api_url, args.limit)


if __name__ == "__main__":
    main()
