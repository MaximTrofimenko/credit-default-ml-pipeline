"""Признаки, одинаковые для обучения и API."""

PAY_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMOUNT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]


def add_features(data):
    data = data.copy()
    data["PAY_TOTAL"] = data[PAY_AMOUNT_COLUMNS].sum(axis=1)
    data["BILL_TOTAL"] = data[BILL_COLUMNS].sum(axis=1)
    data["DELAY_COUNT"] = (data[PAY_COLUMNS] > 0).sum(axis=1)
    data["PAY_TO_BILL_RATIO"] = data["PAY_AMT1"] / (data["BILL_AMT1"].abs() + 1)
    data["PAY_TO_BILL_RATIO"] = data["PAY_TO_BILL_RATIO"].clip(upper=10)
    return data
