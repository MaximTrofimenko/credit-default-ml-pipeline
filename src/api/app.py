"""HTTP API для прогноза дефолта."""

from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.data.features import add_features

app = FastAPI(title="Credit default prediction")


class ClientData(BaseModel):
    LIMIT_BAL: float = Field(gt=0)
    SEX: int = Field(ge=1, le=2)
    EDUCATION: int = Field(ge=0, le=6)
    MARRIAGE: int = Field(ge=0, le=3)
    AGE: int = Field(ge=18, le=100)
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


@lru_cache
def get_model():
    return joblib.load("models/model.joblib")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(client: ClientData):
    data = pd.DataFrame([client.model_dump()])
    data = add_features(data)
    model = get_model()
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])
    return {"default_prediction": prediction, "default_probability": probability}
