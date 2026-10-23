from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query


PROJECT = Path(__file__).resolve().parent
PREDICTIONS_FILE = PROJECT / "arima_predictions.csv"

app = FastAPI(
    title="Explainable Sales Forecasting API",
    description="Provides the saved ARIMA forecast for the project dashboard.",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "message": "Sales forecasting API is running.",
        "docs": "/docs",
        "forecast": "/forecast",
    }


@app.get("/health")
def health():
    return {"status": "ok", "predictions_file_exists": PREDICTIONS_FILE.exists()}


@app.get("/forecast")
def forecast(limit: int = Query(default=189, ge=1, le=189)):
    if not PREDICTIONS_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="arima_predictions.csv is missing. Run arima_baseline.py first.",
        )

    predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=["date"])
    required = {"date", "actual", "forecast"}
    if not required.issubset(predictions.columns):
        raise HTTPException(status_code=500, detail="The forecast file has unexpected columns.")

    predictions = predictions.head(limit)
    rows = predictions.assign(date=predictions["date"].dt.strftime("%Y-%m-%d"))
    return {
        "model": "ARIMA",
        "rows": len(rows),
        "data": rows[["date", "actual", "forecast"]].to_dict(orient="records"),
    }
