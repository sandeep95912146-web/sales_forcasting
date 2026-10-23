import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

# This baseline forecasts the next 20% of dates using prior sales and known factors.
df = pd.read_csv("sales.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
features = ["promo", "holiday", "school_holiday", "weekend"]
missing = [c for c in ["date", "sales", *features] if c not in df.columns]
if missing:
    raise SystemExit(f"sales.csv is missing required columns: {', '.join(missing)}")

split = int(len(df) * 0.8)
y_train, y_test = df["sales"].iloc[:split], df["sales"].iloc[split:]
x_train, x_test = df[features].iloc[:split], df[features].iloc[split:]

# Weekly seasonality uses a seven-day cycle. Promo and calendar factors are supplied as exogenous inputs.
model = ARIMA(
    y_train,
    exog=x_train,
    order=(1, 1, 1),
    seasonal_order=(1, 0, 0, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
fit = model.fit()
pred = fit.forecast(steps=len(y_test), exog=x_test)
mae = mean_absolute_error(y_test, pred)

out = pd.DataFrame({
    "date": df["date"].iloc[split:].values,
    "actual": y_test.values,
    "forecast": pred.values,
})
out.to_csv("arima_predictions.csv", index=False)

print("ARIMA order: (1, 1, 1), weekly seasonal order: (1, 0, 0, 7)")
print("Test rows:", len(y_test))
print("ARIMA MAE:", round(mae, 2))
print("Predictions saved as arima_predictions.csv")

