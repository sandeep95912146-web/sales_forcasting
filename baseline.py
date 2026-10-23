import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("sales.csv", parse_dates=["date"])
features = ["promo", "holiday", "school_holiday", "weekend"]
df["t"] = np.arange(len(df))
X = df[features + ["t"]]
y = df["sales"]

split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

print("Baseline MAE:", round(mean_absolute_error(y_test, pred), 2))
print("Average sales:", round(y_test.mean(), 2))

pd.DataFrame({
    "date": df["date"].iloc[split:].values,
    "actual": y_test.values,
    "forecast": pred,
}).to_csv("linear_predictions.csv", index=False)
print("Predictions saved as linear_predictions.csv")
