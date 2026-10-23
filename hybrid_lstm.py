import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

torch.manual_seed(42)
np.random.seed(42)

# ---------- 1. Load data and build simple features ----------
df = pd.read_csv("sales.csv", parse_dates=["date"])
df = df.dropna(subset=["sales", "promo", "holiday", "school_holiday", "weekend"]).reset_index(drop=True)

split = int(len(df) * 0.8)
df["t"] = np.arange(len(df))
doy = df["date"].dt.dayofyear
df["sin"] = np.sin(2 * np.pi * doy / 365)
df["cos"] = np.cos(2 * np.pi * doy / 365)

# ---------- 2. Step A: linear model captures the trend ----------
lin_cols = ["promo", "holiday", "school_holiday", "weekend", "t"]
lin = LinearRegression().fit(df[lin_cols][:split], df["sales"][:split])
df["base"] = lin.predict(df[lin_cols])
df["resid"] = df["sales"] - df["base"]
base_mae = mean_absolute_error(df["sales"][split:], df["base"][split:])
print("Baseline MAE:", round(base_mae, 2))

# ---------- 3. Step B: LSTM learns what the linear model missed (the residual) ----------
feat = ["resid", "promo", "holiday", "school_holiday", "weekend", "sin", "cos"]
scaler = StandardScaler().fit(df[feat][:split])
data = scaler.transform(df[feat]).astype(np.float32)

W = 14
X, y = [], []
for i in range(W, len(data)):
    seq = data[i - W + 1:i + 1].copy()
    seq[-1, 0] = 0  # hide today's residual so the model can't cheat
    X.append(seq)
    y.append(data[i, 0])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

n_train = split - W
Xtr, ytr = torch.tensor(X[:n_train]), torch.tensor(y[:n_train])
Xte = torch.tensor(X[n_train:])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(len(feat), 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).squeeze(1)


model = Net()
opt = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

# ---------- 4. Train ----------
for epoch in range(40):
    perm = torch.randperm(len(Xtr))
    total = 0.0
    for j in range(0, len(Xtr), 64):
        idx = perm[j:j + 64]
        opt.zero_grad()
        loss = loss_fn(model(Xtr[idx]), ytr[idx])
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    if (epoch + 1) % 10 == 0:
        print("Epoch", epoch + 1, "avg loss:", round(total / len(Xtr), 4))

# ---------- 5. Final forecast = linear trend + LSTM correction ----------
model.eval()
with torch.no_grad():
    resid_pred = model(Xte).numpy()

resid_pred = resid_pred * scaler.scale_[0] + scaler.mean_[0]
final = df["base"].values[split:] + resid_pred
actual = df["sales"].values[split:]
hybrid_mae = mean_absolute_error(actual, final)

print("Hybrid LSTM MAE:", round(hybrid_mae, 2))
print("Improvement over baseline:", round((1 - hybrid_mae / base_mae) * 100, 1), "%")

pd.DataFrame({
    "date": df["date"].iloc[split:].values,
    "actual": actual,
    "forecast": final,
}).to_csv("hybrid_lstm_predictions.csv", index=False)
print("Predictions saved as hybrid_lstm_predictions.csv")
