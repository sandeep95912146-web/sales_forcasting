import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

torch.manual_seed(42)

# ---------- Load & clean ----------
df = pd.read_csv("sales.csv", parse_dates=["date"])
cols = ["sales", "promo", "holiday", "school_holiday", "weekend"]
df = df.dropna(subset=cols).reset_index(drop=True)

split = int(len(df) * 0.8)
scaler = StandardScaler().fit(df[cols][:split])
data = scaler.transform(df[cols])

# ---------- Build sliding windows ----------
W = 14
X, y = [], []
for i in range(W, len(data)):
    seq = data[i - W + 1:i + 1].copy()
    seq[-1, 0] = 0  # hide today's sales so the model can't cheat
    X.append(seq)
    y.append(data[i, 0])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

n_train = split - W
Xtr, ytr = torch.tensor(X[:n_train]), torch.tensor(y[:n_train])
Xte, yte = torch.tensor(X[n_train:]), y[n_train:]


# ---------- Model ----------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).squeeze(1)


model = Net()
opt = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

# ---------- Train ----------
for epoch in range(30):
    perm = torch.randperm(len(Xtr))
    epoch_losses = []
    for j in range(0, len(Xtr), 64):
        idx = perm[j:j + 64]
        opt.zero_grad()
        loss = loss_fn(model(Xtr[idx]), ytr[idx])
        loss.backward()
        opt.step()
        epoch_losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1}  avg loss: {avg_loss:.4f}")

# ---------- Evaluate ----------
model.eval()
with torch.no_grad():
    pred = model(Xte).numpy()

pred = pred * scaler.scale_[0] + scaler.mean_[0]
actual = yte * scaler.scale_[0] + scaler.mean_[0]

print("LSTM MAE:", round(mean_absolute_error(actual, pred), 2))

pd.DataFrame({
    "date": df["date"].iloc[split:].values,
    "actual": actual,
    "forecast": pred,
}).to_csv("lstm_predictions.csv", index=False)
print("Predictions saved as lstm_predictions.csv")
