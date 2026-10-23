import sys
import pandas as pd, numpy as np, torch, torch.nn as nn, shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.enabled = False   # lets SHAP take gradients through the LSTM

# ---------- 1. Load data (works with any factor columns) ----------
# Default is sales.csv. For another file, change the name below,
# or run:  python forecast_explain.py sales_rossmann.csv
CSV = sys.argv[1] if len(sys.argv) > 1 else "sales.csv"
df = pd.read_csv(CSV, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
factors = [c for c in df.columns if c not in ("date", "sales")]
print("File:", CSV)
print("Rows:", len(df), "| Factors used:", factors)

split = int(len(df) * 0.8)
df["t"] = np.arange(len(df))
doy = df["date"].dt.dayofyear
df["sin"] = np.sin(2 * np.pi * doy / 365)
df["cos"] = np.cos(2 * np.pi * doy / 365)


def report(name, actual, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    print(f"{name}: MAE {mae:.2f} | RMSE {rmse:.2f} | MAPE {mape:.2f}%")
    return mae


# ---------- 2. Linear model (trend) ----------
lin_cols = factors + ["t"]
lin = LinearRegression().fit(df[lin_cols][:split], df["sales"][:split])
df["base"] = lin.predict(df[lin_cols])
df["resid"] = df["sales"] - df["base"]
actual = df["sales"].values[split:]
print()
base_mae = report("Baseline (linear)", actual, df["base"].values[split:])

# ---------- 3. LSTM on the residual ----------
feat = ["resid"] + factors + ["sin", "cos"]
scaler = StandardScaler().fit(df[feat][:split])
data = scaler.transform(df[feat]).astype(np.float32)

W = 14
X, y = [], []
for i in range(W, len(data)):
    seq = data[i - W + 1:i + 1].copy()
    seq[-1, 0] = 0
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

print()
print("Training the LSTM (a minute or two)...")
for epoch in range(60):
    perm = torch.randperm(len(Xtr))
    for j in range(0, len(Xtr), 64):
        idx = perm[j:j + 64]
        opt.zero_grad()
        loss = loss_fn(model(Xtr[idx]), ytr[idx])
        loss.backward()
        opt.step()

model.eval()
with torch.no_grad():
    resid_pred = model(Xte).numpy()
resid_pred = resid_pred * scaler.scale_[0] + scaler.mean_[0]
final = df["base"].values[split:] + resid_pred
hybrid_mae = report("Hybrid (linear + LSTM)", actual, final)
print("Improvement over baseline:", round((1 - hybrid_mae / base_mae) * 100, 1), "%")

# ---------- 4. SHAP for the linear (trend) part ----------
print()
print("Running SHAP on the linear part...")
lin_bg = df[lin_cols][:split].sample(min(200, split), random_state=0)
lin_explainer = shap.LinearExplainer(lin, lin_bg)
lin_sv = lin_explainer.shap_values(df[lin_cols][split:])
lin_df = pd.DataFrame(np.array(lin_sv), columns=lin_cols)

# ---------- 5. SHAP for the LSTM part ----------
print("Running SHAP on the LSTM part (may take a minute)...")


class Wrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).unsqueeze(1)


wrapped = Wrap(model).eval()
background = Xtr[torch.randperm(len(Xtr))[:100]]
pick = np.linspace(0, len(Xte) - 1, min(60, len(Xte))).astype(int)
explainer = shap.GradientExplainer(wrapped, background)
sv = explainer.shap_values(Xte[pick])
if isinstance(sv, list):
    sv = sv[0]
sv = np.array(sv)
if sv.ndim == 4:
    sv = sv[..., 0]
lstm_df = pd.DataFrame(sv.sum(axis=1) * scaler.scale_[0], columns=feat)

# ---------- 6. Combine both parts ----------
sel = lin_df.iloc[pick].reset_index(drop=True)
total = pd.DataFrame()
for f in factors:
    total[f] = sel[f] + lstm_df[f]
total["trend (time)"] = sel["t"]
total["season (time of year)"] = lstm_df["sin"] + lstm_df["cos"]
total["recent sales pattern"] = lstm_df["resid"]

# ---------- 7. Chart 1: what matters most overall ----------
imp = total.abs().mean().sort_values()
print()
print("Average impact of each factor on the forecast (sales units):")
print(imp.sort_values(ascending=False).round(2).to_string())

plt.figure(figsize=(8, 4.5))
plt.barh(imp.index, imp.values, color="#3b7ddd")
plt.xlabel("Average impact on forecast (sales units)")
plt.title("What drives the sales forecast")
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=150)
plt.close()

# ---------- 8. Chart 2: explain one day ----------
k = int(np.argmax(final[pick]))
day = pd.Timestamp(df["date"].values[split + pick[k]]).date()
row = total.iloc[k].sort_values()
print()
print("Explaining the day with the highest forecast:", day)
print("Forecast:", round(final[pick[k]], 1), " Actual:", round(actual[pick[k]], 1))
print(row.round(1).to_string())

plt.figure(figsize=(8, 4.5))
plt.barh(row.index, row.values, color=["#2e9e5b" if v > 0 else "#d64545" for v in row.values])
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Effect on forecast vs an average day (sales units)")
plt.title("Why this forecast? " + str(day))
plt.tight_layout()
plt.savefig("shap_one_day.png", dpi=150)
plt.close()

# ---------- 9. Chart 3: forecast vs actual ----------
plt.figure(figsize=(9, 4.5))
plt.plot(df["date"].values[split:], actual, label="Actual", color="#555555")
plt.plot(df["date"].values[split:], final, label="Hybrid forecast", color="#3b7ddd")
plt.legend()
plt.title("Forecast vs actual sales (test period)")
plt.tight_layout()
plt.savefig("forecast_vs_actual.png", dpi=150)
plt.close()

# ---------- 10. Save explanations for the chat (RAG) ----------
out = total.copy()
out.insert(0, "date", [str(pd.Timestamp(df["date"].values[split + p]).date()) for p in pick])
out.insert(1, "forecast", final[pick].round(1))
out.insert(2, "actual", actual[pick].round(1))
out = out.round(2)
out.to_csv("shap_values.csv", index=False)

print()
print("Done. Saved: shap_importance.png, shap_one_day.png, forecast_vs_actual.png, shap_values.csv")
