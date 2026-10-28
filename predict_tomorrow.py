import sys
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)

# Default is sales.csv. For another file, run:  python predict_tomorrow.py sales_sample.csv
CSV = sys.argv[1] if len(sys.argv) > 1 else "sales.csv"

# ---------- 1. Load data ----------
df = pd.read_csv(CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
factors = [c for c in df.columns if c not in ("date", "sales")]
n = len(df)
W = 14
print("File:", CSV, "| Rows:", n, "| Factors:", factors)

df["t"] = np.arange(n)
doy = df["date"].dt.dayofyear
df["sin"] = np.sin(2 * np.pi * doy / 365)
df["cos"] = np.cos(2 * np.pi * doy / 365)

# ---------- 2. Linear model (trend + factors), trained on all data ----------
lin_cols = factors + ["t"]
lin = LinearRegression().fit(df[lin_cols], df["sales"])
df["base"] = lin.predict(df[lin_cols])
df["resid"] = df["sales"] - df["base"]

# ---------- 3. LSTM on the residual, trained on all data ----------
feat = ["resid"] + factors + ["sin", "cos"]
scaler = StandardScaler().fit(df[feat])
data = scaler.transform(df[feat]).astype(np.float32)

X, y = [], []
for i in range(W, n):
    seq = data[i - W + 1:i + 1].copy()
    seq[-1, 0] = 0
    X.append(seq)
    y.append(data[i, 0])
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
Xtr, ytr = torch.tensor(X), torch.tensor(y)


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
print("Training the model (about a minute)...")
for epoch in range(40):
    perm = torch.randperm(len(Xtr))
    for j in range(0, len(Xtr), 64):
        idx = perm[j:j + 64]
        opt.zero_grad()
        loss = loss_fn(model(Xtr[idx]), ytr[idx])
        loss.backward()
        opt.step()
    if (epoch + 1) % 10 == 0:
        print("Epoch", epoch + 1, "done")
model.eval()

# ---------- 4. Tomorrow ----------
next_date = df["date"].iloc[-1] + pd.Timedelta(days=1)
sin_n = np.sin(2 * np.pi * next_date.dayofyear / 365)
cos_n = np.cos(2 * np.pi * next_date.dayofyear / 365)


def ask(name):
    binary = set(df[name].unique()) <= {0, 1}
    if name == "weekend":
        default = 1 if next_date.dayofweek >= 5 else 0
    elif binary:
        default = 0
    else:
        default = float(df[name].iloc[-1])
    hint = "0 = no, 1 = yes" if binary else "a number"
    while True:
        s = input(f"  {name} ({hint}) [press Enter for {default}]: ").strip()
        if s == "":
            return default
        try:
            v = float(s)
        except ValueError:
            print("  Please type a number.")
            continue
        if binary and v not in (0, 1):
            print("  Please type 0 or 1.")
            continue
        return v


def forecast(values):
    lin_row = pd.DataFrame([[values[f] for f in factors] + [n]], columns=lin_cols)
    base = float(lin.predict(lin_row)[0])
    raw = pd.DataFrame([[scaler.mean_[0]] + [values[f] for f in factors] + [sin_n, cos_n]], columns=feat)
    new_row = scaler.transform(raw).astype(np.float32)
    seq = np.vstack([data[n - W + 1:n], new_row])
    with torch.no_grad():
        r = model(torch.tensor(seq[None, :, :])).item()
    correction = r * scaler.scale_[0] + scaler.mean_[0]
    return base, correction, base + correction


print()
print("Last real day in the file:", df["date"].iloc[-1].date())
print("Average sales of the last 7 days:", round(df["sales"].iloc[-7:].mean(), 1))

previous = None
while True:
    print()
    print("Forecast for", next_date.date(), "(", next_date.day_name(), ")")
    print("Enter tomorrow's plan (just press Enter to accept the value in brackets):")
    values = {f: ask(f) for f in factors}
    base, correction, final = forecast(values)
    print()
    print("  Base forecast (trend + factors):", round(base, 1))
    print("  LSTM correction (recent pattern):", round(correction, 1))
    print("  FORECAST FOR TOMORROW:", round(final, 1), "units")
    if previous is not None:
        print("  Change vs previous forecast:", round(final - previous, 1))
    previous = final
    again = input("\nTry another what-if (for example no promo)? (y/n): ").strip().lower()
    if again != "y":
        break
print("Done.")
