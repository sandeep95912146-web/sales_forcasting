import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

WINDOW = 14


class Net(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).squeeze(1)


def validate_upload(raw):
    """Check an uploaded CSV. Returns (clean_dataframe, None) or (None, error_message)."""
    if "date" not in raw.columns or "sales" not in raw.columns:
        return None, "The file needs a 'date' column and a 'sales' column."
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    other = [c for c in df.columns if c != "date"]
    for c in other:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    not_numbers = [c for c in other if df[c].isna().all()]
    if not_numbers:
        return None, "These columns are not numbers, please remove them: " + ", ".join(not_numbers)
    bad_rows = int(df.isna().any(axis=1).sum())
    if bad_rows:
        return None, f"{bad_rows} row(s) have a missing or unreadable value. Please fix them and upload again."
    if df["date"].duplicated().any():
        return None, "Some dates appear more than once. Use one row per day."
    if len(df) < 100:
        return None, f"The file has only {len(df)} rows. At least 100 days are needed."
    return df.sort_values("date").reset_index(drop=True), None


def _prepare(source):
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    factors = [c for c in df.columns if c not in ("date", "sales")]
    n = len(df)
    df["t"] = np.arange(n)
    doy = df["date"].dt.dayofyear
    df["sin"] = np.sin(2 * np.pi * doy / 365)
    df["cos"] = np.cos(2 * np.pi * doy / 365)
    return df, factors, n


def _windows(data, start, stop):
    X, y = [], []
    for i in range(start, stop):
        seq = data[i - WINDOW + 1:i + 1].copy()
        seq[-1, 0] = 0
        X.append(seq)
        y.append(data[i, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _fit_lstm(Xtr, ytr, n_features, epochs):
    model = Net(n_features)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    Xtr, ytr = torch.tensor(Xtr), torch.tensor(ytr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for j in range(0, len(Xtr), 64):
            idx = perm[j:j + 64]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model


def train_model(source, epochs=40):
    """Train the hybrid (linear + LSTM) model on ALL rows. source = CSV path or DataFrame."""
    torch.manual_seed(42)
    np.random.seed(42)
    df, factors, n = _prepare(source)

    lin_cols = factors + ["t"]
    lin = LinearRegression().fit(df[lin_cols], df["sales"])
    df["base"] = lin.predict(df[lin_cols])
    df["resid"] = df["sales"] - df["base"]

    feat = ["resid"] + factors + ["sin", "cos"]
    scaler = StandardScaler().fit(df[feat])
    data = scaler.transform(df[feat]).astype(np.float32)

    X, y = _windows(data, WINDOW, n)
    model = _fit_lstm(X, y, len(feat), epochs)

    return {
        "df": df, "factors": factors, "n": n, "lin": lin, "lin_cols": lin_cols,
        "scaler": scaler, "feat": feat, "data": data, "model": model,
        "next_date": df["date"].iloc[-1] + pd.Timedelta(days=1),
    }


def backtest(source, epochs=40):
    """Train on the first 80% of days, then forecast the last 20% and score both models."""
    torch.manual_seed(42)
    np.random.seed(42)
    df, factors, n = _prepare(source)
    split = int(n * 0.8)

    lin_cols = factors + ["t"]
    lin = LinearRegression().fit(df[lin_cols][:split], df["sales"][:split])
    df["base"] = lin.predict(df[lin_cols])
    df["resid"] = df["sales"] - df["base"]

    feat = ["resid"] + factors + ["sin", "cos"]
    scaler = StandardScaler().fit(df[feat][:split])
    data = scaler.transform(df[feat]).astype(np.float32)

    X, y = _windows(data, WINDOW, n)
    n_train = split - WINDOW
    model = _fit_lstm(X[:n_train], y[:n_train], len(feat), epochs)

    with torch.no_grad():
        r = model(torch.tensor(X[n_train:])).numpy()
    resid_pred = r * scaler.scale_[0] + scaler.mean_[0]
    linear_pred = df["base"].values[split:]
    hybrid_pred = linear_pred + resid_pred
    actual = df["sales"].values[split:]

    def scores(p):
        err = actual - p
        return {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mape": float(np.mean(np.abs(err) / np.abs(actual)) * 100),
        }

    return {
        "dates": df["date"].values[split:], "actual": actual,
        "linear": linear_pred, "hybrid": hybrid_pred,
        "linear_scores": scores(linear_pred), "hybrid_scores": scores(hybrid_pred),
        "n_test": len(actual),
        "internals": {
            "model": model, "scaler": scaler, "lin": lin, "X": X,
            "n_train": n_train, "feat": feat, "split": split,
        },
    }


def is_binary(bundle, name):
    return set(bundle["df"][name].unique()) <= {0, 1}


def default_value(bundle, name):
    if name == "weekend":
        return 1 if bundle["next_date"].dayofweek >= 5 else 0
    if is_binary(bundle, name):
        return 0
    return float(bundle["df"][name].iloc[-1])


def forecast_next(bundle, values):
    """values: dict of factor name -> number for tomorrow.
    Returns (base, lstm_correction, final_forecast)."""
    df, factors, n = bundle["df"], bundle["factors"], bundle["n"]
    lin, lin_cols, scaler = bundle["lin"], bundle["lin_cols"], bundle["scaler"]
    feat, data, model = bundle["feat"], bundle["data"], bundle["model"]
    next_date = bundle["next_date"]

    sin_n = np.sin(2 * np.pi * next_date.dayofyear / 365)
    cos_n = np.cos(2 * np.pi * next_date.dayofyear / 365)

    lin_row = pd.DataFrame([[values[f] for f in factors] + [n]], columns=lin_cols)
    base = float(lin.predict(lin_row)[0])

    raw = pd.DataFrame([[scaler.mean_[0]] + [values[f] for f in factors] + [sin_n, cos_n]], columns=feat)
    new_row = scaler.transform(raw).astype(np.float32)
    seq = np.vstack([data[n - WINDOW + 1:n], new_row])
    with torch.no_grad():
        r = model(torch.tensor(seq[None, :, :])).item()
    correction = r * scaler.scale_[0] + scaler.mean_[0]
    return base, correction, base + correction


# ---------------------------------------------------------------------------
# All four models on one uploaded file (Linear, ARIMA, LSTM, Hybrid LSTM)
# ---------------------------------------------------------------------------
MODEL_NAMES = ["Linear Regression", "ARIMA", "LSTM", "Hybrid LSTM"]


def _scores(actual, pred):
    err = np.asarray(actual) - np.asarray(pred)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE %": float(np.mean(np.abs(err) / np.abs(actual)) * 100),
    }


def _arima_parts(df, factors, split):
    """One-step-ahead ARIMA forecasts for the test days, plus a model fitted on all days."""
    import warnings
    from statsmodels.tsa.arima.model import ARIMA

    cols = [c for c in factors if df[c].iloc[:split].std() > 0]
    endog = df["sales"].values.astype(float)
    exog = df[cols].values.astype(float) if cols else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_train = ARIMA(endog[:split], exog=None if exog is None else exog[:split], order=(1, 1, 1)).fit()
        res_all = res_train.apply(endog, exog=exog)
        test_pred = np.asarray(res_all.fittedvalues)[split:]
        res_full = ARIMA(endog, exog=exog, order=(1, 1, 1)).fit()
    return test_pred, res_full, cols


class _Wrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).unsqueeze(1)


def shap_explanations(df, factors, bt):
    """Same SHAP method as forecast_explain.py, for an uploaded file.
    Returns a table like shap_values.csv (date, forecast, actual, one column per factor)."""
    import shap

    torch.backends.cudnn.enabled = False
    it = bt["internals"]
    lin, model, scaler = it["lin"], it["model"], it["scaler"]
    X, n_train, feat, split = it["X"], it["n_train"], it["feat"], it["split"]
    final, actual = bt["hybrid"], bt["actual"]
    lin_cols = factors + ["t"]

    lin_bg = df[lin_cols][:split].sample(min(200, split), random_state=0)
    lin_sv = shap.LinearExplainer(lin, lin_bg).shap_values(df[lin_cols][split:])
    lin_df = pd.DataFrame(np.array(lin_sv), columns=lin_cols)

    Xtr = torch.tensor(X[:n_train])
    Xte = torch.tensor(X[n_train:])
    wrapped = _Wrap(model).eval()
    background = Xtr[torch.randperm(len(Xtr))[:100]]
    pick = np.linspace(0, len(Xte) - 1, min(60, len(Xte))).astype(int)
    sv = shap.GradientExplainer(wrapped, background).shap_values(Xte[pick])
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.array(sv)
    if sv.ndim == 4:
        sv = sv[..., 0]
    lstm_df = pd.DataFrame(sv.sum(axis=1) * scaler.scale_[0], columns=feat)

    sel = lin_df.iloc[pick].reset_index(drop=True)
    total = pd.DataFrame()
    for f in factors:
        total[f] = sel[f] + lstm_df[f]
    total["trend (time)"] = sel["t"]
    total["season (time of year)"] = lstm_df["sin"] + lstm_df["cos"]
    total["recent sales pattern"] = lstm_df["resid"]

    out = total.copy()
    out.insert(0, "date", pd.to_datetime(df["date"].values[split + pick]))
    out.insert(1, "forecast", np.asarray(final)[pick].round(1))
    out.insert(2, "actual", np.asarray(actual)[pick].round(1))
    number_columns = out.columns.drop("date")
    out[number_columns] = out[number_columns].round(2)
    return out


def run_all_models(df_in, epochs=40, progress=None, explain=True):
    """Score all four models on the last 20% of days, prepare next-day forecasts,
    and (optionally) build SHAP explanations and a question-answering helper.
    progress(fraction, text) is called while it works."""
    torch.manual_seed(42)
    np.random.seed(42)

    def tick(fraction, text):
        if progress is not None:
            progress(fraction, text)

    df, factors, n = _prepare(df_in)
    split = int(n * 0.8)
    n_train = split - WINDOW
    actual = df["sales"].values[split:]
    preds, notes = {}, []

    tick(0.05, "Linear regression and Hybrid LSTM (testing)...")
    bt = backtest(df_in, epochs=epochs)
    preds["Linear Regression"] = bt["linear"]
    preds["Hybrid LSTM"] = bt["hybrid"]

    tick(0.25, "ARIMA...")
    arima_full, arima_cols = None, []
    try:
        preds["ARIMA"], arima_full, arima_cols = _arima_parts(df, factors, split)
    except Exception as exc:
        notes.append(f"ARIMA could not be calculated: {exc}")

    tick(0.35, "LSTM (testing)...")
    feat_p = ["sales"] + factors
    sc_p = StandardScaler().fit(df[feat_p][:split])
    data_p = sc_p.transform(df[feat_p]).astype(np.float32)
    Xp, yp = _windows(data_p, WINDOW, n)
    model_p = _fit_lstm(Xp[:n_train], yp[:n_train], len(feat_p), epochs)
    with torch.no_grad():
        r = model_p(torch.tensor(Xp[n_train:])).numpy()
    preds["LSTM"] = r * sc_p.scale_[0] + sc_p.mean_[0]

    tick(0.50, "LSTM (final model on all days)...")
    sc_full = StandardScaler().fit(df[feat_p])
    data_full = sc_full.transform(df[feat_p]).astype(np.float32)
    Xf, yf = _windows(data_full, WINDOW, n)
    model_full = _fit_lstm(Xf, yf, len(feat_p), epochs)

    tick(0.62, "Hybrid LSTM (final model on all days)...")
    hybrid_bundle = train_model(df_in, epochs=epochs)

    explanations, rag_answer = None, None
    if explain:
        tick(0.72, "SHAP explanations (this is the slowest step)...")
        try:
            explanations = shap_explanations(df, factors, bt)
        except Exception as exc:
            notes.append(f"SHAP explanations could not be calculated: {exc}")
        if explanations is not None:
            try:
                from rag_df import make_answer
                rag_answer = make_answer(explanations)
            except Exception as exc:
                notes.append(f"The question helper could not be built: {exc}")
    tick(1.0, "Done")

    scores = {m: _scores(actual, preds[m]) for m in MODEL_NAMES if m in preds}
    best = min(scores, key=lambda m: scores[m]["MAE"])
    next_date = df["date"].iloc[-1] + pd.Timedelta(days=1)
    test_dates = pd.to_datetime(df["date"].values[split:])
    model_frames = {
        m: pd.DataFrame({"date": test_dates, "actual": actual, "forecast": np.asarray(preds[m])})
        for m in scores
    }

    def predict_next(plan):
        out = {}
        base, _, final = forecast_next(hybrid_bundle, plan)
        out["Linear Regression"] = base
        out["Hybrid LSTM"] = final
        if arima_full is not None:
            try:
                row = np.array([[plan[c] for c in arima_cols]], dtype=float) if arima_cols else None
                out["ARIMA"] = float(np.asarray(arima_full.forecast(steps=1, exog=row))[0])
            except Exception:
                pass
        raw = pd.DataFrame([[sc_full.mean_[0]] + [plan[f] for f in factors]], columns=feat_p)
        new_row = sc_full.transform(raw).astype(np.float32)
        seq = np.vstack([data_full[n - WINDOW + 1:n], new_row])
        with torch.no_grad():
            out["LSTM"] = float(model_full(torch.tensor(seq[None, :, :])).item() * sc_full.scale_[0] + sc_full.mean_[0])
        return {m: out[m] for m in MODEL_NAMES if m in out}

    return {
        "df": df, "clean": df_in, "factors": factors, "next_date": next_date,
        "scores": scores, "best": best, "notes": notes,
        "test_dates": df["date"].values[split:], "actual": actual, "preds": preds,
        "model_frames": model_frames, "explanations": explanations, "rag_answer": rag_answer,
        "n_test": len(actual), "predict_next": predict_next,
    }
