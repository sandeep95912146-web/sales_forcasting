import os
import sys
import pandas as pd

# Leave STORE as None to pick the store that trades on the most days.
# Or type a store number here, for example STORE = 262
STORE = None

if not os.path.exists("rossmann_train.csv"):
    sys.exit("rossmann_train.csv not found. Save Kaggle's train.csv in this folder with that name.")

raw = pd.read_csv("rossmann_train.csv", low_memory=False, dtype={"StateHoliday": str})
raw["Date"] = pd.to_datetime(raw["Date"])

if STORE is None:
    open_days = raw[raw["Open"] == 1].groupby("Store").size()
    STORE = int(open_days.idxmax())
print("Using store number:", STORE)

s = raw[(raw["Store"] == STORE) & (raw["Open"] == 1) & (raw["Sales"] > 0)]
s = s.sort_values("Date")
if s.empty:
    sys.exit(f"No open days found for store {STORE}. Try another store number.")

out = pd.DataFrame({
    "date": s["Date"],
    "sales": s["Sales"],
    "promo": s["Promo"].astype(int),
    "holiday": (s["StateHoliday"].astype(str) != "0").astype(int),
    "school_holiday": s["SchoolHoliday"].astype(int),
    "weekend": (s["DayOfWeek"] >= 6).astype(int),
})

# keep the practice data safe (only the first time)
if os.path.exists("sales.csv") and not os.path.exists("sales_sample.csv"):
    os.rename("sales.csv", "sales_sample.csv")
    print("Old practice data saved as sales_sample.csv")

out.to_csv("sales.csv", index=False)

print("Rows:", len(out))
print("From", out["date"].min().date(), "to", out["date"].max().date())
print("Promo days:", int(out["promo"].sum()),
      "| Holiday days:", int(out["holiday"].sum()),
      "| School holiday days:", int(out["school_holiday"].sum()),
      "| Weekend days:", int(out["weekend"].sum()))
print("Average sales:", round(out["sales"].mean(), 1))
print("New sales.csv is ready.")
