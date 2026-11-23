# Explainable Sales Forecasting

A Streamlit dashboard for exploring daily sales, comparing forecasting models, and understanding the factors behind predictions.

## What it does

- Upload a daily sales CSV and compare Linear Regression, ARIMA, LSTM, and Hybrid LSTM forecasts.
- Review model error using MAE, RMSE, and MAPE. Scores use the final 20% of dates as a test period.
- Explore sales charts, forecast comparisons, and SHAP explanations.
- Forecast the next day using selected values for numeric factors such as promotions or holidays.
- Ask questions about the explanations.

## Requirements

- Windows 10 or 11
- Python 3.10 or newer
- Git

## Setup and run on Windows

Open PowerShell in the project folder and run:

~~~powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
~~~

Streamlit will show a local address, usually http://localhost:8501. Open it in your browser, then create an account or sign in when prompted.

If PowerShell blocks activation, run:

~~~powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
~~~

## Upload your sales data

Use a CSV with one row per day and these required columns:

| Column | Meaning | Example |
| --- | --- | --- |
| date | Date for the row | 2025-01-01 |
| sales | Numeric sales for that date | 1250 |

You may add numeric columns such as promo, holiday, or price; the app uses them as factors. Use at least 100 rows. Dates must be unique, and values must not be missing. Uploads are limited to 5 MB and 5,000 rows.

Example:

~~~csv
date,sales,promo
2025-01-01,1250,0
2025-01-02,1480,1
~~~

The uploaded file is used for the current app session and does not replace the project's saved data.

## Project data

The built-in Rossmann dashboard expects generated files such as sales.csv and shap_values.csv. If they are missing, the app displays an error; use the project's preparation scripts to create them. You can still try the upload feature with your own CSV.

## Understanding the results

Lower MAE, RMSE, and MAPE scores mean smaller errors on the test period. SHAP values show how the model used factors; they do not prove a factor caused sales to change. Forecasts are estimates, not guarantees.

## Main files

- app.py — Streamlit dashboard
- tomorrow_model.py — model training, backtesting, and upload validation
- requirements.txt — Python packages
- sales_sample.csv — sample sales data
