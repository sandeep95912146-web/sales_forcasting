# Explainable Sales Forecasting

A Streamlit app for comparing sales forecasting models and exploring their predictions.

## Features

- Upload a sales CSV with `date` and `sales` columns
- Compare models using MAE
- View forecast explanations using SHAP
- Forecast sales for the next day
- Ask questions about the results

## Run on Windows

Open a terminal in this project folder and run:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app.py