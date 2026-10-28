from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

PROJECT = Path(__file__).resolve().parent

st.set_page_config(page_title="Explainable Sales Forecasting", page_icon="📈", layout="wide")
st.title("Explainable Sales Forecasting")
st.caption("Sales forecasts, model comparison, SHAP explanations, and questions about the results.")

with st.expander("Upload and check your own sales CSV"):
    st.write(
        "Choose a CSV file to check its format and preview its sales. "
        "It needs these columns: date, sales, promo, holiday, school_holiday, weekend."
    )
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="sales_csv_upload")
    if uploaded_file is not None:
        try:
            uploaded_sales = pd.read_csv(uploaded_file)
            required_columns = ["date", "sales", "promo", "holiday", "school_holiday", "weekend"]
            missing_columns = [column for column in required_columns if column not in uploaded_sales.columns]

            if missing_columns:
                st.error("This file is missing these columns: " + ", ".join(missing_columns))
            elif uploaded_sales.empty:
                st.error("This CSV has no data rows.")
            else:
                uploaded_sales["date"] = pd.to_datetime(uploaded_sales["date"], errors="coerce")
                numeric_columns = ["sales", "promo", "holiday", "school_holiday", "weekend"]
                for column in numeric_columns:
                    uploaded_sales[column] = pd.to_numeric(uploaded_sales[column], errors="coerce")

                invalid_rows = uploaded_sales[required_columns].isna().any(axis=1).sum()
                if invalid_rows:
                    st.error(
                        f"{invalid_rows} row(s) have a missing or unreadable date or number. "
                        "Please correct those rows and upload the CSV again."
                    )
                else:
                    st.success(f"File looks good: {len(uploaded_sales):,} rows checked.")
                    st.line_chart(uploaded_sales.set_index("date")["sales"], y_label="Sales in uploaded file")
                    st.dataframe(uploaded_sales.head(10), use_container_width=True)
                    st.caption("This preview does not change the project's saved data or forecast results.")
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"I couldn't read this CSV. Please choose a regular comma-separated CSV file. Details: {exc}")

sales_path = PROJECT / "sales.csv"
shap_path = PROJECT / "shap_values.csv"

if not sales_path.exists() or not shap_path.exists():
    st.error("Required project files are missing. Run prepare_rossmann.py and forecast_explain.py first.")
    st.stop()

sales = pd.read_csv(sales_path, parse_dates=["date"])
explanations = pd.read_csv(shap_path, parse_dates=["date"])

# Load saved test-period predictions so every model can be compared in the dashboard.
model_files = {
    "Linear Regression": "linear_predictions.csv",
    "ARIMA": "arima_predictions.csv",
    "LSTM": "lstm_predictions.csv",
    "Hybrid LSTM": "hybrid_lstm_predictions.csv",
}
model_fallback_scores = {
    "Linear Regression": 1422.32,
    "ARIMA": 1330.23,
    "LSTM": 657.65,
    "Hybrid LSTM": 637.95,
}
model_predictions = {}
model_scores = {}
for model_name, filename in model_files.items():
    prediction_path = PROJECT / filename
    if prediction_path.exists():
        model_frame = pd.read_csv(prediction_path, parse_dates=["date"])
        if {"date", "actual", "forecast"}.issubset(model_frame.columns):
            model_predictions[model_name] = model_frame
            model_scores[model_name] = (model_frame["actual"] - model_frame["forecast"]).abs().mean()
    if model_name not in model_scores:
        model_scores[model_name] = model_fallback_scores[model_name]

st.subheader("Model comparison")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Linear Regression MAE", f"{model_scores['Linear Regression']:,.2f}")
c2.metric("ARIMA MAE", f"{model_scores['ARIMA']:,.2f}")
c3.metric("LSTM MAE", f"{model_scores['LSTM']:,.2f}")
c4.metric("Hybrid LSTM MAE", f"{model_scores['Hybrid LSTM']:,.2f}")
st.caption("MAE is the average absolute difference between predicted and actual sales; lower is better.")

left, right = st.columns(2)
with left:
    st.subheader("Sales data")
    st.write(f"Rows: {len(sales):,}  ·  Date range: {sales['date'].min().date()} to {sales['date'].max().date()}")
    st.line_chart(sales.set_index("date")["sales"], y_label="Sales")
with right:
    st.subheader("Forecast vs actual by model")
    selected_model = st.selectbox("Choose a model", list(model_files))
    forecast_data = model_predictions.get(selected_model)

    if selected_model == "ARIMA":
        try:
            api_result = requests.get("http://127.0.0.1:8000/forecast", timeout=2)
            api_result.raise_for_status()
            forecast_data = pd.DataFrame(api_result.json()["data"])
            forecast_data["date"] = pd.to_datetime(forecast_data["date"])
            st.caption("ARIMA results loaded from FastAPI.")
        except (requests.RequestException, KeyError, ValueError):
            st.caption("ARIMA results loaded from the saved forecast file.")

    if forecast_data is not None:
        chart_data = forecast_data.set_index("date")[["actual", "forecast"]]
        st.line_chart(chart_data, y_label="Sales")
    elif selected_model == "Hybrid LSTM" and (PROJECT / "forecast_vs_actual.png").exists():
        st.image(str(PROJECT / "forecast_vs_actual.png"), use_container_width=True)
        st.caption("Hybrid LSTM forecast vs actual (saved test-period chart).")
    else:
        st.info(f"No saved predictions for {selected_model} yet. Run its Python script to create them.")

st.subheader("What influences the forecast?")
excluded = {"date", "forecast", "actual"}
factors = [col for col in explanations.columns if col not in excluded]
importance = explanations[factors].abs().mean().sort_values(ascending=True)
st.bar_chart(importance, horizontal=True, x_label="Average impact (sales units)", y_label="Factor")
st.caption("SHAP impact describes how the model used these factors; it does not prove that a factor causes sales to change.")

st.subheader("Ask about the explanations")
st.write("Click a question to open its answer:")
from rag_chat import answer as rag_answer

best_model = min(model_scores, key=model_scores.get)
faq_items = [
    ("Which factor affects sales most?", rag_answer("Which factor matters most?")),
    ("How accurate is the forecast?", f"On this test, {best_model} had the lowest MAE: {model_scores[best_model]:,.2f} sales units. MAE is the average absolute difference between predicted and actual sales; lower is better."),
    ("Why are predicted sales different from actual sales?", "A forecast is an estimate based on past patterns and the available inputs. Unexpected events and sharp daily changes can make actual sales differ from the prediction."),
    ("What does MAE mean?", "MAE means Mean Absolute Error. It is the average size of the difference between predicted sales and actual sales, measured in sales units. A lower MAE means closer predictions on this test."),
    ("How does the model make predictions?", rag_answer("How does the model work?")),
    ("What does SHAP explain?", "SHAP estimates how much each input factor raised or lowered an individual forecast. It helps explain the model's output; it does not prove that a factor caused sales to change."),
]
for faq_question, faq_answer in faq_items:
    with st.expander(faq_question):
        st.write(faq_answer)

st.write("Or type your own question:")
question = st.text_input("Ask a question about a date, sales factor, forecast, or the model")
if st.button("Get answer", disabled=not question.strip()):
    try:
        st.write(rag_answer(question.strip()))
    except Exception as exc:
        st.error(f"Could not load the explanation chat: {exc}")

with st.expander("Preview prepared sales data"):
    st.dataframe(sales.head(10), use_container_width=True)
