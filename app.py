from pathlib import Path

import pandas as pd
import requests
import streamlit as st

PROJECT = Path(__file__).resolve().parent

st.set_page_config(page_title="Explainable Sales Forecasting", page_icon="📈", layout="wide")

# ---------------- Security: sign-up / log-in, upload limits ----------------
MAX_UPLOAD_MB = 5
MAX_UPLOAD_ROWS = 5000

from auth import logout_button, require_login

current_user = require_login()
st.sidebar.write(f"Signed in as **{current_user}**")
with st.sidebar:
    logout_button()


def show_range(value, margin):
    """Shows the likely range (forecast +/- usual error) and a wider safe range."""
    low, high = max(value - margin, 0), value + margin
    safe_low, safe_high = max(value - 2 * margin, 0), value + 2 * margin
    st.info(f"**Likely range for tomorrow:** {low:,.0f} to {high:,.0f} units")
    st.write(f"Safe range (very likely): {safe_low:,.0f} to {safe_high:,.0f} units")
    st.caption(
        f"The range uses the model's usual error on past test days (about {margin:,.0f} units). "
        "It is a guide, not a guarantee."
    )
st.title("Explainable Sales Forecasting")
st.caption("Sales forecasts, model comparison, SHAP explanations, and questions about the results.")

# =====================================================================
# Use your own sales CSV: the whole page below switches to the uploaded file
# =====================================================================
st.header("Use your own sales CSV")
st.write(
    "Upload a CSV of your daily sales. It needs a **date** column and a **sales** column, with one row per day "
    "(at least 100 days, a year or more is better). Any other number columns, such as promo, holiday or price, "
    "are used as factors. The page tests four models on your file, and **everything below** (model comparison, "
    "charts, SHAP, tomorrow's forecast, questions) then shows results for your file."
)
st.caption(f"Limits: {MAX_UPLOAD_MB} MB and {MAX_UPLOAD_ROWS:,} days per file.")
# Hide Streamlit's default "200MB per file" text, so only our own limit is shown.
# The real limit is still checked in the code below.
st.markdown(
    """<style>
    [data-testid="stFileUploaderDropzoneInstructions"] small,
    [data-testid="stFileUploaderDropzone"] small { display: none; }
    </style>""",
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader("Choose your sales CSV file", type=["csv"], key="sales_csv_upload")
uploaded_result = None

if uploaded_file is not None:
    try:
        from tomorrow_model import run_all_models, validate_upload

        uploaded_file.seek(0)
        raw_upload = None
        if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"This file is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller file.")
        else:
            try:
                raw_upload = pd.read_csv(uploaded_file)
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
                st.error(f"I couldn't read this CSV. Please choose a regular comma-separated file. Details: {exc}")
        if raw_upload is not None and len(raw_upload) > MAX_UPLOAD_ROWS:
            st.error(f"This file has {len(raw_upload):,} rows. The limit is {MAX_UPLOAD_ROWS:,} rows (days).")
            raw_upload = None

        if raw_upload is not None:
            clean_upload, problem = validate_upload(raw_upload)
            if problem:
                st.error(problem)
            else:
                factor_names = [c for c in clean_upload.columns if c not in ("date", "sales")]
                st.success(
                    f"File looks good: {len(clean_upload):,} days, "
                    f"{clean_upload['date'].min().date()} to {clean_upload['date'].max().date()}. "
                    f"Factors found: {', '.join(factor_names) if factor_names else 'none'}."
                )
                if st.button("Run all models on this file"):
                    progress_bar = st.progress(0.0, text="Starting...")

                    def update_progress(fraction, text):
                        progress_bar.progress(min(float(fraction), 1.0), text=text)

                    outcome = run_all_models(clean_upload, progress=update_progress)
                    progress_bar.empty()
                    st.session_state["upload_result"] = {"name": uploaded_file.name, "result": outcome}

                saved = st.session_state.get("upload_result")
                if saved is not None and saved["name"] == uploaded_file.name:
                    uploaded_result = saved["result"]
                    for note in uploaded_result["notes"]:
                        st.warning(note)
    except Exception as exc:
        st.error(f"Could not run the models on this file: {exc}")

use_uploaded = False
if uploaded_result is not None:
    view_choice = st.radio(
        "Show the results below for:",
        ["My uploaded file", "Project data (Rossmann)"],
        horizontal=True,
        key="data_view",
    )
    use_uploaded = view_choice == "My uploaded file"
    if use_uploaded:
        st.success("The page below now shows results for your uploaded file.")

st.divider()

# =====================================================================
# Pick the data source for the dashboard
# =====================================================================
sales_path = PROJECT / "sales.csv"
shap_path = PROJECT / "shap_values.csv"

model_files = {
    "Linear Regression": "linear_predictions.csv",
    "ARIMA": "arima_predictions.csv",
    "LSTM": "lstm_predictions.csv",
    "Hybrid LSTM": "hybrid_lstm_predictions.csv",
}
model_predictions = {}
model_scores = {}

if use_uploaded:
    sales = uploaded_result["clean"]
    explanations = uploaded_result["explanations"]
    model_predictions = uploaded_result["model_frames"]
    model_scores = {name: values["MAE"] for name, values in uploaded_result["scores"].items()}
    rag_answer = uploaded_result["rag_answer"]
else:
    if not sales_path.exists() or not shap_path.exists():
        st.error("Required project files are missing. Run your prepare script and forecast_explain.py first.")
        st.stop()
    sales = pd.read_csv(sales_path, parse_dates=["date"])
    explanations = pd.read_csv(shap_path, parse_dates=["date"])
    model_fallback_scores = {
        "Linear Regression": 1422.32,
        "ARIMA": 1330.23,
        "LSTM": 657.65,
        "Hybrid LSTM": 637.95,
    }
    for model_name, filename in model_files.items():
        prediction_path = PROJECT / filename
        if prediction_path.exists():
            model_frame = pd.read_csv(prediction_path, parse_dates=["date"])
            if {"date", "actual", "forecast"}.issubset(model_frame.columns):
                model_predictions[model_name] = model_frame
                model_scores[model_name] = (model_frame["actual"] - model_frame["forecast"]).abs().mean()
        if model_name not in model_scores:
            model_scores[model_name] = model_fallback_scores[model_name]
    from rag_chat import answer as rag_answer

best_model = min(model_scores, key=model_scores.get)

# =====================================================================
# Model comparison
# =====================================================================
st.subheader("Model comparison")
metric_columns = st.columns(len(model_scores))
for metric_column, (name, value) in zip(metric_columns, model_scores.items()):
    metric_column.metric(f"{name} MAE", f"{value:,.2f}")
st.caption(
    "MAE is the average absolute difference between predicted and actual sales; lower is better. "
    f"Lowest error here: {best_model}."
)

left, right = st.columns(2)
with left:
    st.subheader("Sales data")
    st.write(f"Rows: {len(sales):,}  ·  Date range: {sales['date'].min().date()} to {sales['date'].max().date()}")
    st.line_chart(sales.set_index("date")["sales"], y_label="Sales")
with right:
    st.subheader("Forecast vs actual by model")
    model_options = list(model_predictions.keys()) if use_uploaded else list(model_files)
    selected_model = st.selectbox("Choose a model", model_options)
    forecast_data = model_predictions.get(selected_model)

    if not use_uploaded and selected_model == "ARIMA":
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
    elif not use_uploaded and selected_model == "Hybrid LSTM" and (PROJECT / "forecast_vs_actual.png").exists():
        st.image(str(PROJECT / "forecast_vs_actual.png"), use_container_width=True)
        st.caption("Hybrid LSTM forecast vs actual (saved test-period chart).")
    else:
        st.info(f"No saved predictions for {selected_model} yet.")

# =====================================================================
# What influences the forecast
# =====================================================================
st.subheader("What influences the forecast?")
if explanations is not None:
    excluded = {"date", "forecast", "actual"}
    factors = [col for col in explanations.columns if col not in excluded]
    importance = explanations[factors].abs().mean().sort_values(ascending=True)
    st.bar_chart(importance, horizontal=True, x_label="Average impact (sales units)", y_label="Factor")
    st.caption("SHAP impact describes how the model used these factors; it does not prove that a factor causes sales to change.")
else:
    st.info("The SHAP explanation could not be calculated for this file.")

# =====================================================================
# Check a particular day
# =====================================================================
st.subheader("Check a particular day")
if model_predictions:
    saved_dates = sorted(set().union(*[set(frame["date"].dt.date) for frame in model_predictions.values()]))
    picked_day = st.date_input(
        "Pick a date from the test period",
        value=saved_dates[0],
        min_value=saved_dates[0],
        max_value=saved_dates[-1],
        key="check_day",
    )

    day_rows = []
    for model_name, frame in model_predictions.items():
        match = frame[frame["date"].dt.date == picked_day]
        if not match.empty:
            actual_sales = float(match["actual"].iloc[0])
            predicted_sales = float(match["forecast"].iloc[0])
            day_rows.append({
                "Model": model_name,
                "Forecast": round(predicted_sales, 1),
                "Actual": round(actual_sales, 1),
                "Error": round(predicted_sales - actual_sales, 1),
                "Error %": round(abs(predicted_sales - actual_sales) / actual_sales * 100, 1),
            })

    if day_rows:
        day_table = pd.DataFrame(day_rows)
        best_row = day_table.loc[day_table["Error %"].idxmin()]
        st.write(
            f"Actual sales on **{picked_day}**: **{day_table['Actual'].iloc[0]:,.0f}** units. "
            f"Closest forecast: **{best_row['Model']}** ({best_row['Error %']}% off)."
        )
        st.dataframe(day_table, hide_index=True, use_container_width=True)
        compare = {row["Model"]: row["Forecast"] for row in day_rows}
        compare["Actual"] = day_rows[0]["Actual"]
        st.bar_chart(pd.Series(compare, name="Sales"), y_label="Sales (units)")
    else:
        st.info("There are no saved predictions for that date. Pick another date.")

    if explanations is not None:
        day_explanation = explanations[explanations["date"].dt.date == picked_day]
        if not day_explanation.empty:
            st.write("Why the model forecast this (SHAP effect of each factor, in sales units):")
            effect_columns = [c for c in explanations.columns if c not in ("date", "forecast", "actual")]
            st.bar_chart(day_explanation[effect_columns].iloc[0], horizontal=True, x_label="Effect on forecast (sales units)")
            if rag_answer is not None:
                st.write(rag_answer(f"Why was the forecast high on {picked_day}?"))
        else:
            st.caption("The SHAP explanation is saved for 60 sample days only. Pick a different date to see it.")
else:
    st.info("No saved model predictions were found, so a single day cannot be compared yet.")

# =====================================================================
# Tomorrow's forecast
# =====================================================================
st.subheader("Tomorrow's forecast")

if use_uploaded:
    from tomorrow_model import default_value, is_binary

    res = uploaded_result
    st.success(
        f"Best model on your data: **{res['best']}**. It had the lowest error on the last "
        f"{res['n_test']} days, which no model trained on."
    )
    st.write(
        f"**Plan for {res['next_date'].date()} ({res['next_date'].day_name()})** - "
        "choose what you expect tomorrow:"
    )
    plan = {}
    if res["factors"]:
        plan_columns = st.columns(len(res["factors"]))
        for plan_column, factor_name in zip(plan_columns, res["factors"]):
            default = default_value(res, factor_name)
            with plan_column:
                if is_binary(res, factor_name):
                    plan[factor_name] = st.selectbox(
                        factor_name, [0, 1], index=int(default),
                        format_func=lambda v: "Yes" if v == 1 else "No",
                        key="store_plan_" + factor_name,
                    )
                else:
                    plan[factor_name] = st.number_input(
                        factor_name, value=float(default), key="store_plan_" + factor_name
                    )

    tomorrow_values = res["predict_next"](plan)
    table_rows = []
    for model_name, error_scores in res["scores"].items():
        table_rows.append({
            "Model": model_name + ("  (best)" if model_name == res["best"] else ""),
            "MAE": round(error_scores["MAE"], 1),
            "RMSE": round(error_scores["RMSE"], 1),
            "MAPE %": round(error_scores["MAPE %"], 2),
            f"Forecast for {res['next_date'].date()}": round(tomorrow_values.get(model_name, float("nan")), 0),
        })
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
    st.caption("MAE, RMSE and MAPE show the error on the last 20% of days. Lower is better.")

    model_list = list(res["scores"].keys())
    chosen_model = st.selectbox(
        "Choose the model you want to use", model_list, index=model_list.index(res["best"]),
        key="chosen_model",
    )
    st.metric(
        f"{chosen_model} forecast for {res['next_date'].date()}",
        f"{tomorrow_values[chosen_model]:,.0f} units",
    )
    show_range(tomorrow_values[chosen_model], res["scores"][chosen_model]["MAE"])
    st.caption(
        "These forecasts are estimates from past patterns and the plan you chose. "
        "Your file is used only for this session and does not change the project's saved data."
    )
else:
    @st.cache_resource(show_spinner=False)
    def load_tomorrow_model(csv_path, file_time):
        from tomorrow_model import train_model
        return train_model(csv_path)

    try:
        from tomorrow_model import default_value, forecast_next, is_binary

        with st.spinner("Preparing the forecast model. The first time takes about a minute..."):
            tm = load_tomorrow_model(str(sales_path), sales_path.stat().st_mtime)

        tomorrow = tm["next_date"]
        st.write(
            f"Forecast for **{tomorrow.date()} ({tomorrow.day_name()})**, "
            f"the day after the last date in the data ({tm['df']['date'].iloc[-1].date()}). "
            "Choose the plan for that day:"
        )

        input_columns = st.columns(len(tm["factors"]))
        plan = {}
        for column, factor_name in zip(input_columns, tm["factors"]):
            default = default_value(tm, factor_name)
            with column:
                if is_binary(tm, factor_name):
                    plan[factor_name] = st.selectbox(
                        factor_name,
                        [0, 1],
                        index=int(default),
                        format_func=lambda v: "Yes" if v == 1 else "No",
                        key="tomorrow_" + factor_name,
                    )
                else:
                    plan[factor_name] = st.number_input(
                        factor_name, value=float(default), key="tomorrow_" + factor_name
                    )

        if st.button("Forecast tomorrow"):
            base_value, correction, final_value = forecast_next(tm, plan)
            recent_average = tm["df"]["sales"].iloc[-7:].mean()

            m1, m2, m3 = st.columns(3)
            m1.metric("Forecast for tomorrow", f"{final_value:,.0f} units",
                      f"{final_value - recent_average:+,.0f} vs last 7-day average")
            m2.metric("Base (trend + factors)", f"{base_value:,.0f}")
            m3.metric("LSTM correction", f"{correction:+,.1f}")
            show_range(final_value, float(model_scores["Hybrid LSTM"]))

            if "promo" in plan:
                other_plan = dict(plan)
                other_plan["promo"] = 0 if plan["promo"] == 1 else 1
                _, _, other_value = forecast_next(tm, other_plan)
                label = "with a promotion" if other_plan["promo"] == 1 else "without a promotion"
                st.info(
                    f"What if {label}? The forecast would be {other_value:,.0f} units "
                    f"({other_value - final_value:+,.0f})."
                )
            st.caption(
                "The model uses the last 14 days of sales and your plan for tomorrow. "
                "It is an estimate, not a guarantee."
            )
    except Exception as exc:
        st.error(f"Tomorrow's forecast could not be loaded: {exc}")

# =====================================================================
# Ask about the explanations
# =====================================================================
st.subheader("Ask about the explanations")
st.write("Click a question to open its answer:")

faq_items = []
if rag_answer is not None:
    faq_items.append(("Which factor affects sales most?", rag_answer("Which factor matters most?")))
faq_items += [
    ("How accurate is the forecast?", f"On this test, {best_model} had the lowest MAE: {model_scores[best_model]:,.2f} sales units. MAE is the average absolute difference between predicted and actual sales; lower is better."),
    ("Why are predicted sales different from actual sales?", "A forecast is an estimate based on past patterns and the available inputs. Unexpected events and sharp daily changes can make actual sales differ from the prediction."),
    ("What does MAE mean?", "MAE means Mean Absolute Error. It is the average size of the difference between predicted sales and actual sales, measured in sales units. A lower MAE means closer predictions on this test."),
]
if rag_answer is not None:
    faq_items.append(("How does the model make predictions?", rag_answer("How does the model work?")))
faq_items.append(("What does SHAP explain?", "SHAP estimates how much each input factor raised or lowered an individual forecast. It helps explain the model's output; it does not prove that a factor caused sales to change."))
for faq_question, faq_answer in faq_items:
    with st.expander(faq_question):
        st.write(faq_answer)

st.write("Or type your own question:")
question = st.text_input("Ask a question about a date, sales factor, forecast, or the model")
if st.button("Get answer", disabled=not question.strip() or rag_answer is None):
    try:
        st.write(rag_answer(question.strip()))
    except Exception as exc:
        st.error(f"Could not load the explanation chat: {exc}")

with st.expander("Preview prepared sales data"):
    st.dataframe(sales.head(10), use_container_width=True)
