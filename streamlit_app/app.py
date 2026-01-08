import os
import time
from urllib.parse import urlparse

import requests
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")


def _format_currency(value):
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _normalize_api_url(value):
    value = (value or "").strip()
    if not value:
        return DEFAULT_API_URL, True
    parsed = urlparse(value)
    if not parsed.scheme:
        return f"http://{value}", False
    return value, False


def main():
    # set the page configuration
    st.set_page_config(page_title="Insurance predictor", layout="wide", initial_sidebar_state="collapsed")

    # add title and description
    st.title("Insurance Premium Predictor")
    st.markdown(
        """
        This application predicts insurance premiums based on user inputs.
        Please fill in the required fields and click on "Predict" to see the estimated premium.
        """
    )

    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "prediction_time" not in st.session_state:
        st.session_state.prediction_time = None

    # create two columns layout
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.header("Input Parameters")
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        sex = st.selectbox("Sex", options=["male", "female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
        smoker = st.selectbox("Smoker", options=["yes", "no"])
        region = st.selectbox(
            "Region",
            options=["southwest", "southeast", "northwest", "northeast"],
        )
        

        predict = st.button("Predict Insurance Premium", use_container_width=True)

    with col2:
        st.header("Prediction Result")
        if predict:
            input_data = {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region,
            }
            api_endpoint, used_default = _normalize_api_url(api_endpoint)
            if used_default:
                st.info(f"Using default API URL: {api_endpoint}")
            predict_url = f"{api_endpoint.rstrip('/')}/predict"

            with st.spinner("Predicting..."):
                try:
                    response = requests.post(predict_url, json=input_data, timeout=20)
                    response.raise_for_status()
                    prediction = response.json()
                except requests.exceptions.ConnectionError as exc:
                    st.error(
                        "API request failed: connection refused. "
                        "Check that the API server is running and reachable."
                    )
                    st.caption(str(exc))
                    prediction = None
                except requests.exceptions.RequestException as exc:
                    st.error(f"API request failed: {exc}")
                    prediction = None
                except ValueError as exc:
                    st.error(f"Invalid response from API: {exc}")
                    prediction = None

            if prediction is not None:
                st.session_state.prediction = prediction
                st.session_state.prediction_time = time.time()
                st.success("Prediction successful!")

        prediction = st.session_state.prediction
        if prediction:
            predicted_price = prediction.get("predicted_price")
            confidence_interval = prediction.get("confidence_interval")
            prediction_time = prediction.get("prediction_time")

            if predicted_price is None and "prediction" in prediction:
                predicted_price = prediction.get("prediction")

            if predicted_price is not None:
                st.metric("Estimated Premium", _format_currency(predicted_price))
            if confidence_interval:
                low = _format_currency(confidence_interval[0])
                high = _format_currency(confidence_interval[1])
                st.write(f"Confidence interval: {low} - {high}")
            if prediction_time:
                st.caption(f"Prediction time: {prediction_time}")
        else:
            st.info("Submit inputs to see a prediction.")


if get_script_run_ctx() is None:
    print("This app should be run with: streamlit run streamlit_app/app.py")
else:
    main()
                
