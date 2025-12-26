import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
from src.models_base import run_arima_like_models
from src.benchmarks import naive_forecast
from src.metrics import evaluate_all_metrics
import plotly.graph_objects as go


st.title("📈 Time Series Model Comparison")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.stop()

df = pd.read_csv(uploaded, parse_dates=["Date"]).set_index("Date")
target = st.selectbox("Select target variable", df.columns)

h = st.selectbox("Forecast horizon", [1,7,30])

series = df[target].dropna()
train = series.iloc[:-h]
test = series.iloc[-h:]

st.subheader("Run models")

if st.button("Run"):
    results = {}

    results["Naive"] = naive_forecast(train, test)

    arima_models = run_arima_like_models(train, None, test)
    results.update(arima_models)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train, name="Train"))
    fig.add_trace(go.Scatter(x=test.index, y=test, name="Test"))

    for name, pred in results.items():
        fig.add_trace(go.Scatter(x=test.index, y=pred, name=name))

    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    table = {}
    for name, pred in results.items():
        table[name] = evaluate_all_metrics(test, pred, train)

    st.dataframe(pd.DataFrame(table).T)
