import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.optimization_engine import run_stochastic_bess_optimization

st.set_page_config(page_title="Grid-Nexus BESS Optimizer", layout="wide")

st.title("⚡ Grid-Nexus: Risk-Averse Stochastic BESS Optimizer")
st.markdown("Advanced MILP optimization engine for Battery Energy Storage Systems in the German Day-Ahead electricity market.")

# سایدبار تنظیمات
st.sidebar.header("Configuration Parameters")
beta_input = st.sidebar.slider("Risk Aversion Parameter (Beta)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
n_scenarios_input = st.sidebar.slider("Number of Price Scenarios", min_value=20, max_value=150, value=50, step=10)

if st.sidebar.button("Run Optimization Model"):
    with st.spinner("Solving stochastic optimization problem..."):
        results = run_stochastic_bess_optimization(n_scenarios=n_scenarios_input, beta=beta_input)
        
    st.success("Optimization completed successfully!")
    
    # نمایش متریک‌ها
    col1, col2, col3 = st.columns(3)
    col1.metric("Optimization Status", results["status"])
    col2.metric("Expected Daily Profit", f"€{results['expected_profit']:.2f}")
    col3.metric("CVaR Risk Metric", f"€{results['cvar_risk']:.2f}")
    
    # رسم نمودار سناریوها
    st.subheader("Generated Market Price Scenarios (Uncertainty)")
    fig, ax = plt.subplots(figsize=(10, 4))
    for s in range(n_scenarios_input):
        ax.plot(results["scenario_prices"][s], color="gray", alpha=0.25)
    ax.set_title("German Day-Ahead Price Scenarios")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Price (€/MWh)")
    ax.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)
