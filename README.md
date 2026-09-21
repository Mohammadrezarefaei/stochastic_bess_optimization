# ⚡ Risk-Averse Stochastic BESS Optimizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stochasticbeappptimization-eg2dxw76la37unkz7kfasm.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced Mixed-Integer Linear Programming (MILP) optimization engine designed for Battery Energy Storage Systems (BESS) market arbitrage under price uncertainty in the German Day-Ahead electricity market (EPEX SPOT), featuring stochastic scenario generation and Conditional Value-at-Risk (CVaR) risk-averse decision-making.

---

## 🚀 Live Demo
Experience the interactive web application deployed on Streamlit Cloud: 
👉 **[Stochastic BESS Optimizer App](https://stochasticbeappptimization-eg2dxw76la37unkz7kfasm.streamlit.app/)**

---

## 📊 Visuals & Outputs

### German Day-Ahead Price Scenarios (Uncertainty Modeling)
![Market Prices](outputs/market_prices_chart.png)

### Pareto Frontier: Expected Profit vs. Financial Risk (CVaR)
![Pareto Frontier](outputs/pareto_frontier_chart.png)

---

## 🛠️ Key Features
- **Stochastic Optimization Engine:** Models price volatility in the German electricity market by generating multi-scenario probabilistic price distributions (handling high peaks and negative pricing).
- **Risk-Averse Decision Making (CVaR):** Integrates Conditional Value-at-Risk (CVaR) constraints to protect investments against tail-end market extremes and severe volatility.
- **Physical Battery Constraints:** Strictly manages energy capacity, maximum charging/discharging C-rates, round-trip efficiencies, and State of Charge (SoC) limits.
- **Interactive Web Dashboard:** A clean, user-friendly Streamlit interface to adjust risk aversion parameters ($\beta$), scale scenarios, and analyze dispatch strategies live.
- **Automated Testing Suite:** Robust unit testing using `pytest` and automated configuration to ensure mathematical and solver reliability.

---

## 📂 Repository Structure
```text
stochastic_bess_optimization/
├── outputs/
│   ├── market_prices_chart.png
│   ├── pareto_frontier_chart.png
│   ├── optimization_results.csv
│   └── scenario_prices.csv
├── src/
│   └── optimization_engine.py
├── tests/
│   └── test_optimizer.py
├── app.py
├── requirements.txt
├── pytest.ini
└── README.md
