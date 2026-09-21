import pulp
import pytest
from src.optimization_engine import run_stochastic_bess_optimization

def test_pulp_solver_availability():
    """بررسی در دسترس بودن سالور CBC"""
    solver = pulp.PULP_CBC_CMD(msg=0)
    assert solver.available(), "CBC solver is not available or installed!"

def test_optimization_execution():
    """تست اجرای موفقیت‌آمیز مدل بهینه‌سازی استخلاصی"""
    res = run_stochastic_bess_optimization(n_scenarios=10, beta=0.1)
    assert res["status"] == "Optimal"
    assert res["expected_profit"] > 0.0
