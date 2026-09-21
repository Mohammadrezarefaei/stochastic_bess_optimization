import pulp
import pytest

def test_pulp_solver_availability():
    """تست بررسی دسترسی به سالور CBC در پکیج PuLP"""
    solver = pulp.PULP_CBC_CMD(msg=0)
    assert solver.available(), "CBC solver is not available or not properly installed!"

def test_basic_milp_optimization():
    """تست اعتبارسنجی یک مدل ساده برنامه‌ریزی ریاضی برای اطمینان از صحت عملکرد موتور بهینه‌سازی"""
    model = pulp.LpProblem("Sanity_Check", pulp.LpMaximize)
    
    # متغیر تست
    x = pulp.LpVariable("x", lowBound=0, upBound=10)
    
    # تابع هدف ساده
    model += x, "Maximize_X"
    
    # حل مدل
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # بررسی وضعیت خروجی
    assert pulp.LpStatus[model.status] == "Optimal"
    assert x.value() == 10.0
