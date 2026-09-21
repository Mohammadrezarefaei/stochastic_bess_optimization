import numpy as np
import pandas as pd
import pulp

def run_stochastic_bess_optimization(n_scenarios=100, beta=0.5):
    """
    اجرای مدل بهینه‌سازی تصادفی باتری تحت عدم قطعیت قیمت بازار آلمان با رویکرد CVaR
    """
    np.random.seed(42)
    n_hours = 24
    
    # الگوی پایه قیمت Day-Ahead بازار آلمان (24 ساعت)
    base_prices = np.array([
        45, 40, 38, 36, 40, 55,  # 00:00 - 05:00
        75, 95, 90, 70, 45, 20,  # 06:00 - 11:00
        15, 25, 60, 85, 110, 120,# 12:00 - 17:00
        115, 95, 75, 60, 50, 48  # 18:00 - 23:00
    ])
    
    # تولید سناریوهای نوسانی
    noise = np.random.normal(loc=1.0, scale=0.45, size=(n_scenarios, n_hours))
    scenario_prices = base_prices * noise
    scenario_prices = np.clip(scenario_prices, -50.0, 400.0)
    
    # مشخصات فیزیکی BESS
    capacity_mwh = 4.0
    max_power_mw = 2.0
    eta_ch = 0.95
    eta_dis = 0.95
    soc_min = 0.1 * capacity_mwh
    soc_max = 0.9 * capacity_mwh
    initial_soc = 0.5 * capacity_mwh
    alpha = 0.95
    
    hours = range(n_hours)
    scenarios = range(n_scenarios)
    
    # تعریف مدل PuLP
    prob = pulp.LpProblem("Stochastic_BESS_Arbitrage_CVaR", pulp.LpMaximize)
    
    # متغیرهای تصمیم
    p_ch = pulp.LpVariable.dicts("P_ch", ((s, t) for s in scenarios for t in hours), lowBound=0, upBound=max_power_mw)
    p_dis = pulp.LpVariable.dicts("P_dis", ((s, t) for s in scenarios for t in hours), lowBound=0, upBound=max_power_mw)
    soc = pulp.LpVariable.dicts("SoC", ((s, t) for s in scenarios for t in hours), lowBound=soc_min, upBound=soc_max)
    
    # متغیرهای ریسک CVaR
    VaR = pulp.LpVariable("VaR", lowBound=None)
    z = pulp.LpVariable.dicts("z", (s for s in scenarios), lowBound=0)
    
    # سود انتظاری
    expected_profit = pulp.lpSum(
        (scenario_prices[s, t] * (p_dis[s, t] - p_ch[s, t])) 
        for s in scenarios for t in hours
    ) / n_scenarios
    
    # پنالتی ریسک
    cvar_penalty = VaR + (1.0 / (1.0 - alpha)) * pulp.lpSum(z[s] for s in scenarios) / n_scenarios
    
    # تابع هدف
    prob += expected_profit - beta * cvar_penalty
    
    # قیود CVaR
    for s in scenarios:
        rev_s = pulp.lpSum(scenario_prices[s, t] * (p_dis[s, t] - p_ch[s, t]) for t in hours)
        prob += -rev_s - VaR <= z[s]
        
    # قیود دینامیک باتری
    for s in scenarios:
        for t in hours:
            if t == 0:
                prob += soc[s, t] == initial_soc + (eta_ch * p_ch[s, t] - (p_dis[s, t] / eta_dis)) * 1.0
            else:
                prob += soc[s, t] == soc[s, t-1] + (eta_ch * p_ch[s, t] - (p_dis[s, t] / eta_dis)) * 1.0
                
    # حل مدل
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    cvar_val = VaR.value() + (1.0 / (1.0 - alpha)) * sum(z[s].value() for s in scenarios) / n_scenarios
    
    return {
        "status": pulp.LpStatus[prob.status],
        "expected_profit": expected_profit.value(),
        "cvar_risk": cvar_val,
        "scenario_prices": scenario_prices
    }
