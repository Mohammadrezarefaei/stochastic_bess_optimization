import random
import pulp

def run_stochastic_bess_optimization(n_scenarios=50, beta=0.5, max_power_mw=5.0, max_energy_mwh=10.0):
    """
    Runs a stochastic MILP optimization for a Battery Energy Storage System (BESS)
    considering Day-Ahead market price uncertainty and CVaR risk aversion.
    """
    hours = list(range(24))
    scenarios = list(range(n_scenarios))
    
    # تولید سناریوهای مصنوعی قیمت بر اساس الگوی معمول بازار آلمان
    scenario_prices = {}
    base_profile = [30, 25, 20, 18, 20, 28, 45, 65, 80, 60, 45, 40, 35, 35, 40, 50, 75, 95, 110, 85, 60, 50, 40, 32]
    
    for s in scenarios:
        scenario_prices[s] = [
            max(5.0, p + random.gauss(0, 12) + (10 if 17 <= h <= 20 else 0))
            for h, p in enumerate(base_profile)
        ]
    
    # راه‌اندازی مدل بهینه‌سازی PuLP
    model = pulp.LpProblem("Stochastic_BESS_Optimization", pulp.LpMaximize)
    
    # تعریف متغیرها بدون خطای نوع (Type Error) با ساختار امن
    p_ch = {}
    p_dis = {}
    e_level = {}
    u_ch = {}
    u_dis = {}
    z = {}

    for s in scenarios:
        z[s] = pulp.LpVariable(f"z_aux_{s}", lowBound=0)
        for t in hours:
            p_ch[s, t] = pulp.LpVariable(f"P_ch_{s}_{t}", 0, max_power_mw)
            p_dis[s, t] = pulp.LpVariable(f"P_dis_{s}_{t}", 0, max_power_mw)
            u_ch[s, t] = pulp.LpVariable(f"u_ch_{s}_{t}", cat='Binary')
            u_dis[s, t] = pulp.LpVariable(f"u_dis_{s}_{t}", cat='Binary')
            
        for t in range(25):
            e_level[s, t] = pulp.LpVariable(f"E_{s}_{t}", 0, max_energy_mwh)

    eta = pulp.LpVariable("VaR_eta", lowBound=None)
    
    efficiency = 0.92
    prob_s = 1.0 / n_scenarios
    
    # تابع هدف: حداکثرسازی سود انتظاری منهای جریمه ریسک (CVaR)
    expected_profit = pulp.lpSum(
        prob_s * (scenario_prices[s][t] * (p_dis[s, t] - p_ch[s, t]))
        for s in scenarios for t in hours
    )
    
    cvar_penalty = beta * (eta + (1.0 / (0.05 * n_scenarios)) * pulp.lpSum(prob_s * z[s] for s in scenarios))
    
    model += expected_profit - cvar_penalty, "Objective_Function"
    
    # محدودیت‌های سیستم
    for s in scenarios:
        # انرژی اولیه و نهایی باتری
        model += e_level[s, 0] == 0.5 * max_energy_mwh
        
        for t in hours:
            # تعادل انرژی در باتری
            model += e_level[s, t+1] == e_level[s, t] + (p_ch[s, t] * efficiency - p_dis[s, t] / efficiency)
            
            # مهار شارژ و دشارژ همزمان با متغیر باینری
            model += u_ch[s, t] + u_dis[s, t] <= 1
            model += p_ch[s, t] <= max_power_mw * u_ch[s, t]
            model += p_dis[s, t] <= max_power_mw * u_dis[s, t]
            
        # محدودیت‌های مربوط به محاسبه CVaR
        profit_s = pulp.lpSum(scenario_prices[s][t] * (p_dis[s, t] - p_ch[s, t]) for t in hours)
        model += z[s] >= -profit_s - eta
        
    # حل مدل با سالور پیش‌فرض PuLP
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    status = pulp.LpStatus[model.status]
    
    # استخراج نتایج
    calculated_expected_profit = sum(
        prob_s * sum(scenario_prices[s][t] * (p_dis[s, t].varValue - p_ch[s, t].varValue) for t in hours)
        for s in scenarios
    ) if status == "Optimal" else 0.0
    
    cvar_val = eta.varValue if status == "Optimal" else 0.0

    return {
        "status": status,
        "expected_profit": calculated_expected_profit,
        "cvar_risk": cvar_val,
        "scenario_prices": scenario_prices
    }
