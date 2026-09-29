import random
from pulp import LpProblem, LpMaximize, LpVariable, lpSum

def run_stochastic_bess_optimization(n_scenarios=50, beta=0.5, max_power_mw=5.0, max_energy_mwh=10.0):
    n_scenarios = int(n_scenarios)
    beta = float(beta)
    max_power_mw = float(max_power_mw)
    max_energy_mwh = float(max_energy_mwh)

    hours = list(range(24))
    scenarios = list(range(n_scenarios))
    
    scenario_prices = {}
    base_profile = [30, 25, 20, 18, 20, 28, 45, 65, 80, 60, 45, 40, 35, 35, 40, 50, 75, 95, 110, 85, 60, 50, 40, 32]
    
    for s in scenarios:
        scenario_prices[s] = [
            max(5.0, p + random.gauss(0, 12) + (10 if 17 <= h <= 20 else 0))
            for h, p in enumerate(base_profile)
        ]
    
    model = LpProblem("Stochastic_BESS_Optimization", LpMaximize)
    
    # ساخت متغیرها به صورت دستی برای جلوگیری ۱۰۰٪ از ارور dicts در پایتون ۳.۱۴
    # استفاده صریح از name=... برای جلوگیری از هرگونه TypeError
    p_ch = {}
    p_dis = {}
    u_ch = {}
    u_dis = {}
    e_level = {}
    z = {}

    for s in scenarios:
        z[s] = LpVariable(name=f"z_aux_{s}", lowBound=0, cat="Continuous")
        for t in hours:
            p_ch[(s, t)] = LpVariable(name=f"P_ch_{s}_{t}", lowBound=0, upBound=max_power_mw, cat="Continuous")
            p_dis[(s, t)] = LpVariable(name=f"P_dis_{s}_{t}", lowBound=0, upBound=max_power_mw, cat="Continuous")
            u_ch[(s, t)] = LpVariable(name=f"u_ch_{s}_{t}", cat="Binary")
            u_dis[(s, t)] = LpVariable(name=f"u_dis_{s}_{t}", cat="Binary")
            
        for t in range(25):
            e_level[(s, t)] = LpVariable(name=f"E_{s}_{t}", lowBound=0, upBound=max_energy_mwh, cat="Continuous")

    eta = LpVariable(name="VaR_eta", cat="Continuous")
    
    efficiency = 0.92
    prob_s = 1.0 / n_scenarios
    
    expected_profit = lpSum([
        prob_s * (scenario_prices[s][t] * (p_dis[(s, t)] - p_ch[(s, t)]))
        for s in scenarios for t in hours
    ])
    
    cvar_penalty = beta * (eta + (1.0 / (0.05 * n_scenarios)) * lpSum([prob_s * z[s] for s in scenarios]))
    
    model += expected_profit - cvar_penalty, "Objective_Function"
    
    for s in scenarios:
        model += e_level[(s, 0)] == 0.5 * max_energy_mwh
        
        for t in hours:
            model += e_level[(s, t+1)] == e_level[(s, t)] + (p_ch[(s, t)] * efficiency) - (p_dis[(s, t)] / efficiency)
            model += u_ch[(s, t)] + u_dis[(s, t)] <= 1
            model += p_ch[(s, t)] <= max_power_mw * u_ch[(s, t)]
            model += p_dis[(s, t)] <= max_power_mw * u_dis[(s, t)]
            
        profit_s = lpSum([scenario_prices[s][t] * (p_dis[(s, t)] - p_ch[(s, t)]) for t in hours])
        model += z[s] >= -profit_s - eta
        
    model.solve()
    
    # ساخت دیکشنری وضعیت به صورت محلی برای حذف کامل ارور ImportError مربوط به LpStatus
    status_map = {1: "Optimal", 0: "Not Solved", -1: "Infeasible", -2: "Unbounded", -3: "Undefined"}
    status_str = status_map.get(model.status, "Unknown")
    
    expected_prof_val = 0.0
    cvar_val = 0.0
    
    if status_str == "Optimal":
        expected_prof_val = sum(
            prob_s * sum(scenario_prices[s][t] * (p_dis[(s, t)].varValue - p_ch[(s, t)].varValue) for t in hours)
            for s in scenarios
        )
        cvar_val = getattr(eta, 'varValue', 0.0) if getattr(eta, 'varValue', None) is not None else 0.0

    return {
        "status": status_str,
        "expected_profit": expected_prof_val,
        "cvar_risk": cvar_val,
        "scenario_prices": scenario_prices
    }
