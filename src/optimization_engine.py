import random
import pulp

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
    
    model = pulp.LpProblem("Stochastic_BESS_Optimization", pulp.LpMaximize)
    
    idx_st = [(s, t) for s in scenarios for t in hours]
    idx_e = [(s, t) for s in scenarios for t in range(25)]
    
    p_ch = pulp.LpVariable.dicts("P_ch", idx_st, lowBound=0, upBound=max_power_mw, cat='Continuous')
    p_dis = pulp.LpVariable.dicts("P_dis", idx_st, lowBound=0, upBound=max_power_mw, cat='Continuous')
    e_level = pulp.LpVariable.dicts("E", idx_e, lowBound=0, upBound=max_energy_mwh, cat='Continuous')
    
    u_ch = pulp.LpVariable.dicts("u_ch", idx_st, cat='Binary')
    u_dis = pulp.LpVariable.dicts("u_dis", idx_st, cat='Binary')
    
    z = pulp.LpVariable.dicts("z_aux", scenarios, lowBound=0, cat='Continuous')
    eta = pulp.LpVariable("VaR_eta", cat='Continuous')
    
    efficiency = 0.92
    prob_s = 1.0 / n_scenarios
    
    expected_profit = pulp.lpSum([
        prob_s * (scenario_prices[s][t] * (p_dis[(s, t)] - p_ch[(s, t)]))
        for s in scenarios for t in hours
    ])
    
    cvar_penalty = beta * (eta + (1.0 / (0.05 * n_scenarios)) * pulp.lpSum([prob_s * z[s] for s in scenarios]))
    
    model += expected_profit - cvar_penalty, "Objective_Function"
    
    for s in scenarios:
        model += e_level[(s, 0)] == 0.5 * max_energy_mwh
        
        for t in hours:
            # استفاده از ضرب در معکوس (1.0 / efficiency) به جای عملگر تقسیم برای جلوگیری از ارور
            model += e_level[(s, t+1)] == e_level[(s, t)] + (p_ch[(s, t)] * efficiency) - (p_dis[(s, t)] * (1.0 / efficiency))
            model += u_ch[(s, t)] + u_dis[(s, t)] <= 1
            model += p_ch[(s, t)] <= max_power_mw * u_ch[(s, t)]
            model += p_dis[(s, t)] <= max_power_mw * u_dis[(s, t)]
            
        profit_s = pulp.lpSum([scenario_prices[s][t] * (p_dis[(s, t)] - p_ch[(s, t)]) for t in hours])
        model += z[s] >= -profit_s - eta
        
    model.solve()
    
    status_str = pulp.LpStatus[model.status]
    
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
