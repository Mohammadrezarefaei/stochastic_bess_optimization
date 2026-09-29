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
    
    p_ch = {}
    p_dis = {}
    u_ch = {}
    u_dis = {}
    e_level = {}
    z = {}

    # دور زدن کامل باگ‌های __init__ با تزریق مستقیم ویژگی‌ها (Attributes)
    for s in scenarios:
        var_z = LpVariable(f"z_aux_{s}")
        var_z.lowBound = 0
        var_z.cat = "Continuous"
        z[s] = var_z

        for t in hours:
            # P_ch
            v_pch = LpVariable(f"P_ch_{s}_{t}")
            v_pch.lowBound = 0
            v_pch.upBound = max_power_mw
            v_pch.cat = "Continuous"
            p_ch[(s, t)] = v_pch
            
            # P_dis
            v_pdis = LpVariable(f"P_dis_{s}_{t}")
            v_pdis.lowBound = 0
            v_pdis.upBound = max_power_mw
            v_pdis.cat = "Continuous"
            p_dis[(s, t)] = v_pdis
            
            # u_ch
            v_uch = LpVariable(f"u_ch_{s}_{t}")
            v_uch.cat = "Binary"
            u_ch[(s, t)] = v_uch
            
            # u_dis
            v_udis = LpVariable(f"u_dis_{s}_{t}")
            v_udis.cat = "Binary"
            u_dis[(s, t)] = v_udis
            
        for t in range(25):
            v_e = LpVariable(f"E_{s}_{t}")
            v_e.lowBound = 0
            v_e.upBound = max_energy_mwh
            v_e.cat = "Continuous"
            e_level[(s, t)] = v_e

    eta = LpVariable("VaR_eta")
    eta.cat = "Continuous"
    
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
