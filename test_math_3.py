import gurobipy as gp
from gurobipy import GRB
import random


def build_and_solve(lambda_param, H, par_factor, setup_time, hard_cap, cap_factor, w_soft, w_var_assign, time_limit, patient_types, elective_patients, emergency_patients, elective_patient_type_map, emergency_type_prob, duration_stats, specialty_allowed):
    num_ors = len(par_factor)
    ors = range(num_ors)
    model = gp.Model("Robust_OR_Scheduling_Combined_SOCP")
    alpha = model.addVars(ors, name="alpha", lb=0.0)
    x_elective = model.addVars(elective_patients, ors, vtype=GRB.BINARY, name="x_e")
    x_emergency = model.addVars(emergency_patients, ors, patient_types, vtype=GRB.BINARY, name="x_m")
    sigma = model.addVars(ors, name="sigma", lb=0.0)
    z = model.addVars(ors, name="z", lb=0.0)
    slack = model.addVars(ors, name="slack", lb=0.0)
    amax = model.addVar(name="amax", lb=0.0)

    obj = gp.quicksum(alpha[k] for k in ors) + w_soft * gp.quicksum(slack[k] for k in ors) + w_var_assign * gp.quicksum(emergency_type_prob[j][s] * duration_stats[s]['var'] * x_emergency[j, k, s] for j in emergency_patients for k in ors for s in patient_types) + 0.1 * amax
    model.setObjective(obj, GRB.MINIMIZE)

    for i in elective_patients:
        model.addConstr(gp.quicksum(x_elective[i, k] for k in ors) == 1)

    for j in emergency_patients:
        for s in patient_types:
            model.addConstr(gp.quicksum(x_emergency[j, k, s] for k in ors) == 1)

    for k in ors:
        Hk = H * par_factor[k]
        exp_Tk_elective = gp.quicksum(duration_stats[elective_patient_type_map[i]]['mean'] * x_elective[i, k] for i in elective_patients)
        exp_Tk_emergency = gp.quicksum(emergency_type_prob[j][s] * duration_stats[s]['mean'] * x_emergency[j, k, s] for j in emergency_patients for s in patient_types)
        cnt_e = gp.quicksum(x_elective[i, k] for i in elective_patients)
        cnt_m = gp.quicksum(emergency_type_prob[j][s] * x_emergency[j, k, s] for j in emergency_patients for s in patient_types)
        exp_Tk = exp_Tk_elective + exp_Tk_emergency + setup_time * (cnt_e + cnt_m)
        exp_Dk = exp_Tk - Hk

        var_elective_part = gp.quicksum(duration_stats[elective_patient_type_map[i]]['var'] * x_elective[i, k] for i in elective_patients)
        var_emergency_part = gp.quicksum(emergency_type_prob[j][s] * duration_stats[s]['var'] * x_emergency[j, k, s] for j in emergency_patients for s in patient_types)
        var_upper = var_elective_part + var_emergency_part

        model.addQConstr(sigma[k] * sigma[k] >= var_upper)
        model.addQConstr(z[k] * z[k] >= exp_Dk * exp_Dk + sigma[k] * sigma[k])

        model.addConstr(alpha[k] >= exp_Dk + lambda_param * sigma[k])
        model.addConstr(alpha[k] >= 0.5 * (exp_Dk + z[k]))
        model.addConstr(slack[k] >= exp_Tk - Hk)
        model.addConstr(amax >= alpha[k])
        if hard_cap:
            model.addConstr(exp_Tk <= cap_factor * Hk)

    for k in ors:
        allowed = specialty_allowed.get(k, set(patient_types))
        for i in elective_patients:
            pt = elective_patient_type_map[i]
            if pt not in allowed:
                x_elective[i, k].UB = 0.0
        for j in emergency_patients:
            for s in patient_types:
                if s not in allowed:
                    x_emergency[j, k, s].UB = 0.0

    model.setParam('TimeLimit', time_limit)
    model.optimize()

    has_sol = model.SolCount and model.SolCount > 0
    res = {
        'status': model.status,
        'obj': model.ObjVal if has_sol else None,
        'alpha': [alpha[k].X if has_sol else None for k in ors],
        'sigma': [sigma[k].X if has_sol else None for k in ors],
        'z': [z[k].X if has_sol else None for k in ors],
        'assign_e': {i: next((k for k in ors if has_sol and x_elective[i, k].X > 0.5), None) for i in elective_patients},
        'assign_m_example': None
    }
    if has_sol and len(list(emergency_patients)) > 0:
        j_example = list(emergency_patients)[0]
        res['assign_m_example'] = {s: next((k for k in ors if x_emergency[j_example, k, s].X > 0.5), None) for s in patient_types}
    return res


def solve_grid():
    random.seed(42)
    num_ors = 3
    ors = range(num_ors)
    H = 8 * 60
    par_factor = [1.0, 1.0, 1.0]
    setup_time = 10.0
    hard_cap = False
    cap_factor = 1.0
    w_soft = 1.0
    w_var_assign = 0.001
    time_limit = 60
    patient_types = ['TypeA', 'TypeB', 'TypeC', 'DUMMY']
    num_elective = 10
    elective_patients = range(num_elective)
    num_potential_emergency = 5
    emergency_patients = range(num_elective, num_elective + num_potential_emergency)
    elective_patient_type_map = {i: random.choice(['TypeA', 'TypeB', 'TypeC']) for i in elective_patients}
    emergency_type_prob = {j: {s: random.random() for s in patient_types} for j in emergency_patients}
    for j in emergency_patients:
        total_prob = sum(emergency_type_prob[j].values())
        if total_prob > 0:
            for s in patient_types:
                emergency_type_prob[j][s] /= total_prob
    duration_stats = {
        'TypeA': {'mean': 120, 'var': 30 ** 2},
        'TypeB': {'mean': 90, 'var': 20 ** 2},
        'TypeC': {'mean': 180, 'var': 45 ** 2},
        'DUMMY': {'mean': 0, 'var': 0}
    }
    specialty_allowed = {
        0: {'TypeA', 'TypeB', 'TypeC', 'DUMMY'},
        1: {'TypeA', 'TypeB', 'DUMMY'},
        2: {'TypeB', 'TypeC', 'DUMMY'}
    }
    lambda_grid = [0.5, 1.0, 2.0]
    results = []
    for lam in lambda_grid:
        res = build_and_solve(lam, H, par_factor, setup_time, hard_cap, cap_factor, w_soft, w_var_assign, time_limit, patient_types, elective_patients, emergency_patients, elective_patient_type_map, emergency_type_prob, duration_stats, specialty_allowed)
        results.append((lam, res))
    for lam, res in results:
        if res['status'] in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            print(f"\nλ={lam}，目标值={res['obj']:.4f}")
            print("--- 手术室风险与不确定性 ---")
            for k in range(num_ors):
                print(f"手术室 {k}: 风险指数(alpha)={res['alpha'][k]:.4f}，不确定性(sigma)={res['sigma'][k]:.4f}，合成(z)={res['z'][k]:.4f}")
            print("\n--- 电择病人分配 ---")
            for i in elective_patients:
                print(f"电择病人 {i} -> 手术室 {res['assign_e'][i]}")
            if res['assign_m_example'] is not None:
                print("\n--- 急诊病人策略分配示例（第一个急诊病人）---")
                for s, k in res['assign_m_example'].items():
                    print(f"若类型 {s} -> 手术室 {k}")
        else:
            print(f"\nλ={lam} 不可行，状态码={res['status']}")


if __name__ == "__main__":
    solve_grid()