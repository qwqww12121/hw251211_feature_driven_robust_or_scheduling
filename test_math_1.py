import gurobipy as gp
from gurobipy import GRB
import random


def solve_robust_surgery_scheduling():
    # --- 1. 数据与参数设定 ---
    random.seed(42)

    # 手术室集合
    num_ors = 3
    ors = range(num_ors)
    H = 8 * 60  # 每个手术室的计划时间长度，单位分钟

    # 电择病人
    num_elective = 10
    elective_patients = range(num_elective)

    # 潜在急诊病人
    num_potential_emergency = 5
    emergency_patients = range(num_elective, num_elective + num_potential_emergency)

    # 病人类型集合
    patient_types = ['TypeA', 'TypeB', 'TypeC', 'DUMMY']

    # 电择病人类型映射（简单随机生成）
    elective_patient_type_map = {
        i: random.choice(['TypeA', 'TypeB', 'TypeC']) for i in elective_patients
    }

    # 急诊病人类型概率 p_{js}
    emergency_type_prob = {
        j: {s: random.random() for s in patient_types}
        for j in emergency_patients
    }
    # 归一化概率
    for j in emergency_patients:
        total_prob = sum(emergency_type_prob[j].values())
        if total_prob > 0:
            for s in patient_types:
                emergency_type_prob[j][s] /= total_prob

    # 各类型手术时长的均值与方差参数（可根据真实数据设定）
    duration_stats = {
        'TypeA': {'mean': 120, 'var': 30 ** 2},
        'TypeB': {'mean': 90,  'var': 20 ** 2},
        'TypeC': {'mean': 180, 'var': 45 ** 2},
        'DUMMY': {'mean': 0,   'var': 0}
    }

    # 风险厌恶系数 lambda（可调）
    lambda_param = 1.0

    # --- 2. 创建 Gurobi 模型 ---
    model = gp.Model("Robust_OR_Scheduling_SOCP")

    # --- 3. 决策变量 ---
    # 每个手术室的 ORI 指数 alpha_k >= 0
    alpha = model.addVars(ors, name="alpha", lb=0.0)

    # 电择病人分配变量 x_{ik} ∈ {0,1}
    x_elective = model.addVars(
        elective_patients, ors, vtype=GRB.BINARY, name="x_e"
    )

    # 急诊病人策略分配变量 x_{jk}^s ∈ {0,1}
    x_emergency = model.addVars(
        emergency_patients, ors, patient_types, vtype=GRB.BINARY, name="x_m"
    )

    # 每个手术室 k 关联的方差上界的平方根 t_k >= 0
    t_var = model.addVars(ors, name="t_var", lb=0.0)

    # --- 4. 目标函数 ---
    # 最小化所有手术室的 ORI 之和
    model.setObjective(gp.quicksum(alpha[k] for k in ors), GRB.MINIMIZE)

    # --- 5. 分配约束 ---
    # 每个电择病人必须且只分配到一个手术室
    for i in elective_patients:
        model.addConstr(
            gp.quicksum(x_elective[i, k] for k in ors) == 1,
            name=f"assign_e_{i}"
        )

    # 每个急诊病人-类型组合的策略分配：对每种类型 s，若发生该类型，则必须分到且只分到一个 OR
    for j in emergency_patients:
        for s in patient_types:
            model.addConstr(
                gp.quicksum(x_emergency[j, k, s] for k in ors) == 1,
                name=f"assign_m_{j}_{s}"
            )

    # --- 6. ORI 相关的期望与方差约束（SOCP 部分） ---
    for k in ors:
        # 6.1 计算期望 E[D_k] = E[T_k] - H
        # E[T_k] 包含电择和急诊部分
        exp_Tk_elective = gp.quicksum(
            duration_stats[elective_patient_type_map[i]]['mean'] * x_elective[i, k]
            for i in elective_patients
        )

        exp_Tk_emergency = gp.quicksum(
            emergency_type_prob[j][s] * duration_stats[s]['mean'] * x_emergency[j, k, s]
            for j in emergency_patients
            for s in patient_types
        )

        exp_Tk = exp_Tk_elective + exp_Tk_emergency
        exp_Dk = exp_Tk - H  # E[D_k]

        # 6.2 构造 Var[D_k] 的线性上界
        var_elective_part = gp.quicksum(
            duration_stats[elective_patient_type_map[i]]['var'] * x_elective[i, k]
            for i in elective_patients
        )

        var_emergency_part = gp.quicksum(
            emergency_type_prob[j][s] * duration_stats[s]['var'] * x_emergency[j, k, s]
            for j in emergency_patients
            for s in patient_types
        )

        var_upper = var_elective_part + var_emergency_part

        # 6.3 t_var[k]^2 >= Var_upper （标准二阶锥约束）
        model.addQConstr(
            t_var[k] * t_var[k] >= var_upper,
            name=f"q_var_upper_{k}"
        )

        # 6.4 线性 ORI 约束：alpha_k >= E[D_k] + lambda * t_k
        # 这对应文中“均值 + λ * 标准差”的风险度量形式
        model.addConstr(
            alpha[k] >= exp_Dk + lambda_param * t_var[k],
            name=f"ori_risk_{k}"
        )

    # --- 7. 求解模型 ---
    # 因为模型是凸 SOCP，不需要 NonConvex = 2
    model.setParam('TimeLimit', 120)
    model.optimize()

    # --- 8. 结果展示 ---
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        print("\n求解完成！（分布鲁棒 ORI 模型，SOCP 版本）")
        if model.status == GRB.TIME_LIMIT:
            print("注意：达到时间上限，解可能是次优可行解。")

        print(f"目标值 (总ORI): {model.ObjVal:.4f}\n")

        print("--- 手术室风险指数 (alpha_k) ---")
        for k in ors:
            if alpha[k].X is not None:
                print(f"手术室 {k}: alpha = {alpha[k].X:.4f}")

        print("\n--- 每个手术室的方差上界 t_k ---")
        for k in ors:
            if t_var[k].X is not None:
                print(f"手术室 {k}: t_var = {t_var[k].X:.4f}")

        print("\n--- 电择病人分配方案 ---")
        for i in elective_patients:
            assigned_or = None
            for k in ors:
                if x_elective[i, k].X > 0.5:
                    assigned_or = k
                    break
            print(f"  电择病人 {i} -> 手术室 {assigned_or}")

        print("\n--- 急诊病人策略分配规则（示例显示第一个急诊病人）---")
        if len(emergency_patients) > 0:
            j_example = emergency_patients[0]
            for s in patient_types:
                assigned_or = None
                for k in ors:
                    if x_emergency[j_example, k, s].X > 0.5:
                        assigned_or = k
                        break
                print(f"  若急诊病人 {j_example} 为 {s} 型，则分配到手术室 {assigned_or}")
    else:
        print(f"\n求解失败或未找到可行解，状态码: {model.status}")


if __name__ == "__main__":
    solve_robust_surgery_scheduling()