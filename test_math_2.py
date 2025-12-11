### 只考虑手术室全部使用，没有预留空手术室的急诊的情况
###我写错了，不用管__2025.11.19 23:04
#ustc

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

    # 各类型手术时长的均值与方差参数
    duration_stats = {
        'TypeA': {'mean': 120, 'var': 30 ** 2},
        'TypeB': {'mean': 90,  'var': 20 ** 2},
        'TypeC': {'mean': 180, 'var': 45 ** 2},
        'DUMMY': {'mean': 0,   'var': 0}
    }

    # # --- 1. 数据与参数设定（替换为真实值后）---
    # random.seed(42)  # 随机种子可保留（确保结果可复现）

    # # 手术室集合（真实：2间手术室，每天工作7小时）
    # num_ors = 2
    # ors = range(num_ors)
    # H = 7 * 60  # 420分钟

    # # 择期病人（真实：8个择期病人，类型已知）
    # num_elective = 8
    # elective_patients = range(num_elective)
    # elective_patient_type_map = {
    #     0: 'TypeA',
    #     1: 'TypeB',
    #     2: 'TypeA',
    #     3: 'TypeC',
    #     4: 'TypeB',
    #     5: 'TypeA',
    #     6: 'TypeC',
    #     7: 'TypeB'
    # }

    # # 潜在急诊病人（真实：3个潜在急诊）
    # num_potential_emergency = 3
    # emergency_patients = range(num_elective, num_elective + num_potential_emergency)  # 索引8、9、10

    # # 急诊病人类型概率（真实统计的概率分布）
    # emergency_type_prob = {
    #     8: {'TypeA': 0.3, 'TypeB': 0.4, 'TypeC': 0.2, 'DUMMY': 0.1},
    #     9: {'TypeA': 0.2, 'TypeB': 0.5, 'TypeC': 0.3, 'DUMMY': 0.0},
    #     10: {'TypeA': 0.4, 'TypeB': 0.3, 'TypeC': 0.2, 'DUMMY': 0.1}
    # }
    # # 归一化
    # for j in emergency_patients:
    #     total_prob = sum(emergency_type_prob[j].values())
    #     if total_prob > 0:
    #         for s in patient_types:
    #             emergency_type_prob[j][s] /= total_prob

    # # 手术时长统计（真实历史数据计算）
    # duration_stats = {
    #     'TypeA': {'mean': 110, 'var': 25**2},  # 均值110分钟，标准差25
    #     'TypeB': {'mean': 85,  'var': 18**2},  # 均值85分钟，标准差18
    #     'TypeC': {'mean': 160, 'var': 40**2},  # 均值160分钟，标准差40
    #     'DUMMY': {'mean': 0,   'var': 0}       # 虚拟类型固定不变
    # }

    # --- 2. 创建 Gurobi 模型 ---
    model = gp.Model("Robust_OR_Scheduling_Exact_ORI_SOCP")

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

    # 方差上界的平方根 v_k >= 0 （Var_upper 的 sqrt）
    v_var = model.addVars(ors, name="v_var", lb=0.0)

    # 表示 sqrt(E[D_k]^2 + Var[D_k]) 的变量 z_k >= 0
    z_soc = model.addVars(ors, name="z_soc", lb=0.0)

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

    # --- 6. ORI 相关的期望与方差约束（精确 ORI 公式 + Var 上界） ---
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
        exp_Dk = exp_Tk - H  # 这是 E[D_k]，一个线性表达式

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

        # 6.3 v_var[k]^2 >= Var_upper （表示 sqrt(Var_upper) 的二阶锥约束）
        model.addQConstr(
            v_var[k] * v_var[k] >= var_upper,
            name=f"q_var_upper_{k}"
        )

        # 6.4 z_soc[k]^2 >= (E[D_k])^2 + v_var[k]^2
        # 这是 sqrt(E[D_k]^2 + Var[D_k]) 的 SOCP 形式
        model.addQConstr(
            z_soc[k] * z_soc[k] >= exp_Dk * exp_Dk + v_var[k] * v_var[k],
            name=f"q_z_soc_{k}"
        )

        # 6.5 精确 ORI 约束：0.5 * (E[D_k] + z_k) <= alpha_k
        model.addConstr(
            0.5 * (exp_Dk + z_soc[k]) <= alpha[k],
            name=f"ori_exact_{k}"
        )

    # --- 7. 求解模型 ---
    # 这是一个凸 SOCP，不需要 NonConvex 参数
    model.setParam('TimeLimit', 120)
    model.optimize()

    # --- 8. 结果展示 ---
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        print("\n求解完成！（分布鲁棒 ORI 模型，精确公式 SOCP 版本）")
        if model.status == GRB.TIME_LIMIT:
            print("注意：达到时间上限，解可能是次优可行解。")

        print(f"目标值 (总ORI): {model.ObjVal:.4f}\n")

        print("--- 手术室风险指数 (alpha_k) ---")
        for k in ors:
            if alpha[k].X is not None:
                print(f"手术室 {k}: alpha = {alpha[k].X:.4f}")

        print("\n--- 每个手术室的 v_var (≈ sqrt(Var_upper)) 和 z_soc ---")
        for k in ors:
            print(
                f"手术室 {k}: v_var = {v_var[k].X:.4f}, "
                f"z_soc = {z_soc[k].X:.4f}"
            )

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
