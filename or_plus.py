import random
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt

random.seed(0)

# 1. 全局参数

num_OR = 3              # 手术室数量

# Case H：较高负载
num_patients = 24       # 择期病人数量

# 中等负载 Case M
# num_patients = 18
# num_patients = 15

T = 8.0                 # 每间 OR 正常工作时长（小时）
num_train_scen = 50     # 训练场景数（用于求解 SAA / 评估）
num_test_scen = 200     # 测试场景数（用于评估策略）

# 患者类型：类型 -> (均值, 标准差)，用于生成“真实时长”场景
# 高负载版本
patient_types = {
    0: (1.2, 0.2),   # 简单
    1: (1.8, 0.3),   # 中等
    2: (2.4, 0.4)    # 困难
}

# 中等负载 Case M
# patient_types = {
#     0: (1.0, 0.2),   # 简单
#     1: (1.5, 0.3),   # 中等
#     2: (2.0, 0.4)    # 困难
# }

type_list = list(patient_types.keys())
num_types = len(type_list)

# 类型 -> 平均手术时长
mean_by_type = {t: m for t, (m, s) in patient_types.items()}
# 新增：类型 -> 标准差（用于策略D修正）
sigma_by_type = {t: s for t, (m, s) in patient_types.items()}

# 为每个患者随机指定类型（可调整不同类型比例）
patient_type = {}
for i in range(num_patients):
    t = random.choices(type_list, weights=[0.4, 0.4, 0.2])[0]
    patient_type[i] = t

# 2. 生成训练场景 / 测试场景的手术时长

def generate_durations(num_scen):
    """生成 num_scen 个场景下，每个患者的手术时长 durations[i][w]"""
    durations = [[0.0 for _ in range(num_scen)] for _ in range(num_patients)]
    for i in range(num_patients):
        t = patient_type[i]
        mean, std = patient_types[t]
        for w in range(num_scen):
            d = random.gauss(mean, std)
            # 简单截断，避免出现负值或过小值
            d = max(d, 0.3)
            durations[i][w] = d
    return durations

train_durations = generate_durations(num_train_scen)
test_durations = generate_durations(num_test_scen)

# 3. 策略 C：SAA 模型 (M1)

def solve_policy_SAA(durations, T_value):
    """
    基于场景的 SAA 模型 (M1)，
    返回分配方案 x[i][r] 以及训练场景的平均加班目标值
    """
    model = Model("policy_SAA")
    model.setParam("OutputFlag", 0)

    x = {}
    z = {}

    num_scen = len(durations[0])

    # 决策变量 x[i,r]
    for i in range(num_patients):
        for r in range(num_OR):
            x[i, r] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{r}")

    # 决策变量 z[r,w]
    for r in range(num_OR):
        for w in range(num_scen):
            z[r, w] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"z_{r}_{w}")

    model.update()

    # 每个病人恰好分配到一个 OR
    for i in range(num_patients):
        model.addConstr(quicksum(x[i, r] for r in range(num_OR)) == 1)

    # 加班约束
    for r in range(num_OR):
        for w in range(num_scen):
            model.addConstr(
                quicksum(durations[i][w] * x[i, r] for i in range(num_patients)) - T_value <= z[r, w]
            )

    # 目标：训练场景上的平均加班时间
    avg_ot = (1.0 / num_scen) * quicksum(z[r, w] for r in range(num_OR) for w in range(num_scen))
    model.setObjective(avg_ot, GRB.MINIMIZE)

    model.optimize()

    # 提取解
    x_sol = [[0 for _ in range(num_OR)] for _ in range(num_patients)]
    for i in range(num_patients):
        for r in range(num_OR):
            if x[i, r].X > 0.5:
                x_sol[i][r] = 1

    return x_sol, model.ObjVal

# 4. 策略 A / B：确定性模型（M0 及基于类型均值）
def solve_policy_deterministic(d_mean, T_value, name="deterministic"):
    """
    确定性模型：d_mean[i] 是每个病人的确定时长（统一均值或类型均值）
    返回分配方案 x[i][r] 及目标值（总加班时间之和）
    """
    model = Model(name)
    model.setParam("OutputFlag", 0)

    x = {}
    z = {}

    # 决策变量 x[i,r]
    for i in range(num_patients):
        for r in range(num_OR):
            x[i, r] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{r}")

    # 每个 OR 的加班时间 z[r]
    for r in range(num_OR):
        z[r] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"z_{r}")

    model.update()

    # 每个病人恰好分配到一个 OR
    for i in range(num_patients):
        model.addConstr(quicksum(x[i, r] for r in range(num_OR)) == 1)

    # 加班约束（使用确定性时长 d_mean）
    for r in range(num_OR):
        model.addConstr(
            quicksum(d_mean[i] * x[i, r] for i in range(num_patients)) - T_value <= z[r]
        )

    # 目标：所有 OR 加班时间之和
    model.setObjective(quicksum(z[r] for r in range(num_OR)), GRB.MINIMIZE)

    model.optimize()

    # 提取解
    x_sol = [[0 for _ in range(num_OR)] for _ in range(num_patients)]
    for i in range(num_patients):
        for r in range(num_OR):
            if x[i, r].X > 0.5:
                x_sol[i][r] = 1

    return x_sol, model.ObjVal

# 5. ORI / Fano 辅助函数（仅修改策略D相关，其余不变）

# 类型 -> 索引 映射（便于梯度向量取值）
type_index = {s: idx for idx, s in enumerate(type_list)}

# 计算OR中各类型患者数n_sr和总期望时长y_sr
def compute_y_and_n_from_x(x_sol):
    """
    给定整数解 x_sol[i][r]，计算：
    y[(s,r)] = 类型s在OR r的总期望时长（n_sr * μ_s）
    n[(s,r)] = 类型s在OR r的患者数
    """
    y = {}
    n = {}
    for r in range(num_OR):
        for s in type_list:
            y[s, r] = 0.0
            n[s, r] = 0
    for i in range(num_patients):
        s = patient_type[i]
        dur_mean = mean_by_type[s]
        for r in range(num_OR):
            if x_sol[i][r] == 1:
                n[s, r] += 1  # 统计患者数
                y[s, r] += dur_mean  # 累加总期望时长
    return y, n

# Fano因子梯度计算
def grad_Fano_given_n(n_vec, sigma_vec, mu_vec, D_val, V_val):
    """
    计算Fano因子对n_sr的梯度：dF/dn_sr = [σ_s²×D + V×μ_s] / D²
    n_vec: 各类型患者数列表；sigma_vec: 各类型标准差；mu_vec: 各类型均值
    D_val: 期望裕量（T - sum(n_sr×μ_s)）；V_val: 总方差（sum(n_sr×σ_s²)）
    """
    g = []
    for s_idx in range(len(n_vec)):
        sigma_s = sigma_vec[s_idx]
        mu_s = mu_vec[s_idx]
        numerator = (sigma_s ** 2) * D_val + V_val * mu_s
        g_s = numerator / (D_val ** 2)
        g.append(g_s)
    return g

# 6. 策略 D：特征驱动 ORI/Fano 模型

def solve_policy_ORI_Fano(max_iters=30, tol=1e-4):
    """
    近似求解 Wang et al. (2023) 的特征驱动 DRO 模型的简化版本：
        min sum_r alpha_r
        s.t. alpha_r >= Fano_r(y_r(x)),  对所有 r
             sum_r x_ir = 1, x_ir ∈ {0,1}

    其中 Fano_r(y) = 总方差 / 期望裕量（修正后）；
    为了在系统严重超载时保持可行性：
    - 若 D_r = T - 期望总时长 <= 0，则强制 alpha_r >= bigM（大罚值）

    使用切平面(outer-approximation)方法：不断添加线性近似约束。
    """
    bigM = 1e4  # 严重超载时的惩罚

    # 初始 MILP（master problem） 
    model = Model("policy_D_ORI_Fano")
    model.setParam("OutputFlag", 0)

    x = {}
    alpha = {}

    # 决策变量 x[i,r]
    for i in range(num_patients):
        for r in range(num_OR):
            x[i, r] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{r}")

    # 每个 OR 的 alpha_r
    for r in range(num_OR):
        alpha[r] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"alpha_{r}")

    model.update()

    # 每个病人恰好分配到一个 OR
    for i in range(num_patients):
        model.addConstr(quicksum(x[i, r] for r in range(num_OR)) == 1)

    # 目标：min sum_r alpha_r
    model.setObjective(quicksum(alpha[r] for r in range(num_OR)), GRB.MINIMIZE)
    model.update()

    # 切平面迭代
    for it in range(max_iters):
        model.optimize()

        # 检查是否有可行解
        if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) or model.SolCount == 0:
            raise RuntimeError(
                f"ORI/Fano 主问题无可行解（status = {model.Status}）。"
            )

        # 提取当前解 x_sol, alpha_sol
        x_sol = [[0 for _ in range(num_OR)] for _ in range(num_patients)]
        alpha_sol = [0.0 for _ in range(num_OR)]

        for i in range(num_patients):
            for r in range(num_OR):
                if x[i, r].X > 0.5:
                    x_sol[i][r] = 1

        for r in range(num_OR):
            alpha_sol[r] = alpha[r].X

        # 计算当前 n_sr（患者数）、y_sr（总期望时长）和 Fano_r
        violated_any = False
        y, n = compute_y_and_n_from_x(x_sol)

        for r in range(num_OR):
            # 提取当前OR的n_sr、sigma_s、mu_s
            n_vec = [n[s, r] for s in type_list]
            sigma_vec = [sigma_by_type[s] for s in type_list]
            mu_vec = [mean_by_type[s] for s in type_list]
            
            # 计算期望总时长Y_val和期望裕量D_val
            Y_val = sum(n_vec[s_idx] * mu_vec[s_idx] for s_idx in range(len(type_list)))
            D_val = T - Y_val
            
            # 计算真实总方差V_val
            V_val = sum(n_vec[s_idx] * (sigma_vec[s_idx] ** 2) for s_idx in range(len(type_list)))
            
            # 计算Fano因子
            if D_val <= 1e-6:
                # 严重超载或刚好满负荷，用大罚值代替Fano
                f_val = bigM
                if f_val > alpha_sol[r] + tol:
                    violated_any = True
                    # 简单线性约束：alpha_r >= bigM
                    model.addConstr(alpha[r] >= bigM,
                                    name=f"bigM_OR{r}_iter{it}")
                continue
            else:
                f_val = V_val / D_val  # 正确的Fano因子（方差/期望裕量）

            if f_val <= alpha_sol[r] + tol:
                continue

            violated_any = True

            # 计算梯度 g(n_r) 并添加切平面约束：f(n0) + g(n0)^T (n(x) - n0) <= alpha_r
            g_vec = grad_Fano_given_n(n_vec, sigma_vec, mu_vec, D_val, V_val)
            # 常数项：-f(n0) - sum(g_s * n_s0)
            const_term = -f_val - sum(g_vec[s_idx] * n_vec[s_idx] for s_idx in range(len(type_list)))

            # 构造左边的线性表达式：sum_s g_s * n_sr(x)，其中n_sr(x)是类型s在OR r的患者数
            lhs_terms = []
            for s_idx in range(len(type_list)):
                s = type_list[s_idx]
                g_sr = g_vec[s_idx]
                # 累加类型s患者的x_ir项（x_ir=1表示患者i分配到OR r）
                for i in range(num_patients):
                    if patient_type[i] == s:
                        lhs_terms.append(g_sr * x[i, r])

            lhs = quicksum(lhs_terms)

            model.addConstr(lhs <= alpha[r] + const_term,
                            name=f"cut_OR{r}_iter{it}")

        if not violated_any:
            # 所有 OR 都不违反，认为收敛
            break

    # 最终提取整数解
    final_x_sol = [[0 for _ in range(num_OR)] for _ in range(num_patients)]
    for i in range(num_patients):
        for r in range(num_OR):
            if x[i, r].X > 0.5:
                final_x_sol[i][r] = 1

    final_obj = sum(alpha[r].X for r in range(num_OR))
    return final_x_sol, final_obj

# 7. 评估函数：在测试场景上评估某个分配方案

def evaluate_policy(x_sol, durations_test, T_value, threshold=0.5):
    """
    在测试场景上评估方案的 TE, WO, JP, JPT
    """
    num_scen = len(durations_test[0])

    total_overtime = 0.0     # 所有场景、所有 OR 的加班总和
    max_overtime = 0.0       # 所有场景中的最大 OR 加班
    count_overtime = 0       # 至少一个 OR 加班的场景个数
    count_overtime_big = 0   # 至少一个 OR 加班超过 threshold 的场景个数

    for w in range(num_scen):
        scen_overtime_sum = 0.0
        scen_max_or_ot = 0.0
        for r in range(num_OR):
            total_time = 0.0
            for i in range(num_patients):
                if x_sol[i][r] == 1:
                    total_time += durations_test[i][w]
            overtime = max(0.0, total_time - T_value)
            scen_overtime_sum += overtime
            scen_max_or_ot = max(scen_max_or_ot, overtime)

        total_overtime += scen_overtime_sum
        max_overtime = max(max_overtime, scen_max_or_ot)

        if scen_max_or_ot > 1e-6:
            count_overtime += 1
        if scen_max_or_ot > threshold:
            count_overtime_big += 1

    TE = total_overtime / num_scen               # Total Expected overtime
    WO = max_overtime                            # Worst-case overtime
    JP = count_overtime / num_scen               # Joint overtime probability
    JPT = count_overtime_big / num_scen          # Joint overtime > threshold

    return TE, WO, JP, JPT

# 8. 主流程：四种策略对比 (A, B, C, D)

# 策略 B：按类型的均值（type mean）
d_mean_type = [mean_by_type[patient_type[i]] for i in range(num_patients)]

# 策略 A：统一均值（global mean）
global_mean = sum(d_mean_type) / len(d_mean_type)
d_mean_global = [global_mean for _ in range(num_patients)]

# 解策略 A、B、C、D（在 T=8 的训练场景下）
x_A, obj_A = solve_policy_deterministic(d_mean_global, T, name="policy_A_global")
x_B, obj_B = solve_policy_deterministic(d_mean_type, T, name="policy_B_type")
x_C, obj_C = solve_policy_SAA(train_durations, T)
x_D, obj_D = solve_policy_ORI_Fano()

print("Training objective (A, B, C, D):", obj_A, obj_B, obj_C, obj_D)

# 在测试场景上评估
TE_A, WO_A, JP_A, JPT_A = evaluate_policy(x_A, test_durations, T)
TE_B, WO_B, JP_B, JPT_B = evaluate_policy(x_B, test_durations, T)
TE_C, WO_C, JP_C, JPT_C = evaluate_policy(x_C, test_durations, T)
TE_D, WO_D, JP_D, JPT_D = evaluate_policy(x_D, test_durations, T)

print("Policy A (global mean): TE, WO, JP, JPT =",
      TE_A, WO_A, JP_A, JPT_A)
print("Policy B (type mean):   TE, WO, JP, JPT =",
      TE_B, WO_B, JP_B, JPT_B)
print("Policy C (SAA):         TE, WO, JP, JPT =",
      TE_C, WO_C, JP_C, JPT_C)
print("Policy D (ORI/Fano):    TE, WO, JP, JPT =",
      TE_D, WO_D, JP_D, JPT_D)

# 9. 可视化：四种策略多指标比较（图 1–图 4）

labels = ["A_global", "B_type", "C_SAA", "D_ORI"]

TE_values  = [TE_A,  TE_B,  TE_C,  TE_D]
WO_values  = [WO_A,  WO_B,  WO_C,  WO_D]
JP_values  = [JP_A,  JP_B,  JP_C,  JP_D]
JPT_values = [JPT_A, JPT_B, JPT_C, JPT_D]

# 图 1：TE 比较
plt.figure()
plt.bar(labels, TE_values)
plt.ylabel("Expected total overtime (hours)")
plt.title("Comparison of policies on TE")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# 图 2：WO 比较
plt.figure()
plt.bar(labels, WO_values)
plt.ylabel("Worst-case overtime (hours)")
plt.title("Comparison of policies on WO")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# 图 3：JP 比较
plt.figure()
plt.bar(labels, JP_values)
plt.ylabel("Joint overtime probability")
plt.ylim(0.0, 1.05)
plt.title("Comparison of policies on JP")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# 图 4：JPT 比较
plt.figure()
plt.bar(labels, JPT_values)
plt.ylabel("Severe overtime probability (>0.5h)")
plt.ylim(0.0, 1.05)
plt.title("Comparison of policies on JPT")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# 10. 基于策略 C 的工作时长灵敏度分析（图 7）

def sensitivity_on_T(T_values):
    """
    给定一组 T 值（正常工作时长），在每个 T 下：
    - 重新求解策略 C（SAA）；
    - 在测试场景上计算 TE；
    返回 TE_C(T) 的列表。
    """
    TE_list = []
    for T_val in T_values:
        # 重新求解策略 C
        x_C_T, obj_C_T = solve_policy_SAA(train_durations, T_val)
        TE_C_T, _, _, _ = evaluate_policy(x_C_T, test_durations, T_val)
        TE_list.append(TE_C_T)
        print(f"T = {T_val}, 训练目标值 = {obj_C_T:.4f}, 测试 TE_C = {TE_C_T:.4f}")
    return TE_list

T_list = [7.0, 8.0, 9.0]
TE_sens = sensitivity_on_T(T_list)

plt.figure()
plt.plot(T_list, TE_sens, marker="o")
plt.xlabel("Regular working time T (hours)")
plt.ylabel("Expected total overtime TE under policy C")
plt.title("Sensitivity of TE to T under SAA policy (Policy C)")
plt.grid(True)
plt.tight_layout()
plt.show()