import random
from gurobipy import Model, GRB, quicksum
import math
import matplotlib.pyplot as plt

random.seed(0)

# ==============
# 1. 全局参数
# ==============
num_OR = 3
num_patients = 18        # 增加病人数量，让系统更紧张
T = 8.0
num_train_scen = 50      # 训练场景数（用于求解 SAA）
num_test_scen = 200      # 测试场景数（用于评估策略）

# 患者类型：类型 -> (均值, 标准差)
patient_types = {
    0: (1.0, 0.2),#简单
    1: (1.5, 0.3),#中等
    2: (2.0, 0.4)#困难
}

# num_OR = 3
# num_patients = 24        # 增加病人数量，让系统更紧张
# T = 8.0
# num_train_scen = 50      # 训练场景数（用于求解 SAA）
# num_test_scen = 200      # 测试场景数（用于评估策略）

# # 患者类型：类型 -> (均值, 标准差)
# patient_types = {
#     0: (1.2, 0.2),   # 简单
#     1: (1.8, 0.3),   # 中等
#     2: (2.4, 0.4)    # 困难
# }

# 为每个患者指定类型
patient_type = {}
for i in range(num_patients):
    t = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]
    patient_type[i] = t

# ==============
# 2. 生成训练场景、测试场景的手术时长
# ==============

def generate_durations(num_scen):
    durations = [[0.0 for _ in range(num_scen)] for _ in range(num_patients)]
    for i in range(num_patients):
        t = patient_type[i]
        mean, std = patient_types[t]
        for w in range(num_scen):
            d = random.gauss(mean, std)
            d = max(d, 0.3)
            durations[i][w] = d
    return durations

train_durations = generate_durations(num_train_scen)
test_durations = generate_durations(num_test_scen)

# ==============
# 3. 策略 C：SAA 模型 (M1) 求解
# ==============

def solve_policy_SAA(durations, T_value):
    """基于场景的 SAA 模型，返回分配方案 x[i,r]"""
    model = Model("policy_SAA")
    model.setParam("OutputFlag", 0)

    x = {}
    z = {}

    for i in range(num_patients):
        for r in range(num_OR):
            x[i, r] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{r}")

    num_scen = len(durations[0])
    for r in range(num_OR):
        for w in range(num_scen):
            z[r, w] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"z_{r}_{w}")

    model.update()

    # 每个病人分配到且仅到一个 OR
    for i in range(num_patients):
        model.addConstr(quicksum(x[i, r] for r in range(num_OR)) == 1)

    # 定义加班
    for r in range(num_OR):
        for w in range(num_scen):
            model.addConstr(
                quicksum(durations[i][w] * x[i, r] for i in range(num_patients)) - T_value <= z[r, w]
            )

    avg_ot = (1.0 / num_scen) * quicksum(z[r, w] for r in range(num_OR) for w in range(num_scen))
    model.setObjective(avg_ot, GRB.MINIMIZE)

    model.optimize()

    x_sol = [[0 for _ in range(num_OR)] for _ in range(num_patients)]
    for i in range(num_patients):
        for r in range(num_OR):
            if x[i, r].X > 0.5:
                x_sol[i][r] = 1

    return x_sol, model.ObjVal

# ==============
# 4. 策略 A / B：确定性模型（不区分类型 / 区分类型）
# ==============

def solve_policy_deterministic(d_mean, T_value, name="deterministic"):
    """d_mean[i] 是每个病人的确定时长"""
    model = Model(name)
    model.setParam("OutputFlag", 0)

    x = {}
    z = {}

    for i in range(num_patients):
        for r in range(num_OR):
            x[i, r] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{r}")

    for r in range(num_OR):
        z[r] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"z_{r}")

    model.update()

    for i in range(num_patients):
        model.addConstr(quicksum(x[i, r] for r in range(num_OR)) == 1)

    for r in range(num_OR):
        model.addConstr(
            quicksum(d_mean[i] * x[i, r] for i in range(num_patients)) - T_value <= z[r]
        )

    model.setObjective(quicksum(z[r] for r in range(num_OR)), GRB.MINIMIZE)

    model.optimize()

    x_sol = [[0 for _ in range(num_OR)] for _ in range(num_patients)]
    for i in range(num_patients):
        for r in range(num_OR):
            if x[i, r].X > 0.5:
                x_sol[i][r] = 1

    return x_sol, model.ObjVal

# ==============
# 5. 评估函数：在测试场景上评估某个分配方案
# ==============

def evaluate_policy(x_sol, durations_test, T_value):
    num_scen = len(durations_test[0])

    # 指标
    total_overtime = 0.0      # 所有场景、所有 OR 的加班总和
    max_overtime = 0.0        # 所有场景中最大的 OR 加班
    count_overtime = 0        # 有加班的场景次数（至少一个 OR 加班）
    count_overtime_big = 0    # 有 OR 加班超过 0.5 小时的场景次数

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
        if scen_max_or_ot > 0.5:
            count_overtime_big += 1

    TE = total_overtime / num_scen                     # Total Expected overtime
    WO = max_overtime                                  # Worst-case overtime
    JP = count_overtime / num_scen                     # Joint overtime prob
    JPT = count_overtime_big / num_scen                # Joint overtime >0.5h

    return TE, WO, JP, JPT

# ==============
# 6. 主流程：三种策略对比
# ==============

# 策略 B：区分类型，但视为均值确定
mean_by_type = {}
for t, (m, s) in patient_types.items():
    mean_by_type[t] = m

d_mean_type = [mean_by_type[patient_type[i]] for i in range(num_patients)]

# 策略 A：不区分类型，统一均值
global_mean = sum(d_mean_type) / len(d_mean_type)
d_mean_global = [global_mean for _ in range(num_patients)]

# 解策略 A、B、C
x_A, obj_A = solve_policy_deterministic(d_mean_global, T, name="policy_A_global")
x_B, obj_B = solve_policy_deterministic(d_mean_type, T, name="policy_B_type")
x_C, obj_C = solve_policy_SAA(train_durations, T)

print("Training objective (A, B, C):", obj_A, obj_B, obj_C)

# 在测试场景上评估
TE_A, WO_A, JP_A, JPT_A = evaluate_policy(x_A, test_durations, T)
TE_B, WO_B, JP_B, JPT_B = evaluate_policy(x_B, test_durations, T)
TE_C, WO_C, JP_C, JPT_C = evaluate_policy(x_C, test_durations, T)

print("Policy A (global mean): TE, WO, JP, JPT =",
      TE_A, WO_A, JP_A, JPT_A)
print("Policy B (type mean):   TE, WO, JP, JPT =",
      TE_B, WO_B, JP_B, JPT_B)
print("Policy C (SAA):         TE, WO, JP, JPT =",
      TE_C, WO_C, JP_C, JPT_C)

# 简单画一张柱状图比较 TE
labels = ["A_global", "B_type", "C_SAA"]
TE_values = [TE_A, TE_B, TE_C]

plt.figure()
plt.bar(labels, TE_values)
plt.ylabel("Expected total overtime")
plt.title("Comparison of policies (TE)")
plt.grid(axis="y")
plt.show()
