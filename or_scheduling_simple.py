import random
import math
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt

# ======================
# 1. 基本参数设置
# ======================

random.seed(0)

num_OR = 3          # 手术室数量
num_patients = 12   # 病人数
T = 8.0             # 每个 OR 正常工作时间（小时）
num_scenarios = 50  # 场景数（样本数）

# 患者类型设置：类型 -> (均值, 标准差)
patient_types = {
    0: (1.0, 0.2),  # 简单手术，平均 1 小时
    1: (1.5, 0.3),  # 中等
    2: (2.0, 0.4)   # 困难
}

# 为每个病人随机指定一个类型
patient_type = {}
for i in range(num_patients):
    # 简单写法：大部分是简单/中等，少数是困难
    t = random.choices([0, 1, 2], weights=[0.5, 0.3, 0.2])[0]
    patient_type[i] = t

# ======================
# 2. 生成场景下的手术时长 d[i][omega]
# ======================

durations = [[0.0 for _ in range(num_scenarios)] for _ in range(num_patients)]

for i in range(num_patients):
    t = patient_type[i]
    mean, std = patient_types[t]
    for w in range(num_scenarios):
        # 用正态分布生成一个样本，截断到 >= 0.2 小时
        d = random.gauss(mean, std)
        d = max(d, 0.2)
        durations[i][w] = d

# ======================
# 3. 建立模型
# ======================

model = Model("OR_scheduling_simple")

# 决策变量 x[i,r]: 病人 i 是否分配到 OR r
x = {}
for i in range(num_patients):
    for r in range(num_OR):
        x[i, r] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{r}")

# 决策变量 z[r,w]: 场景 w 下，OR r 的加班时间
z = {}
for r in range(num_OR):
    for w in range(num_scenarios):
        z[r, w] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"z_{r}_{w}")

model.update()

# ======================
# 4. 约束
# ======================

# 每个病人必须分配到且仅到一个 OR
for i in range(num_patients):
    model.addConstr(quicksum(x[i, r] for r in range(num_OR)) == 1, name=f"assign_{i}")

# 定义每个场景下的加班量
for r in range(num_OR):
    for w in range(num_scenarios):
        # sum_i d[i][w] * x[i,r] - T <= z[r,w]
        model.addConstr(
            quicksum(durations[i][w] * x[i, r] for i in range(num_patients)) - T <= z[r, w],
            name=f"overtime_{r}_{w}"
        )

# ======================
# 5. 目标：最小化平均加班时间
# ======================

avg_overtime = (1.0 / num_scenarios) * quicksum(z[r, w] for r in range(num_OR) for w in range(num_scenarios))
model.setObjective(avg_overtime, GRB.MINIMIZE)

# ======================
# 6. 求解
# ======================

model.optimize()

print("Optimal objective (平均加班时间):", model.ObjVal)

# 输出每个 OR 上的病人列表
for r in range(num_OR):
    assigned_patients = [i for i in range(num_patients) if x[i, r].X > 0.5]
    print(f"OR {r}: patients {assigned_patients}")

# ======================
# 7. 简单可视化：不同 T 下的平均加班
# ======================

def solve_for_T(T_value):
    # 重新建模，复用 durations 等数据，只改 T
    m = Model("OR_scheduling_T")
    x2 = {}
    z2 = {}

    for i in range(num_patients):
        for r in range(num_OR):
            x2[i, r] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{r}")
    for r in range(num_OR):
        for w in range(num_scenarios):
            z2[r, w] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"z_{r}_{w}")

    m.update()

    for i in range(num_patients):
        m.addConstr(quicksum(x2[i, r] for r in range(num_OR)) == 1)

    for r in range(num_OR):
        for w in range(num_scenarios):
            m.addConstr(
                quicksum(durations[i][w] * x2[i, r] for i in range(num_patients)) - T_value <= z2[r, w]
            )

    avg_ot = (1.0 / num_scenarios) * quicksum(z2[r, w] for r in range(num_OR) for w in range(num_scenarios))
    m.setObjective(avg_ot, GRB.MINIMIZE)
    m.setParam("OutputFlag", 0)  # 不输出详细日志
    m.optimize()

    return m.ObjVal

T_values = [6.0, 7.0, 8.0, 9.0, 10.0]
avg_overtimes = []

for Tv in T_values:
    val = solve_for_T(Tv)
    avg_overtimes.append(val)
    print(f"T = {Tv}, avg overtime = {val}")

plt.figure()
plt.plot(T_values, avg_overtimes, marker="o")
plt.xlabel("Regular working time T (hours)")
plt.ylabel("Average overtime (hours)")
plt.title("Effect of T on average overtime")
plt.grid(True)
plt.show()

