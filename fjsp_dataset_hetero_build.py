# fjsp_dataset_hetero_build.py
# 生成 100 个 FJSP 实例（机器相关时长），用 OR-Tools 限时 10s 求解，
# 仅保留 OPTIMAL/FEASIBLE 的样本，保存到 dataset/fjsp100_hetero.jsonl + summary.csv

import os, json, math, random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt


# ---------- 数据结构 ----------
@dataclass
class Operation:
    machines: List[int]            # 可行机器ID
    durations: Dict[int, int]      # 每机器加工时间 {m: dur}

@dataclass
class Job:
    operations: List[Operation]

@dataclass
class FJSPInstance:
    jobs: List[Job]
    machine_count: int
    seed: int

# ---------- 随机实例（机器相关时间：base∈[1,20]，每机=round(base*U[0.8,1.2])>=1） ----------
def generate_fjsp_instance(
    job_count=20, machine_count=10, ops_min=8, ops_max=12,
    comp_min=1, comp_max=10, base_min=1, base_max=20,
    factor_lo=0.8, factor_hi=1.2, rng: random.Random=None, seed=0
) -> FJSPInstance:
    if rng is None: rng = random.Random(seed)
    jobs = []
    for _ in range(job_count):
        op_cnt = rng.randint(ops_min, ops_max)
        ops = []
        for _k in range(op_cnt):
            comp_size = max(1, min(machine_count, rng.randint(comp_min, comp_max)))
            machines = sorted(rng.sample(range(machine_count), comp_size))
            base = rng.randint(base_min, base_max)
            durations = {}
            for m in machines:
                factor = rng.uniform(factor_lo, factor_hi)
                dur = max(1, int(round(base * factor)))
                durations[m] = dur
            ops.append(Operation(machines=machines, durations=durations))
        jobs.append(Job(operations=ops))
    return FJSPInstance(jobs=jobs, machine_count=machine_count, seed=seed)

# ---------- OR-Tools CP-SAT 求解（含正确顺序约束） ----------
def solve_fjsp_cp_sat(instance: FJSPInstance, time_limit_s: float = 10.0):
    model = cp_model.CpModel()
    jobs = instance.jobs
    M = instance.machine_count

    horizon = sum(max(op.durations.values()) for job in jobs for op in job.operations)
    horizon = max(1, horizon)

    start, end, pres, interval = {}, {}, {}, {}
    machine_to_intervals = {m: [] for m in range(M)}
    op_alts = {}

    for j, job in enumerate(jobs):
        for k, op in enumerate(job.operations):
            op_alts[(j, k)] = list(op.machines)
            lits = []
            for m in op.machines:
                d = op.durations[m]
                s = model.NewIntVar(0, horizon, f"s_j{j}_o{k}_m{m}")
                e = model.NewIntVar(0, horizon, f"e_j{j}_o{k}_m{m}")
                x = model.NewBoolVar(f"x_j{j}_o{k}_m{m}")
                itv = model.NewOptionalIntervalVar(s, d, e, x, f"iv_j{j}_o{k}_m{m}")
                start[(j,k,m)], end[(j,k,m)], pres[(j,k,m)] = s, e, x
                interval[(j,k,m)] = itv
                machine_to_intervals[m].append(itv)
                lits.append(x)
            model.AddExactlyOne(lits)

    for m in range(M):
        if machine_to_intervals[m]:
            model.AddNoOverlap(machine_to_intervals[m])

    # 相邻工序顺序：k -> k+1，仅在各自机器被选中时生效
    for j, job in enumerate(jobs):
        for k in range(len(job.operations)-1):
            for m in op_alts[(j,k)]:
                for mp in op_alts[(j,k+1)]:
                    model.Add(start[(j,k+1,mp)] >= end[(j,k,m)]).OnlyEnforceIf(
                        [pres[(j,k,m)], pres[(j,k+1,mp)]]
                    )

    makespan = model.NewIntVar(0, horizon, "makespan")
    for j, job in enumerate(jobs):
        for k, op in enumerate(job.operations):
            for m in op.machines:
                model.Add(end[(j,k,m)] <= makespan).OnlyEnforceIf(pres[(j,k,m)])
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8

    status_code = solver.Solve(model)
    status_str = solver.StatusName(status_code)
    schedule = []
    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for j, job in enumerate(jobs):
            for k, op in enumerate(job.operations):
                for m in op.machines:
                    if solver.Value(pres[(j,k,m)]) == 1:
                        s, e = solver.Value(start[(j,k,m)]), solver.Value(end[(j,k,m)])
                        d = op.durations[m]
                        schedule.append((j,k,m,s,e,d))
    obj = solver.ObjectiveValue() if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE) else math.nan
    return status_str, obj, schedule

# ---------- 序列化（保存实例到 JSON） ----------
def instance_to_dict(inst: FJSPInstance) -> dict:
    return {
        "machine_count": inst.machine_count,
        "seed": inst.seed,
        "jobs": [
            {"operations": [
                {"machines": op.machines, "durations": op.durations}
                for op in job.operations
            ]} for job in inst.jobs
        ]
    }

def plot_gantt(schedule, machine_count, title, out_png):
    """
    schedule: list of (job, op, machine, start, end, dur)
    按机器为轨道，条块按工件着色；条上标注 J#-O#
    """
    # 空调度也生成一张标注图，方便排查
    if not schedule:
        plt.figure(figsize=(10, 3))
        plt.title(title + " (No schedule)")
        plt.xlabel("Time")
        plt.ylabel("Machine")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        return

    # 为每个工件分配固定颜色
    jobs = sorted({j for (j, k, m, s, e, d) in schedule})
    cmap = plt.get_cmap("tab20")
    job_color = {j: cmap(j % 20) for j in jobs}

    # 按机器分组
    by_machine = {m: [] for m in range(machine_count)}
    for (j, k, m, s, e, d) in schedule:
        by_machine[m].append((j, k, s, e, d))

    plt.figure(figsize=(12, 4 + machine_count * 0.3))
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")

    yticks, ylabels = [], []
    for m in range(machine_count):
        yticks.append(m)
        ylabels.append(f"M{m}")
        rows = by_machine[m]
        for (j, k, s, e, d) in rows:
            ax.barh(m, e - s, left=s, height=0.6,
                    edgecolor="black", color=job_color[j])
            ax.text(s + (e - s) / 2, m, f"J{j}-O{k}",
                    ha="center", va="center", fontsize=8)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()  # M0 在最上
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    # 图例（工件颜色）
    handles = [plt.Line2D([0], [0], marker="s", linestyle="",
                          markersize=8, markerfacecolor=job_color[j],
                          markeredgecolor="black", label=f"Job {j}") for j in jobs]
    if handles:
        ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left", title="Jobs")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ---------- 主：构建数据集 ----------
def build_dataset(out_dir="dataset", num_needed=60, max_tries=100, time_limit_s=100.0, base_seed=20250924123):
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "fjsp100_hetero.jsonl")
    csv_path   = os.path.join(out_dir, "fjsp100_summary.csv")

    rng = random.Random(base_seed)
    kept, tried = 0, 0
    rows = []
    gantt_dir = os.path.join(out_dir, "gantt_train")
    os.makedirs(gantt_dir, exist_ok=True)

    with open(jsonl_path, "a", encoding="utf-8") as f:
        while kept < num_needed and tried < max_tries:
            tried += 1
            seed = base_seed + tried
            inst = generate_fjsp_instance(rng=rng, seed=seed)
            status, makespan, schedule = solve_fjsp_cp_sat(inst, time_limit_s=time_limit_s)

            if status in ("OPTIMAL", "FEASIBLE"):
                rec = {
                    "instance": instance_to_dict(inst),
                    "label_makespan": int(makespan),
                    "status": status
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # —— 新增：保存甘特图 ——
                png_name = f"gantt_{kept + 1:03d}_ms{int(makespan)}.png"
                png_path = os.path.join(gantt_dir, png_name)
                title = f"FJSP seed={seed} | makespan={int(makespan)}"
                plot_gantt(schedule, inst.machine_count, title, png_path)

                rows.append({
                    "id": kept + 1,
                    "jobs": len(inst.jobs),
                    "total_ops": sum(len(job.operations) for job in inst.jobs),
                    "machines": inst.machine_count,
                    "makespan": int(makespan),
                    "status": status,
                    "gantt_png": os.path.relpath(png_path, start=out_dir),  # 可选：把图路径也写到 summary
                })
                kept += 1
                print(
                    f"[dataset] kept {kept}/{num_needed} (try {tried}) makespan={int(makespan)} status={status} -> {png_name}")
            else:
                print(f"[dataset] skip (try {tried}) status={status}")

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved {kept} samples to:\n- {jsonl_path}\n- {csv_path}")




    return jsonl_path

if __name__ == "__main__":
    build_dataset()
