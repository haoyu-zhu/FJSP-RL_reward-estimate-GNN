# fjsp_hetero_gnn_test.py
# 加载已训练的异构GNN，读取JSONL数据，评估并保存预测CSV

import os, json, argparse, random
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ========= 数据反序列化 & 构图（与训练脚本一致） =========
def dict_to_instance(d: dict):
    jobs = []
    for jobd in d["jobs"]:
        ops = []
        for opd in jobd["operations"]:
            durations = {int(k): int(v) for k, v in opd["durations"].items()}
            machines = [int(m) for m in opd["machines"]]
            ops.append({"machines": machines, "durations": durations})
        jobs.append({"operations": ops})
    return jobs, int(d["machine_count"])

def build_hetero_graph(jobs: List[Dict], M: int):
    # Operation 节点索引
    node_of = []
    for j, job in enumerate(jobs):
        for k, _ in enumerate(job["operations"]):
            node_of.append((j, k))
    Nop = len(node_of)
    idx_of = {(j, k): i for i, (j, k) in enumerate(node_of)}

    # Operation 特征
    X_op = []
    job_lens = [len(job["operations"]) for job in jobs]
    for j, k in node_of:
        op = jobs[j]["operations"][k]
        durs = [op["durations"][m] for m in op["machines"]]
        mn = float(np.min(durs)); mx = float(np.max(durs))
        mean = float(np.mean(durs)); std = float(np.std(durs)) if len(durs) > 1 else 0.0
        nm = float(len(op["machines"]))
        job_len = float(job_lens[j])
        pos = float(k)
        pos_norm = (pos / max(1.0, job_len - 1.0)) if job_len > 1 else 0.0
        X_op.append([mn, mx, mean, std, nm, pos, job_len, pos_norm])
    X_op = torch.tensor(X_op, dtype=torch.float32)

    # 工序顺序边（双向）
    edges_seq = []
    for j, job in enumerate(jobs):
        for k in range(len(job["operations"]) - 1):
            a, b = idx_of[(j, k)], idx_of[(j, k + 1)]
            edges_seq.append((a, b))
            edges_seq.append((b, a))
    E_seq = torch.tensor(edges_seq, dtype=torch.long).t().contiguous() if edges_seq else torch.empty((2,0), dtype=torch.long)

    # Operation->Machine 可行边，同时统计机器度和平均时长
    edges_op2m = []
    deg_m = [0]*M
    sumdur_m = [0.0]*M
    for j, k in node_of:
        i = idx_of[(j, k)]
        op = jobs[j]["operations"][k]
        for m in op["machines"]:
            edges_op2m.append((i, m))
            deg_m[m] += 1
            sumdur_m[m] += float(op["durations"][m])
    E_op2m = torch.tensor(edges_op2m, dtype=torch.long).t().contiguous() if edges_op2m else torch.empty((2,0), dtype=torch.long)

    # Machine 特征
    X_m = []
    for m in range(M):
        id_norm = m / max(1, M-1) if M>1 else 0.0
        deg = float(deg_m[m])
        mean_dur = float(sumdur_m[m] / deg) if deg > 0 else 0.0
        X_m.append([id_norm, deg, mean_dur])
    X_m = torch.tensor(X_m, dtype=torch.float32)

    return X_op, X_m, E_seq, E_op2m, Nop, M

class GraphItem:
    def __init__(self, X_op, X_m, E_seq, E_op2m, y, n_ops, n_machines):
        self.X_op = X_op
        self.X_m = X_m
        self.E_seq = E_seq
        self.E_op2m = E_op2m
        self.y = torch.tensor([float(y)], dtype=torch.float32)
        self.n_ops = n_ops
        self.n_machines = n_machines

def load_dataset(paths) -> List[GraphItem]:
    if isinstance(paths, str): paths = [paths]
    items = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                jobs, M = dict_to_instance(rec["instance"])
                Xop, Xm, Eseq, Eop2m, Nop, MM = build_hetero_graph(jobs, M)
                items.append(GraphItem(Xop, Xm, Eseq, Eop2m, rec["label_makespan"], Nop, MM))
    return items

# ========= 异构GNN（与训练脚本一致，含维度修复版本） =========
def mean_aggregate(messages: torch.Tensor, dst_index: torch.Tensor, num_nodes: int):
    out = torch.zeros((num_nodes, messages.size(1)), device=messages.device, dtype=messages.dtype)
    out.index_add_(0, dst_index, messages)
    deg = torch.bincount(dst_index, minlength=num_nodes).clamp(min=1).unsqueeze(1).to(messages.dtype)
    return out / deg

class HeteroGCNLayer(nn.Module):
    def __init__(self, in_op, in_m, out_dim):
        super().__init__()
        self.lin_op = nn.Linear(in_op, out_dim)
        self.lin_m  = nn.Linear(in_m,  out_dim)
    def forward(self, H_op, H_m, E_seq, E_op2m):
        H_op_proj = self.lin_op(H_op)
        H_m_proj  = self.lin_m(H_m)
        # seq 聚合到 op
        if E_seq.numel() > 0:
            src, dst = E_seq[0], E_seq[1]
            agg_seq_to_op = mean_aggregate(H_op_proj[src], dst, H_op_proj.size(0))
        else:
            agg_seq_to_op = torch.zeros_like(H_op_proj)
        # op<->m 聚合
        if E_op2m.numel() > 0:
            src_op, dst_m = E_op2m[0], E_op2m[1]
            agg_to_m  = mean_aggregate(H_op_proj[src_op], dst_m, H_m_proj.size(0))
            agg_to_op = mean_aggregate(H_m_proj[dst_m], src_op, H_op_proj.size(0))
        else:
            agg_to_m  = torch.zeros_like(H_m_proj)
            agg_to_op = torch.zeros_like(H_op_proj)
        H_op_new = torch.relu(H_op_proj + agg_seq_to_op + agg_to_op)
        H_m_new  = torch.relu(H_m_proj  + agg_to_m)
        return H_op_new, H_m_new

class HeteroGNNRegressor(nn.Module):
    def __init__(self, f_op, f_m, hidden=64, layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HeteroGCNLayer(f_op, f_m, hidden))
        for _ in range(layers - 1):
            self.layers.append(HeteroGCNLayer(hidden, hidden, hidden))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(2*hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, X_op, X_m, E_seq, E_op2m):
        H_op, H_m = X_op, X_m
        for i, layer in enumerate(self.layers):
            H_op, H_m = layer(H_op, H_m, E_seq, E_op2m)
            if i < len(self.layers)-1:
                H_op = self.dropout(H_op)
                H_m  = self.dropout(H_m)
        g_op = H_op.mean(dim=0, keepdim=True)
        g_m  = H_m.mean(dim=0, keepdim=True)
        g = torch.cat([g_op, g_m], dim=1)
        y = self.head(g).squeeze(0)   # [1]
        return y

# ========= 测试 =========
def evaluate(paths, ckpt_path, out_csv="dataset/test_predictions_eval_3.csv", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    items = load_dataset(paths)
    if len(items) == 0:
        raise RuntimeError("No graphs loaded. Check your JSONL path(s).")
    print(f"Loaded {len(items)} graphs for evaluation.")

    # 读取checkpoint（与训练脚本的保存格式一致）
    ckpt = torch.load(ckpt_path, map_location=device)
    f_op, f_m = ckpt["f_op"], ckpt["f_m"]
    op_mean, op_std = ckpt["op_mean"], ckpt["op_std"]
    m_mean,  m_std  = ckpt["m_mean"],  ckpt["m_std"]

    def norm_op(X): return (X - op_mean) / op_std
    def norm_m(X):  return (X - m_mean)  / m_std

    model = HeteroGNNRegressor(f_op, f_m, hidden=64, layers=3, dropout=0.0).to(device)
    state_key = "state" if "state" in ckpt else "model_state"  # 兼容旧命名
    model.load_state_dict(ckpt[state_key], strict=True)
    model.eval()

    ys, yhats, nops, nms = [], [], [], []
    with torch.no_grad():
        for it in items:
            Xi_op = norm_op(it.X_op).to(device)
            Xi_m  = norm_m(it.X_m).to(device)
            Ei_seq  = it.E_seq.to(device)
            Ei_op2m = it.E_op2m.to(device)
            y_hat = model(Xi_op, Xi_m, Ei_seq, Ei_op2m)[0]
            ys.append(it.y.item()); yhats.append(y_hat.item())
            nops.append(it.n_ops); nms.append(it.n_machines)

    ys = np.array(ys, dtype=float)
    yh = np.array(yhats, dtype=float)
    mse  = float(np.mean((ys - yh)**2))
    mae  = float(np.mean(np.abs(ys - yh)))
    mape = float(np.mean(np.abs(ys - yh) / np.maximum(ys, 1e-6))) * 100.0
    print(f"[EVAL] MSE={mse:.2f}  MAE={mae:.2f}  MAPE={mape:.2f}%  (N={len(items)})")
    ys = np.array(ys, dtype=float)
    yh = np.array(yhats, dtype=float)

    # mse  = float(np.mean((ys - yh)**2))
    # mae  = float(np.mean(np.abs(ys - yh)))
    # mape = float(np.mean(np.abs(ys - yh) / np.maximum(ys, 1e-6))) * 100.0
    # ss_res = np.sum((ys - yh)**2)
    # ss_tot = np.sum((ys - np.mean(ys))**2)
    # r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    #
    # print(f"[EVAL] MSE={mse:.2f}  MAE={mae:.2f}  MAPE={mape:.2f}%  R2={r2:.3f}  (N={len(items)})")
    ys = np.array(ys, dtype=float)
    yh = np.array(yhats, dtype=float)

    mse  = float(np.mean((ys - yh)**2))
    mae  = float(np.mean(np.abs(ys - yh)))
    mape = float(np.mean(np.abs(ys - yh) / np.maximum(ys, 1e-6))) * 100.0
    ss_res = np.sum((ys - yh)**2)
    ss_tot = np.sum((ys - np.mean(ys))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"[EVAL] MSE={mse:.2f}  MAE={mae:.2f}  MAPE={mape:.2f}%  R2={r2:.3f}  (N={len(items)})")

    # 保存预测CSV
    df = pd.DataFrame({
        "graph_id": np.arange(1, len(items)+1),
        "y_true": ys,
        "y_pred": yh,
        "abs_err": np.abs(ys - yh),
        "rel_err_pct": np.abs(ys - yh) / np.maximum(ys, 1e-6) * 100.0,
        "num_ops": nops,
        "num_machines": nms,
    })
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=["dataset/fjsp100_hetero_mess.jsonl"]
                        , help="one or more JSONL files")
    parser.add_argument("--ckpt", default=
                        "models/fjsp_hetero_gnn2.pt"
                        #"fjsp_hetero_gnn_5-10_5-20_10-20.pt"
                        , help="checkpoint path")
    parser.add_argument("--out",  default="dataset/test_predictions_eval3.csv", help="output CSV path")
    parser.add_argument("--device", default=None, help="cpu / cuda")
    args = parser.parse_args()
    evaluate(args.data, args.ckpt, args.out, args.device)
