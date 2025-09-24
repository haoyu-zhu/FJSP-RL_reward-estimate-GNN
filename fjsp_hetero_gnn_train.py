# fjsp_hetero_gnn_train.py
# 从 dataset/fjsp100_hetero.jsonl 读取实例，
# 为每个实例构造异构图（Operation 节点 + Machine 节点；两类边）并训练回归 GNN 预测 makespan

import os, json, random, math
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------- 反序列化 ----------
def dict_to_instance(d: dict):
    # 返回 (jobs, machine_count)
    jobs = []
    for jobd in d["jobs"]:
        ops = []
        for opd in jobd["operations"]:
            # keys 可能是字符串，统一成 int
            durations = {int(k): int(v) for k, v in opd["durations"].items()}
            machines = [int(m) for m in opd["machines"]]
            ops.append({"machines": machines, "durations": durations})
        jobs.append({"operations": ops})
    return jobs, int(d["machine_count"])

# ---------- 构图（异构） ----------
def build_hetero_graph(jobs: List[Dict], M: int):
    """
    返回：
      X_op: [Nop, F_op]   工序特征
      X_m:  [M,   F_m]    机器特征
      E_seq:   [2, E1]    工序-工序 顺序边（双向）
      E_op2m:  [2, E2]    工序->机器 可行边（有向；训练时会自动加反向消息）
    """
    # --- 工序节点索引 ---
    node_of = []  # [(j,k)]
    for j, job in enumerate(jobs):
        for k, _ in enumerate(job["operations"]):
            node_of.append((j, k))
    Nop = len(node_of)
    idx_of = {(j, k): i for i, (j, k) in enumerate(node_of)}

    # --- 工序特征 ---
    # features: [min, max, mean, std, num_machines, op_index, job_len, pos_norm]
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

    # --- 工序顺序边（双向） ---
    edges_seq = []
    for j, job in enumerate(jobs):
        for k in range(len(job["operations"]) - 1):
            a, b = idx_of[(j, k)], idx_of[(j, k + 1)]
            edges_seq.append((a, b))
            edges_seq.append((b, a))
    E_seq = torch.tensor(edges_seq, dtype=torch.long).t().contiguous() if edges_seq else torch.empty((2,0), dtype=torch.long)

    # --- 工序 -> 机器 可行边 ---
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

    # --- 机器特征 ---
    # features: [id_norm, feasible_deg, mean_dur_on_incident_ops]
    X_m = []
    for m in range(M):
        id_norm = m / max(1, M-1) if M>1 else 0.0
        deg = float(deg_m[m])
        mean_dur = float(sumdur_m[m] / deg) if deg > 0 else 0.0
        X_m.append([id_norm, deg, mean_dur])
    X_m = torch.tensor(X_m, dtype=torch.float32)

    return X_op, X_m, E_seq, E_op2m

# ---------- 数据加载 ----------
class GraphItem:
    def __init__(self, X_op, X_m, E_seq, E_op2m, y):
        self.X_op = X_op           # [Nop, Fop]
        self.X_m  = X_m            # [M, Fm]
        self.E_seq   = E_seq       # [2, E1]
        self.E_op2m  = E_op2m      # [2, E2]
        self.y = torch.tensor([float(y)], dtype=torch.float32)

def load_dataset(jsonl_path: str) -> List[GraphItem]:
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            jobs, M = dict_to_instance(rec["instance"])
            Xop, Xm, Eseq, Eop2m = build_hetero_graph(jobs, M)
            items.append(GraphItem(Xop, Xm, Eseq, Eop2m, rec["label_makespan"]))
    return items

# ---------- 异构 GNN 层（纯 PyTorch） ----------
def mean_aggregate(messages: torch.Tensor, dst_index: torch.Tensor, num_nodes: int):
    out = torch.zeros((num_nodes, messages.size(1)), device=messages.device, dtype=messages.dtype)
    out.index_add_(0, dst_index, messages)
    deg = torch.bincount(dst_index, minlength=num_nodes).clamp(min=1).unsqueeze(1).to(messages.dtype)
    return out / deg

class HeteroGCNLayer(nn.Module):
    """
    更新规则（统一维度）：
      先把 H_op, H_m 线性投影到 out_dim：
        H_op_proj = W_op * H_op
        H_m_proj  = W_m  * H_m
      再做三路聚合到工序，和一路聚合到机器：
        H_op' = ReLU( H_op_proj + mean_seq(H_op_proj) + mean_m2op(H_m_proj) )
        H_m'  = ReLU( H_m_proj  + mean_op2m(H_op_proj) )
    """
    def __init__(self, in_op, in_m, out_dim):
        super().__init__()
        self.lin_op = nn.Linear(in_op, out_dim)
        self.lin_m  = nn.Linear(in_m,  out_dim)

    def forward(self, H_op, H_m, E_seq, E_op2m):
        # 统一投影到 out_dim
        H_op_proj = self.lin_op(H_op)   # [Nop, D]
        H_m_proj  = self.lin_m(H_m)     # [M,   D]

        # 工序-工序顺序边（双向已构造）
        if E_seq.numel() > 0:
            src, dst = E_seq[0], E_seq[1]
            msg_seq = H_op_proj[src]                                 # [E1, D]
            agg_seq_to_op = mean_aggregate(msg_seq, dst, H_op_proj.size(0))  # [Nop, D]
        else:
            agg_seq_to_op = torch.zeros_like(H_op_proj)

        # 工序->机器、机器->工序（用投影后的特征做消息）
        if E_op2m.numel() > 0:
            src_op, dst_m = E_op2m[0], E_op2m[1]
            # 到机器
            msg_op2m = H_op_proj[src_op]                              # [E2, D]
            agg_to_m = mean_aggregate(msg_op2m, dst_m, H_m_proj.size(0))      # [M, D]
            # 反向到工序
            msg_m2op = H_m_proj[dst_m]                                # [E2, D]
            agg_to_op = mean_aggregate(msg_m2op, src_op, H_op_proj.size(0))   # [Nop, D]
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
        # 第一层：输入维度到 hidden
        self.layers.append(HeteroGCNLayer(f_op, f_m, hidden))
        # 后续层：hidden -> hidden
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
        # 池化（平均）
        g_op = H_op.mean(dim=0, keepdim=True)   # [1, hidden]
        g_m  = H_m.mean(dim=0, keepdim=True)    # [1, hidden]
        g = torch.cat([g_op, g_m], dim=1)       # [1, 2*hidden]
        y = self.head(g).squeeze(0)             # [1] -> scalar
        return y

# ---------- 训练 ----------
def train(jsonl_path="dataset/fjsp100_hetero.jsonl", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    items = load_dataset(jsonl_path)
    print(f"Loaded {len(items)} graphs.")

    # 切分
    rng = random.Random(42)
    idx = list(range(len(items))); rng.shuffle(idx)
    n = len(idx); n_train = int(0.7*n); n_val = int(0.1*n)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    # 统计训练集特征均值方差（分别对 op 与 m）
    Xop_cat = torch.cat([items[i].X_op for i in train_idx], dim=0)
    Xm_cat  = torch.cat([items[i].X_m  for i in train_idx], dim=0)
    op_mean, op_std = Xop_cat.mean(0, keepdim=True), Xop_cat.std(0, keepdim=True).clamp(min=1e-6)
    m_mean,  m_std  = Xm_cat.mean(0, keepdim=True),  Xm_cat.std(0, keepdim=True).clamp(min=1e-6)
    def norm_op(X): return (X - op_mean) / op_std
    def norm_m(X):  return (X - m_mean)  / m_std

    f_op, f_m = items[0].X_op.size(1), items[0].X_m.size(1)
    model = HeteroGNNRegressor(f_op, f_m, hidden=64, layers=3, dropout=0.1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    def eval_split(split):
        model.eval()
        ys, yh = [], []
        with torch.no_grad():
            for i in split:
                Xi_op = norm_op(items[i].X_op).to(device)
                Xi_m  = norm_m(items[i].X_m).to(device)
                Ei_seq  = items[i].E_seq.to(device)
                Ei_op2m = items[i].E_op2m.to(device)
                y_hat = model(Xi_op, Xi_m, Ei_seq, Ei_op2m)[0]
                ys.append(items[i].y.item()); yh.append(y_hat.item())
        ys = np.array(ys, float); yh = np.array(yh, float)
        mse = float(np.mean((ys - yh)**2)); mae = float(np.mean(np.abs(ys - yh)))
        mape = float(np.mean(np.abs(ys - yh) / np.maximum(ys, 1e-6))) * 100.0
        return mse, mae, mape, ys, yh

    best_val = float("inf"); best = None
    epochs = 150
    for ep in range(1, epochs+1):
        model.train()
        rng.shuffle(train_idx)
        total = 0.0
        for i in train_idx:
            Xi_op = norm_op(items[i].X_op).to(device)
            Xi_m  = norm_m(items[i].X_m).to(device)
            Ei_seq  = items[i].E_seq.to(device)
            Ei_op2m = items[i].E_op2m.to(device)
            y = items[i].y.to(device)
            y_hat = model(Xi_op, Xi_m, Ei_seq, Ei_op2m)
            loss = loss_fn(y_hat, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if ep % 10 == 0 or ep == 1:
            val_mse, val_mae, val_mape, _, _ = eval_split(val_idx)
            print(f"[Epoch {ep:3d}] train_loss={total/len(train_idx):.4f} | "
                  f"val_MSE={val_mse:.2f} val_MAE={val_mae:.2f} val_MAPE={val_mape:.2f}%")
            if val_mse < best_val:
                best_val = val_mse
                best = (ep, {k: v.cpu().clone() for k, v in model.state_dict().items()})

    if best is not None:
        ep_best, state = best
        model.load_state_dict(state)
        print(f"Load best @epoch {ep_best} (val_MSE={best_val:.2f})")

    test_mse, test_mae, test_mape, ys, yh = eval_split(test_idx)
    print(f"\n[TEST] MSE={test_mse:.2f}  MAE={test_mae:.2f}  MAPE={test_mape:.2f}%  (N={len(test_idx)})")

    os.makedirs("models", exist_ok=True)
    torch.save({
        "state": model.state_dict(),
        "f_op": f_op, "f_m": f_m,
        "op_mean": op_mean, "op_std": op_std,
        "m_mean": m_mean, "m_std": m_std,
    }, "models/fjsp_hetero_gnn_5-10_5-20_10-20.pt")
    print("Saved model to models/fjsp_hetero_gnn_5-10_5-20_10-20.pt")

    # 保存测试集预测对比
    pd.DataFrame({"y_true": ys, "y_pred": yh}).to_csv("dataset/test_predictions_hetero.csv", index=False)
    print("Saved predictions to dataset/test_predictions_hetero.csv")

if __name__ == "__main__":
    # 默认读取脚本A生成的文件
    train("dataset/fjsp100_hetero.jsonl")
