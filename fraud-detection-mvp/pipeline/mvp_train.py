import os
import math
import random
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from dateutil.parser import parse as parse_dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import DeepGraphInfomax


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def synthesize_data(
    data_dir: str,
    num_accounts: int = 200,
    fraud_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random.seed(seed)
    np.random.seed(seed)
    ensure_dir(data_dir)

    # Accounts
    account_ids = [f"A{i:05d}" for i in range(num_accounts)]
    start_ts = pd.Timestamp("2023-01-01T00:00:00Z")
    created_offsets = np.random.randint(0, 360, size=num_accounts)
    created_at = [start_ts + pd.Timedelta(int(d), unit="D") for d in created_offsets]
    accounts = pd.DataFrame({"account_id": account_ids, "created_at": created_at})

    # Labels
    num_fraud = max(5, int(num_accounts * fraud_ratio))
    fraud_indices = np.random.choice(num_accounts, size=num_fraud, replace=False)
    is_fraud = np.zeros(num_accounts, dtype=int)
    is_fraud[fraud_indices] = 1
    labels = pd.DataFrame({"account_id": account_ids, "is_fraud": is_fraud})

    # Transactions
    # Normal accounts: moderate degree, smaller amounts; Fraud: higher degree among themselves and larger amounts
    tx_rows = []
    current_ts = pd.Timestamp("2024-01-01T00:00:00Z")
    for i, src in enumerate(account_ids):
        src_is_fraud = is_fraud[i] == 1
        # Base number of transactions
        base_deg = np.random.poisson(6 if not src_is_fraud else 14)
        for _ in range(base_deg):
            if src_is_fraud and np.random.rand() < 0.6:
                # Prefer connect to fraud
                dst_idx = np.random.choice(fraud_indices)
            else:
                dst_idx = np.random.randint(0, num_accounts)
            if dst_idx == i:
                dst_idx = (dst_idx + 1) % num_accounts
            dst = account_ids[dst_idx]

            if src_is_fraud:
                amount = float(np.clip(np.random.lognormal(mean=3.8, sigma=0.6), 50, 50000))
            else:
                amount = float(np.clip(np.random.lognormal(mean=3.0, sigma=0.5), 5, 5000))

            # Randomize time around current_ts
            ts_offset_min = int(np.random.randint(-60 * 24 * 90, 60 * 24 * 90))
            ts = current_ts + pd.Timedelta(ts_offset_min, unit="m")
            tx_rows.append({
                "src_account_id": src,
                "dst_account_id": dst,
                "amount": amount,
                "ts": ts,
            })

    transactions = pd.DataFrame(tx_rows)

    # Devices: fraud share devices more to produce dense subgraph; normal mostly one-to-one devices
    device_rows = []
    device_id_counter = 1

    # Fraud clusters share a few devices
    fraud_groups = np.array_split(fraud_indices, max(1, math.ceil(len(fraud_indices) / 5)))
    for group in fraud_groups:
        if len(group) == 0:
            continue
        # Shared device for the group
        dev_id = f"D{device_id_counter:05d}"; device_id_counter += 1
        for idx in group:
            device_rows.append({"device_id": dev_id, "account_id": account_ids[idx]})
        # Each also has a personal device
        for idx in group:
            dev_id = f"D{device_id_counter:05d}"; device_id_counter += 1
            device_rows.append({"device_id": dev_id, "account_id": account_ids[idx]})

    # Normal users: mostly unique devices; a few small shared pairs
    normal_indices = [i for i in range(num_accounts) if is_fraud[i] == 0]
    for idx in normal_indices:
        dev_id = f"D{device_id_counter:05d}"; device_id_counter += 1
        device_rows.append({"device_id": dev_id, "account_id": account_ids[idx]})

    # Add a few small shared normal devices (pairs/triads)
    rng = np.random.default_rng(seed)
    for _ in range(max(2, num_accounts // 50)):
        group = rng.choice(normal_indices, size=min(3, len(normal_indices)), replace=False)
        dev_id = f"D{device_id_counter:05d}"; device_id_counter += 1
        for idx in group:
            device_rows.append({"device_id": dev_id, "account_id": account_ids[idx]})

    devices = pd.DataFrame(device_rows)

    # Save CSVs
    accounts.to_csv(os.path.join(data_dir, "accounts.csv"), index=False)
    transactions.to_csv(os.path.join(data_dir, "transactions.csv"), index=False)
    devices.to_csv(os.path.join(data_dir, "devices.csv"), index=False)
    labels.to_csv(os.path.join(data_dir, "labels.csv"), index=False)

    return accounts, transactions, devices, labels


def normalize_timeseries_cols(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col].apply(lambda x: parse_dt(str(x), fuzzy=True)), utc=True)


def build_account_index(accounts: pd.DataFrame) -> Dict[str, int]:
    account_ids = accounts["account_id"].astype(str).unique().tolist()
    return {acc_id: idx for idx, acc_id in enumerate(account_ids)}


def build_transaction_edges(tx: pd.DataFrame, account_to_idx: Dict[str, int], undirected: bool = True):
    tx = tx.copy()
    tx["src_account_id"] = tx["src_account_id"].astype(str)
    tx["dst_account_id"] = tx["dst_account_id"].astype(str)
    tx = tx[tx["src_account_id"].isin(account_to_idx) & tx["dst_account_id"].isin(account_to_idx)]

    src = tx["src_account_id"].map(account_to_idx).values
    dst = tx["dst_account_id"].map(account_to_idx).values
    amounts = tx.get("amount", pd.Series(np.ones(len(tx)))).values.astype(float)

    edges_src = src.tolist()
    edges_dst = dst.tolist()
    weights = amounts.copy()

    if undirected:
        edges_src = edges_src + dst.tolist()
        edges_dst = edges_dst + src.tolist()
        weights = np.concatenate([weights, amounts])

    return edges_src, edges_dst, weights


def build_device_edges(devices: pd.DataFrame, account_to_idx: Dict[str, int], undirected: bool = True):
    devices = devices.copy()
    devices["account_id"] = devices["account_id"].astype(str)
    devices = devices[devices["account_id"].isin(account_to_idx)]
    edges_src, edges_dst = [], []

    for device_id, group in devices.groupby("device_id"):
        accs = group["account_id"].astype(str).map(account_to_idx).dropna().astype(int).tolist()
        if len(accs) <= 1:
            continue
        for i in range(len(accs)):
            for j in range(i + 1, len(accs)):
                a, b = accs[i], accs[j]
                edges_src.append(a); edges_dst.append(b)
                if undirected:
                    edges_src.append(b); edges_dst.append(a)
    return edges_src, edges_dst


def compute_node_features(
    accounts: pd.DataFrame,
    tx: pd.DataFrame,
    devices: pd.DataFrame,
    account_to_idx: Dict[str, int]
) -> np.ndarray:
    num_nodes = len(account_to_idx)
    sent_counts = np.zeros(num_nodes)
    recv_counts = np.zeros(num_nodes)
    sent_amounts = np.zeros(num_nodes)
    recv_amounts = np.zeros(num_nodes)
    device_counts = np.zeros(num_nodes)
    account_age_days = np.zeros(num_nodes)

    if not tx.empty:
        tx = tx.copy()
        tx["src_account_id"] = tx["src_account_id"].astype(str)
        tx["dst_account_id"] = tx["dst_account_id"].astype(str)
        tx = tx[tx["src_account_id"].isin(account_to_idx) & tx["dst_account_id"].isin(account_to_idx)]
        amounts = tx.get("amount", pd.Series(np.ones(len(tx)))).astype(float).values

        for (src_id, dst_id, amt) in zip(tx["src_account_id"], tx["dst_account_id"], amounts):
            si = account_to_idx[src_id]; di = account_to_idx[dst_id]
            sent_counts[si] += 1
            recv_counts[di] += 1
            sent_amounts[si] += amt
            recv_amounts[di] += amt

    if not devices.empty:
        dev_cnt = devices.copy()
        dev_cnt["account_id"] = dev_cnt["account_id"].astype(str)
        dev_cnt = dev_cnt[dev_cnt["account_id"].isin(account_to_idx)]
        for acc_id, n in dev_cnt["account_id"].value_counts().items():
            device_counts[account_to_idx[acc_id]] = n

    if "created_at" in accounts.columns:
        accounts = accounts.copy()
        accounts["account_id"] = accounts["account_id"].astype(str)
        accounts = accounts[accounts["account_id"].isin(account_to_idx)]
        accounts["created_at"] = pd.to_datetime(accounts["created_at"], utc=True)
        now_ts = accounts["created_at"].max()
        for _, row in accounts.iterrows():
            idx = account_to_idx[row["account_id"]]
            age_days = max(0.0, (now_ts - row["created_at"]).total_seconds() / 86400.0)
            account_age_days[idx] = age_days

    X = np.vstack([
        sent_counts,
        recv_counts,
        sent_amounts,
        recv_amounts,
        device_counts,
        account_age_days,
    ]).T

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    return X_std


def to_pyg_data(
    accounts: pd.DataFrame,
    tx: pd.DataFrame,
    devices: pd.DataFrame,
    labels: Optional[pd.DataFrame],
):
    account_to_idx = build_account_index(accounts)

    X = compute_node_features(accounts, tx, devices, account_to_idx)
    x = torch.tensor(X, dtype=torch.float32)

    t_src, t_dst, _ = build_transaction_edges(tx, account_to_idx, undirected=True)
    d_src, d_dst = build_device_edges(devices, account_to_idx, undirected=True)

    src = torch.tensor(t_src + d_src, dtype=torch.long)
    dst = torch.tensor(t_dst + d_dst, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    data = Data(x=x, edge_index=edge_index)

    y = None
    if labels is not None and "is_fraud" in labels.columns:
        labels = labels.copy()
        labels["account_id"] = labels["account_id"].astype(str)
        labels = labels[labels["account_id"].isin(account_to_idx)]
        y = torch.full((len(account_to_idx),), -1, dtype=torch.long)
        for _, row in labels.iterrows():
            y[account_to_idx[row["account_id"]]] = int(row["is_fraud"])
    return data, y, account_to_idx


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def corruption(x, edge_index):
    perm = torch.randperm(x.size(0))
    return x[perm], edge_index


def train_dgi(data: Data, hidden_dim=64, out_dim=64, epochs=120, lr=1e-3, weight_decay=1e-4, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    encoder = GCNEncoder(in_dim=data.num_features, hidden_dim=hidden_dim, out_dim=out_dim).to(device)
    model = DeepGraphInfomax(
        hidden_channels=out_dim,
        encoder=encoder,
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(data.x, data.edge_index)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | DGI Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        z = model.encoder(data.x, data.edge_index)
    return z.cpu()


def majority_vote_map(clusters: np.ndarray, labels: np.ndarray) -> dict:
    mapping = {}
    for c in np.unique(clusters):
        mask = clusters == c
        votes = labels[mask]
        votes = votes[votes >= 0]
        if len(votes) == 0:
            mapping[c] = 0
        else:
            ones = (votes == 1).sum()
            zeros = (votes == 0).sum()
            mapping[c] = 1 if ones >= zeros else 0
    return mapping


def run_pipeline(base_dir: str) -> None:
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "outputs")
    ensure_dir(out_dir)

    print("[1/5] Synthesizing CSVs...")
    accounts, transactions, devices, labels = synthesize_data(data_dir)

    print("[2/5] Building graph and features...")
    data, y, account_to_idx = to_pyg_data(accounts, transactions, devices, labels)
    torch.save({"data": data, "y": y, "account_to_idx": account_to_idx}, os.path.join(out_dir, "graph.pt"))
    print(data)

    print("[3/5] Training DGI to get embeddings...")
    z = train_dgi(data)
    torch.save({"embeddings": z, "num_nodes": data.num_nodes}, os.path.join(out_dir, "node_embeddings.pt"))

    print("[4/5] KMeans clustering and evaluation...")
    Z = z.numpy()
    kmeans = KMeans(n_clusters=2, n_init="auto", random_state=42)
    clusters = kmeans.fit_predict(Z)

    sil = silhouette_score(Z, clusters)
    dbi = davies_bouldin_score(Z, clusters)

    results = {
        "silhouette_score": sil,
        "davies_bouldin_score": dbi,
    }

    if y is not None and (y >= 0).any().item():
        y_np = y.numpy()
        known_mask = y_np >= 0
        cl_known = clusters[known_mask]
        y_known = y_np[known_mask]

        ari = adjusted_rand_score(y_known, cl_known)
        nmi = normalized_mutual_info_score(y_known, cl_known)

        c2l = majority_vote_map(clusters, y_np)
        pred_labels = np.array([c2l[c] for c in cl_known])
        acc = accuracy_score(y_known, pred_labels)
        f1 = f1_score(y_known, pred_labels)

        centers = kmeans.cluster_centers_
        fraud_cluster = 1 if c2l.get(1, 0) == 1 else 0
        fraud_center = centers[fraud_cluster]
        d_fraud = np.linalg.norm(Z[known_mask] - fraud_center, axis=1)
        scores = -d_fraud
        try:
            auc = roc_auc_score(y_known, scores)
        except Exception:
            auc = float("nan")

        results.update({
            "adjusted_rand_index": ari,
            "normalized_mutual_info": nmi,
            "accuracy_majority_vote": acc,
            "f1_majority_vote": f1,
            "roc_auc_distance": auc,
        })

    pd.DataFrame([results]).to_csv(os.path.join(out_dir, "cluster_metrics.csv"), index=False)
    print("Evaluation metrics:")
    for k, v in results.items():
        try:
            print(f"  {k}: {v:.4f}")
        except Exception:
            print(f"  {k}: {v}")

    print("[5/5] UMAP visualization...")
    try:
        import umap
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="whitegrid", context="talk")

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        coords = reducer.fit_transform(Z)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=clusters.astype(int), s=10, linewidth=0, alpha=0.85)
        plt.title("UMAP - KMeans Clusters")
        plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
        plt.legend(title="Cluster", loc="best", markerscale=2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "umap_clusters.png"), dpi=160)
        plt.close()

        if y is not None and (y >= 0).any().item():
            y_np = y.numpy()
            known_mask = y_np >= 0
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=coords[known_mask, 0], y=coords[known_mask, 1], hue=y_np[known_mask].astype(int), s=10, linewidth=0, alpha=0.85)
            plt.title("UMAP - Ground Truth Labels")
            plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
            plt.legend(title="is_fraud", loc="best", markerscale=2)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "umap_labels.png"), dpi=160)
            plt.close()
        print("Saved UMAP figures in outputs/ directory.")
    except Exception as e:
        print("UMAP/plotting skipped:", repr(e))


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    run_pipeline(base_dir)


