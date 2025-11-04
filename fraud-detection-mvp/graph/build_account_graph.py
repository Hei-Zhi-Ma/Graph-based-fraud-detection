import os
import yaml
import torch
from torch_geometric.data import Data
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple

from src.io.load_accounts import load_accounts, filter_accounts_by_time
from src.io.load_transactions import load_transactions
from src.io.load_devices import load_devices
 from src.features.basic import build_node_index, compute_transaction_edges, compute_device_co_usage_edges, basic_node_features


 def build_graph_from_config(cfg_path: str) -> Tuple[Data, pd.DataFrame]:
 	with open(cfg_path, "r", encoding="utf-8") as f:
 		cfg = yaml.safe_load(f)

 	acc_p = cfg["data"]["accounts_path"]
 	tx_p = cfg["data"]["transactions_path"]
 	dev_p = cfg["data"]["devices_path"]
 	win_days = cfg["data"].get("time_window_days", 90)
 	make_undirected = bool(cfg["graph"].get("make_undirected", True))
 	min_acc = int(cfg["graph"].get("device_co_usage_min_accounts", 2))
 	max_deg = int(cfg["graph"].get("max_device_degree_per_account", 50))

 	# load
 	acc = load_accounts(acc_p)
 	tx = load_transactions(tx_p)
 	dev = load_devices(dev_p)

 	# time window cutoff (use max ts in tx)
 	if not tx.empty:
 		max_ts = pd.to_datetime(tx["ts"].max())
 		cutoff = max_ts - timedelta(days=win_days)
 		acc = filter_accounts_by_time(acc, cutoff)
 		tx = tx[tx["ts"] >= cutoff].reset_index(drop=True)

 	# node index
 	node_index = build_node_index(acc)
 	num_nodes = len(node_index)

 	# edges
 	tr_ei, tr_w = compute_transaction_edges(tx, node_index, make_undirected)
 	dev_ei, dev_w = compute_device_co_usage_edges(dev, node_index, min_acc, make_undirected, max_deg)
 	if tr_ei.numel() == 0 and dev_ei.numel() == 0:
 		edge_index = torch.empty((2, 0), dtype=torch.long)
 		edge_weight = torch.empty((0,), dtype=torch.float32)
 	else:
 		edge_index = torch.cat([tr_ei, dev_ei], dim=1)
 		edge_weight = torch.cat([tr_w, dev_w], dim=0)

 	# basic features
 	x = basic_node_features(num_nodes, edge_index, edge_weight)

 	data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
 	# map account_id order back for downstream reference
 	inv_index = [None] * num_nodes
 	for aid, idx in node_index.items():
 		inv_index[idx] = aid
 	data.account_id = inv_index

 	return data, acc


 def save_graph(data: Data, out_path: str) -> None:
 	os.makedirs(os.path.dirname(out_path), exist_ok=True)
 	torch.save(data, out_path)

