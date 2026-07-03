import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from sklearn.metrics import roc_auc_score
from collections import defaultdict


TRAIN_DIR = "graphs_train"
VAL_DIR = "graphs_valid"
PWM_DIR = "/home/compbio/aastha/gnn_protein/preprocessing/RCSB/all_pfms_jaspar"

BATCH_SIZE = 8
EPOCHS = 100
LR = 3e-4
HIDDEN = 128
LAYERS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_NODE_TYPES = 2
NUM_DNA = 4
NUM_PROTEIN = 20

EDGE_DIM = 28

# Node feature dim:
# node_type (2) + protein one-hot (20)
NODE_DIM = 2 + 20 + 18


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_pwms(pdbid):
    pwms =[]
    for f in os.listdir(PWM_DIR):
        if f.startswith(pdbid + "_"):
            pwm=np.loadtxt(os.path.join(PWM_DIR, f))
            pwms.append((f,pwm))
    return pwms

def rc_pwm(pwm):
    return pwm[::-1][:, [3,2,1,0]]

def information_content(pwm):
    p=np.clip(pwm, 1e-6, 1.0)
    H= -(p*np.log2(p)).sum(axis=1)
    return 2.0-H

def score_pwm(seq, pwm):
    ic= information_content(pwm)
    score=0.0
    for i, base in enumerate(seq):
        score+=pwm[i,base]*ic[i]
    return score

def best_match(seq, pwm):

    best_score = -1e9
    best_start = None
    best_orientation = None

    candidates = [("forward", pwm), ("reverse", rc_pwm(pwm))]

    for orient, motif in candidates:
        L = len(motif)
        if L > len(seq):
            continue
        for start in range(len(seq)-L+1):
            window = seq[start:start+L]
            score = score_pwm(
                window,
                motif
            )
            if score > best_score:

                best_score = score
                best_start = start
                best_orientation = orient

    return best_start, best_orientation, best_score


def build_node_features(data):

    N = data.node_type.shape[0]

    node_type_oh = F.one_hot(
        data.node_type,
        num_classes=2
    ).float()

    protein_oh = torch.zeros(
        (N, NUM_PROTEIN),
        dtype=torch.float
    )

    protein_mask = (data.node_type == 1)

    if protein_mask.any():
        protein_oh[protein_mask] = F.one_hot(
            data.res_type[protein_mask],
            num_classes=NUM_PROTEIN
        ).float()

    x = torch.cat([
        node_type_oh,
        protein_oh,
        data.dssr_feat.float()
    ], dim=-1)

    return x

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.files = sorted([f for f in os.listdir(folder) if f.endswith(".pt")])
        self.folder = folder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        fname = self.files[idx]
        data = torch.load(
            os.path.join(self.folder, self.files[idx]),
            weights_only=False
        )

        pdbid = fname.replace(".pt", "")
        pwms = load_pwms(pdbid)

        data.x = build_node_features(data)

        # labels
        N = data.node_type.shape[0]
        targets=torch.full((N,4),0.25,dtype=torch.float)

        dna_mask=(data.node_type ==0)
        interface_mask=(data.interface>0.5)

        dna_interface =dna_mask& interface_mask
        
        if dna_interface.any():
            targets[dna_interface]=F.one_hot(data.res_type[dna_interface], num_classes=4).float()

        chain_nodes =defaultdict(list)
        for idx2 in torch.where(dna_interface)[0].tolist():
            chain_nodes[data.chain[idx2]].append(idx2)

        for chain in chain_nodes:
            chain_nodes[chain]=sorted(chain_nodes[chain],key=lambda x:data.resnum[x].item())

        for pwm_name, pwm in pwms:
            best_chain = None
            best_start = None
            best_orientation = None
            best_score = -1e9
            for chain, node_list in chain_nodes.items():
                seq = [
                    data.res_type[i].item()
                    for i in node_list
                ]
                start, orient, score = best_match(
                    seq,
                    pwm
                )
                if start is None:
                    continue
                if score > best_score:
                    best_score = score
                    best_chain = chain
                    best_start = start
                    best_orientation = orient

            if best_chain is None:
                continue
            if best_orientation == "forward":
                motif = pwm
            else:
                motif = rc_pwm(pwm)
            node_list = chain_nodes[best_chain]
            L = len(motif)
            matched_nodes = node_list[
                best_start:
                best_start + L
            ]
            targets[matched_nodes] = torch.tensor(
                motif,
                dtype=torch.float
            )
        
        data.y=targets

        return data

class NAMPNNLayer(MessagePassing):
    def __init__(self, hidden):
        super().__init__(aggr="add")

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden + EDGE_DIM, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Sigmoid()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        msg = torch.cat([x_j, edge_attr], dim=-1)
        msg = self.edge_mlp(msg)
        msg = msg * self.gate(msg)
        return self.dropout(msg)

    def update(self, aggr_out, x):
        upd = self.node_mlp(torch.cat([x, aggr_out], dim=-1))
        gate = torch.sigmoid(upd)
        return x + gate * upd

class ProteinDNAMPNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_proj = nn.Linear(NODE_DIM, HIDDEN)

        self.layers = nn.ModuleList([
            NAMPNNLayer(HIDDEN) for _ in range(LAYERS)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(HIDDEN) for _ in range(LAYERS)
        ])

        self.dropout = nn.Dropout(0.1)

        self.out = nn.Linear(HIDDEN, NUM_DNA)

    def forward(self, data):

        x = self.input_proj(data.x)
        x = self.dropout(x)

        for layer, norm in zip(self.layers, self.norms):
            h = layer(x, data.edge_index, data.edge_attr)
            x = norm(x + h)
            x = self.dropout(x)

        logits = self.out(x)
        return logits

def compute_auroc(y_true, y_prob):
    aurocs = []

    for c in range(NUM_DNA):
        y_bin = (y_true == c).astype(int)

        if len(np.unique(y_bin)) < 2:
            continue

        score = roc_auc_score(y_bin, y_prob[:, c])
        aurocs.append(score)

    if len(aurocs) == 0:
        return np.nan

    return np.mean(aurocs)

def train():

    set_seed()

    train_loader = DataLoader(
        GraphDataset(TRAIN_DIR),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        GraphDataset(VAL_DIR),
        batch_size=BATCH_SIZE
    )

    model = ProteinDNAMPNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")

    for epoch in range(EPOCHS):

        # ================= TRAIN =================
        model.train()
        train_loss = 0
        train_probs = []
        train_labels = []

        for batch in train_loader:
            batch = batch.to(DEVICE)

            optimizer.zero_grad()

            logits = model(batch)

            dna_mask = (batch.node_type == 0)

            log_probs = F.log_softmax(logits[dna_mask], dim=-1)
            loss= -(batch.y[dna_mask]*log_probs).sum(dim=-1).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            probs = F.softmax(logits, dim=-1)

            if dna_mask.any():
                train_probs.append(probs[dna_mask].detach().cpu())
                train_labels.append(batch.res_type[dna_mask].detach().cpu())

        train_loss /= len(train_loader)

        # ================= VALID =================
        model.eval()
        val_loss = 0
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)

                logits = model(batch)

                dna_mask = (batch.node_type == 0)

                log_probs = F.log_softmax(logits[dna_mask], dim=-1)
                loss= -(batch.y[dna_mask]*log_probs).sum(dim=-1).mean()

                val_loss += loss.item()

                probs = F.softmax(logits, dim=-1)

                if dna_mask.any():
                    val_probs.append(probs[dna_mask].cpu())
                    val_labels.append(batch.res_type[dna_mask].cpu())

        val_loss /= len(val_loader)

        # ================= AUROC =================
        if train_labels:
            train_y = torch.cat(train_labels).numpy()
            train_p = torch.cat(train_probs).numpy()
            train_auc = compute_auroc(train_y, train_p)
        else:
            train_auc = np.nan

        if val_labels:
            val_y = torch.cat(val_labels).numpy()
            val_p = torch.cat(val_probs).numpy()
            val_auc = compute_auroc(val_y, val_p)
        else:
            val_auc = np.nan

        print(
            f"Epoch {epoch:03d} | "
            f"TrainLoss {train_loss:.4f} | "
            f"ValLoss {val_loss:.4f} | "
            f"TrainAUROC {train_auc:.3f} | "
            f"ValAUROC {val_auc:.3f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "na_mpnn2.pt")


if __name__ == "__main__":
    train()