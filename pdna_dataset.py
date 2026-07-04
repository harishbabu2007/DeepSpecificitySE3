import os
import torch
from torch.utils.data import Dataset


def collate_fn(batch):

    batch_size = len(batch)

    max_nodes = max(g.pos.size(0) for g in batch)

    edge_dim = batch[0].edges.size(-1)
    dssr_dim = batch[0].dssr_feat.size(-1)

    max_bp = max(g.num_base_pairs for g in batch)

    coords = torch.zeros(batch_size, max_nodes, 3)

    node_type = torch.zeros(batch_size, max_nodes, dtype=torch.long)

    res_type = torch.zeros(batch_size, max_nodes, dtype=torch.long)

    interface = torch.zeros(batch_size, max_nodes)

    dssr = torch.zeros(batch_size, max_nodes, dssr_dim)

    mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)

    adj_mat = torch.zeros(
        batch_size,
        max_nodes,
        max_nodes,
        dtype=torch.bool,
    )

    edges = torch.zeros(
        batch_size,
        max_nodes,
        max_nodes,
        edge_dim,
        dtype=torch.float32,
    )

    pwm_forward = torch.full(
        (batch_size, max_bp, 4),
        0.25,
        dtype=torch.float32,
    )

    pwm_reverse = torch.full(
        (batch_size, max_bp, 4),
        0.25,
        dtype=torch.float32,
    )

    pwm_mask_forward = torch.zeros(
        batch_size,
        max_bp,
        dtype=torch.bool,
    )

    pwm_mask_reverse = torch.zeros(
        batch_size,
        max_bp,
        dtype=torch.bool,
    )

    proximity_forward = torch.zeros(
        batch_size,
        max_bp,
        dtype=torch.bool,
    )

    proximity_reverse = torch.zeros(
        batch_size,
        max_bp,
        dtype=torch.bool,
    )

    num_dna_nodes = []
    num_protein_nodes = []
    num_base_pairs = []
    pwm_present = []

    for i, graph in enumerate(batch):

        n = graph.pos.size(0)
        bp = graph.num_base_pairs

        coords[i, :n] = graph.pos

        node_type[i, :n] = graph.node_type

        res_type[i, :n] = graph.res_type

        interface[i, :n] = graph.interface

        dssr[i, :n] = graph.dssr_feat

        mask[i, :n] = True

        # adj_mat[i, :n, :n] = graph.adj_mat

        edges[i, :n, :n] = graph.edges

        pwm_forward[i, :bp] = graph.pwm_forward
        pwm_reverse[i, :bp] = graph.pwm_reverse

        pwm_mask_forward[i, :bp] = graph.pwm_mask_forward
        pwm_mask_reverse[i, :bp] = graph.pwm_mask_reverse

        proximity_forward[i, :bp] = graph.proximity_forward
        proximity_reverse[i, :bp] = graph.proximity_reverse

        num_dna_nodes.append(graph.num_dna_nodes)
        num_protein_nodes.append(graph.num_protein_nodes)
        num_base_pairs.append(graph.num_base_pairs)
        pwm_present.append(graph.pwm_present)

    return {
        "coords": coords,
        "node_type": node_type,
        "res_type": res_type,
        "interface": interface,
        "dssr": dssr,
        "mask": mask,
        # "adj_mat": adj_mat,
        "edges": edges,
        "pwm_forward": pwm_forward,
        "pwm_reverse": pwm_reverse,
        "pwm_mask_forward": pwm_mask_forward,
        "pwm_mask_reverse": pwm_mask_reverse,
        "proximity_forward": proximity_forward,
        "proximity_reverse": proximity_reverse,
        "num_dna_nodes": torch.tensor(
            num_dna_nodes,
            dtype=torch.long,
        ),
        "num_protein_nodes": torch.tensor(
            num_protein_nodes,
            dtype=torch.long,
        ),
        "num_base_pairs": torch.tensor(
            num_base_pairs,
            dtype=torch.long,
        ),
        "pwm_present": torch.tensor(
            pwm_present,
            dtype=torch.bool,
        ),
    }


class DeepSpecificityDataset(Dataset):
    def __init__(self, root):
        self.root = root

        self.files = sorted(
            [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".pt")]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        graph = torch.load(
            self.files[idx],
            weights_only=False,
            map_location="cpu",
        )

        return graph
