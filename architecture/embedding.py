import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim=128,
        dssr_dim=18,
    ):
        super().__init__()

        self.protein_embed = nn.Embedding(20, 32)
        self.dna_embed = nn.Embedding(4, 32)

        self.node_type_embed = nn.Embedding(2, 8)

        self.interface_proj = nn.Embedding(2, 8)

        self.dssr_proj = nn.Sequential(
            nn.Linear(dssr_dim, 32), nn.GELU(), nn.Linear(32, 32)
        )

        total_dim = 32 + 8 + 8 + 32  # residue/base  # node type  # interface  # dssr

        self.project = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
        )

    def forward(
        self,
        node_type,
        residue_id,
        interface,
        dssr,
    ):

        device = node_type.device

        N = node_type.shape[0]

        identity = torch.zeros(
            *node_type.shape,
            32,
            device=device,
        )

        protein_mask = node_type == 1

        dna_mask = node_type == 0

        if protein_mask.any():

            identity[protein_mask] = self.protein_embed(residue_id[protein_mask])

        if dna_mask.any():

            identity[dna_mask] = self.dna_embed(residue_id[dna_mask])

        node_type = self.node_type_embed(node_type)

        interface = self.interface_proj(interface.long())

        dssr = self.dssr_proj(dssr)

        x = torch.cat(
            [
                identity,
                node_type,
                interface,
                dssr,
            ],
            dim=-1,
        )

        return self.project(x)
