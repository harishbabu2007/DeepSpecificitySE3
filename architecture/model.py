import torch
import torch.nn as nn

from architecture.embedding import FeatureEncoder
from se3_transformer_pytorch import SE3Transformer


class DeepSpecificitySE3(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_dim = 48

        self.encoder = FeatureEncoder(
            hidden_dim=self.hidden_dim,
            dssr_dim=18,
        )

        self.se3 = SE3Transformer(
            dim=self.hidden_dim,
            depth=2,
            heads=1,
            dim_head=16,
            input_degrees=1,
            num_degrees=2,
            output_degrees=1,
            edge_dim=32,
            num_neighbors=32,
            valid_radius=12.0,
            fourier_encode_dist=True,
            attend_self=True,
            use_null_kv=True,
            norm_out=True,
        )

        self.dropout = nn.Dropout(0.2)

        self.out = nn.Linear(
            self.hidden_dim,
            4,
        )

    def forward(
        self,
        node_type,
        residue_id,
        interface,
        dssr,
        coords,
        mask,
        edges,
        num_dna_nodes,
        adj_mat=None,
    ):
        x = self.encoder(
            node_type=node_type,
            residue_id=residue_id,
            interface=interface,
            dssr=dssr,
        )

        x = self.se3(
            feats=x,
            coors=coords,
            mask=mask,
            edges=edges,
            # adj_mat=adj_mat,
        )

        x = self.dropout(x)

        num_dna = int(num_dna_nodes[0])
        dna = x[:, :num_dna, :]
        logits = self.out(dna)

        return logits
