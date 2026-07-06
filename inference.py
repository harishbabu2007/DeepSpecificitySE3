import argparse
import torch
import os

import sys
from pathlib import Path
from Bio.PDB import PDBParser
import numpy as np
from scipy.spatial import cKDTree

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

preprocessing_dir = Path(__file__).resolve().parent / "preprocessing"
sys.path.append(str(preprocessing_dir))

from preprocessing.build_graph import extract_dssr_features, build_dna_frame, build_protein_frame, normalize, rbf_expand
from preprocessing.build_graph import PROTEIN_RES, DNA_BASE, INTERFACE_RADIUS, NODE_TYPE, K_NEIGHBORS
from architecture.model import DeepSpecificitySE3
from plotting import plot_ppms

def preprocess(pdb_path, device):
    pdb_name = os.path.basename(pdb_path)
    pdb_id = os.path.splitext(pdb_name)[0]

    print(f"Processing, {pdb_name}")

    parser = PDBParser(QUIET=True)

    try:
        dssr_feature_map = extract_dssr_features(pdb_path, relative="preprocessing")
        structure = parser.get_structure("X", pdb_path)

        protein_atoms =[]
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.get_resname().strip() in PROTEIN_RES:
                        for atom in res:
                            protein_atoms.append(atom.coord)

        protein_atoms= np.array(protein_atoms)
        if len(protein_atoms)>0:
            protein_tree = cKDTree(protein_atoms)
        else:
            protein_tree = None

        nodes = []

        dna_nodes = []
        protein_nodes = []

        for model in structure:
            for chain in model:

                residues = list(chain)

                for i, res in enumerate(residues):

                    resname = res.get_resname().strip()

                    prev_res = residues[i - 1] if i > 0 else None
                    next_res = residues[i + 1] if i < len(residues) - 1 else None

                    # DNA
                    if resname in DNA_BASE:

                        out = build_dna_frame(res, prev_res, next_res)
                        if out is None:
                            continue

                        ref, R = out

                        interface = 0.0
                        if protein_tree is not None:
                            dna_atoms = np.array([a.coord for a in res])
                            dists, _ = protein_tree.query(dna_atoms, k=1)
                            if np.min(dists) <= INTERFACE_RADIUS:
                                interface = 1.0

                        resnum = res.id[1]
                        key = (chain.id, resnum)

                        dssr_feat = dssr_feature_map.get(
                            key, np.zeros(18, dtype=np.float32)
                        )

                        dna_nodes.append(
                            {
                                "ref": ref,
                                "R": R,
                                "type": NODE_TYPE["DNA"],
                                "res_type": DNA_BASE[resname],
                                "interface": interface,
                                "dssr_feat": dssr_feat,
                                "chain": chain.id,
                                "resnum": res.id[1],
                            }
                        )

                    # PROTEIN
                    elif resname in PROTEIN_RES:

                        out = build_protein_frame(res)
                        if out is None:
                            continue

                        ref, R = out

                        protein_nodes.append(
                            {
                                "ref": ref,
                                "R": R,
                                "type": NODE_TYPE["PROTEIN"],
                                "res_type": PROTEIN_RES[resname],
                                "interface": 0,
                                "dssr_feat": np.zeros(18, dtype=np.float32),
                                "chain": chain.id,
                                "resnum": res.id[1],
                            }
                        )

        nodes = dna_nodes + protein_nodes
        if len(nodes) == 0:
            raise Exception("no nodes processed")

        pos = np.array([n["ref"] for n in nodes])
        R = np.array([n["R"] for n in nodes])

        pos_t = torch.tensor(pos, dtype=torch.float)
        R_t = torch.tensor(R, dtype=torch.float)

        node_type = torch.tensor([n["type"] for n in nodes])
        res_type = torch.tensor([n["res_type"] for n in nodes])
        interface = torch.tensor([n["interface"] for n in nodes], dtype=torch.float)

        dssr_feat = torch.tensor(np.array([n["dssr_feat"] for n in nodes]), dtype=torch.float)

        tree = cKDTree(pos)
        k = min(K_NEIGHBORS+1, len(nodes))
        dists_knn, idx_knn = tree.query(pos, k=k)

        edge_src, edge_dst = [], []
        edge_attr = []

        for i in range(len(nodes)):

            nbrs = idx_knn[i][1:]

            for j in nbrs:

                diff = pos[j] - pos[i]
                d = np.linalg.norm(diff)

                local_vec = R[i] @ diff
                local_vec = normalize(local_vec)

                rbf = rbf_expand(d)

                R_rel = R[i] @ R[j].T
                rot_feat = R_rel.reshape(-1)

                edge_type = np.zeros(4, dtype=np.float32)

                src_is_dna = nodes[i]["type"] == NODE_TYPE["DNA"]
                dst_is_dna = nodes[j]["type"] == NODE_TYPE["DNA"]

                if src_is_dna and dst_is_dna:
                    edge_type[0] = 1.0  # DNA → DNA
                elif src_is_dna and not dst_is_dna:
                    edge_type[1] = 1.0  # DNA → Protein
                elif (not src_is_dna) and dst_is_dna:
                    edge_type[2] = 1.0  # Protein → DNA
                else:
                    edge_type[3] = 1.0  # Protein → Protein

                feat = np.concatenate([rbf, local_vec, rot_feat, edge_type])

                edge_src.append(i)
                edge_dst.append(j)
                edge_attr.append(feat)

                # reverse edge
                local_vec_rev = R[j] @ (-diff)
                local_vec_rev = normalize(local_vec_rev)

                R_rel_rev = R[j] @ R[i].T
                rot_feat_rev = R_rel_rev.reshape(-1)

                edge_type_rev = np.zeros(4, dtype=np.float32)

                src_is_dna = nodes[j]["type"] == NODE_TYPE["DNA"]
                dst_is_dna = nodes[i]["type"] == NODE_TYPE["DNA"]

                if src_is_dna and dst_is_dna:
                    edge_type_rev[0] = 1.0
                elif src_is_dna and not dst_is_dna:
                    edge_type_rev[1] = 1.0
                elif (not src_is_dna) and dst_is_dna:
                    edge_type_rev[2] = 1.0
                else:
                    edge_type_rev[3] = 1.0

                feat_rev = np.concatenate(
                    [rbf, local_vec_rev, rot_feat_rev, edge_type_rev]
                )

                edge_src.append(j)
                edge_dst.append(i)
                edge_attr.append(feat_rev)

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        num_nodes = len(nodes)

        edges = torch.zeros(
                (num_nodes, num_nodes, edge_attr.size(1)),
                dtype=torch.float32,
            )

        for k in range(edge_index.size(1)):

            src = edge_index[0, k].item()
            dst = edge_index[1, k].item()

            # adj_mat[src, dst] = True
            edges[src, dst] = edge_attr[k]

        chains = [n["chain"] for n in nodes]
        resnums = [n["resnum"] for n in nodes]

        return {
            "coords": pos_t,
            "frame": R_t,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edges": edges,
            "node_type": node_type,
            "res_type": res_type,
            "interface": interface,
            "dssr_feat": dssr_feat,
            "chain": chains,
            "num_dna_nodes": len(dna_nodes),
            "num_protein_nodes": len(protein_nodes),
            "pdb_id": pdb_id
        }

    except Exception as e:
        print("Error parsing PDB")
        print(e)
        exit(1)


def inference_model(data, device, checkpoint_path):

    model = DeepSpecificitySE3().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    coords = data["coords"].unsqueeze(0).to(device)
    node_type = data["node_type"].unsqueeze(0).to(device)
    res_type = data["res_type"].unsqueeze(0).to(device)
    interface = data["interface"].unsqueeze(0).to(device)
    dssr = data["dssr_feat"].unsqueeze(0).to(device)

    edges = data["edges"].unsqueeze(0).to(device)

    num_dna_nodes = num_dna_nodes = torch.tensor(
        [data["num_dna_nodes"]],
        device=device,
        dtype=torch.long,
    )

    mask = torch.ones(
        (1, coords.shape[1]),
        dtype=torch.bool,
        device=device,
    )

    with torch.no_grad():

        logits = model(
            coords=coords,
            node_type=node_type,
            residue_id=res_type,
            interface=interface,
            dssr=dssr,
            mask=mask,
            edges=edges,
            num_dna_nodes=num_dna_nodes,
        )

    num_bp = num_dna_nodes.item() // 2

    logits_forward = logits[:, :num_bp]

    logits_reverse = logits[:, num_bp:]

    logits_reverse = torch.flip(
        logits_reverse,
        dims=[1],
    )

    logits_reverse = logits_reverse[..., [3, 2, 1, 0]]

    ppm_forward = torch.softmax(
        logits_forward,
        dim=-1,
    )

    ppm_reverse = torch.softmax(
        logits_reverse,
        dim=-1,
    )

    return (
        ppm_forward.squeeze(0).cpu().numpy(),
        ppm_reverse.squeeze(0).cpu().numpy(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Inference pipeline for DeepSpecificity"
    )

    parser.add_argument("--pdb", type=str, help="path to pdb file")
    parser.add_argument("--checkpoint", type=str, help="path to the checkpoint file")
    parser.add_argument("--amap", action="store_true")
    parser.add_argument("--nohb", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = preprocess(args.pdb, device)

    ppm_fwd, ppm_rev = inference_model(data, device, args.checkpoint)

    plot_ppms(ppm_fwd, ppm_rev, data['pdb_id'])

if __name__ == "__main__":
    main()
