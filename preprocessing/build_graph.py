import os
import torch
import numpy as np
import json
import subprocess
from Bio.PDB import PDBParser
from torch_geometric.data import Data
from scipy.spatial import cKDTree
from pwm_utils import generate_training_pwm
from preprocessing_utils import load_motif_annotations
from get_pwm import build_jaspar_index
from pwm_utils import find_base_pairs

PDB_DIR = "../data/pdbs"
OUT_DIR = "../data/processed"

RADIUS = 12.0
K_NEIGHBORS = 32
RBF_DIM = 16
RBF_DMAX = 20.0
INTERFACE_RADIUS = 10.0

os.makedirs(OUT_DIR, exist_ok=True)

NODE_TYPE = {"DNA": 0, "PROTEIN": 1}

DNA_BASE = {"DA": 0, "DC": 1, "DG": 2, "DT": 3,
            "A": 0, "C": 1, "G": 2, "T": 3}

PROTEIN_RES = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3,
    "CYS": 4, "GLN": 5, "GLU": 6, "GLY": 7,
    "HIS": 8, "ILE": 9, "LEU": 10, "LYS": 11,
    "MET": 12, "PHE": 13, "PRO": 14, "SER": 15,
    "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19
}

def rbf_expand(d, D_min=0.0, D_max=RBF_DMAX, D_count=RBF_DIM):
    centers = np.linspace(D_min, D_max, D_count)
    widths = (D_max - D_min) / D_count
    return np.exp(-((d - centers) ** 2) / (widths ** 2))


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)

def parse_nt_id(nt_string):
    """
    C.DC14 -> ('C',14)
    D.DG17 -> ('D',17)
    """
    chain, rest = nt_string.split(".")
    num = int("".join(c for c in rest if c.isdigit()))
    return chain, num


def extract_dssr_features(pdb_path):

    cmd = [
        "./x3dna-dssr",
        "-i=" + pdb_path,
        "--more",
        "--json",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    data = json.loads(result.stdout)

    bp_feature = {}
    dssr_feature = {}

    helices = data.get("helices", [])

    for helix in helices:

        pairs = helix.get("pairs", [])

        n = len(pairs)

        # ---------- bp params ----------

        for i, pair in enumerate(pairs):

            nt1 = parse_nt_id(pair["nt1"])
            nt2 = parse_nt_id(pair["nt2"])

            if "bp1_params" in pair:
                bp = np.array(
                    pair["bp1_params"],
                    dtype=np.float32
                )

            elif i > 0 and "bp2_params" in pairs[i - 1]:
                bp = np.array(
                    pairs[i - 1]["bp2_params"],
                    dtype=np.float32
                )

            else:
                bp = np.zeros(6, dtype=np.float32)

            bp_feature[nt1] = bp
            bp_feature[nt2] = bp

        # ---------- build 18D DSSR features ----------

        pair_steps = []

        for pair in pairs:

            if "step_params" in pair:
                pair_steps.append(
                    np.array(
                        pair["step_params"],
                        dtype=np.float32
                    )
                )
            else:
                pair_steps.append(
                    np.zeros(6, dtype=np.float32)
                )

        for i, pair in enumerate(pairs):

            nt1 = parse_nt_id(pair["nt1"])
            nt2 = parse_nt_id(pair["nt2"])

            bp = bp_feature[nt1]

            prev_step = (
                pair_steps[i - 1]
                if i > 0
                else np.zeros(6, dtype=np.float32)
            )

            next_step = (
                pair_steps[i]
                if i < n - 1
                else np.zeros(6, dtype=np.float32)
            )

            feat18 = np.concatenate(
                [bp, prev_step, next_step]
            )

            dssr_feature[nt1] = feat18
            dssr_feature[nt2] = feat18

    return dssr_feature


def build_protein_frame(res):
    atoms = {a.get_name(): a.coord for a in res}

    if not all(k in atoms for k in ["N", "CA", "C"]):
        return None

    CA = atoms["CA"]
    N = atoms["N"]
    C = atoms["C"]

    x = normalize(C - CA)
    y = normalize(N - CA)
    z = normalize(np.cross(x, y))
    y = normalize(np.cross(z, x))

    R = np.stack([x, y, z], axis=0)
    return CA, R


def build_dna_frame(res, prev_res=None, next_res=None):
    atoms = {a.get_name(): a.coord for a in res}

    if "C1'" not in atoms:
        return None

    ref = atoms["C1'"]

    # base direction
    base_atom = None
    for name, coord in atoms.items():
        if "'" not in name and name not in ["P", "OP1", "OP2"]:
            base_atom = coord
            break

    if base_atom is None:
        return None

    x = normalize(base_atom - ref)

    # backbone direction (5'→3')
    if next_res and "C1'" in {a.get_name() for a in next_res}:
        next_atoms = {a.get_name(): a.coord for a in next_res}
        y = normalize(next_atoms["C1'"] - ref)
    elif prev_res and "C1'" in {a.get_name() for a in prev_res}:
        prev_atoms = {a.get_name(): a.coord for a in prev_res}
        y = normalize(ref - prev_atoms["C1'"])
    else:
        return None

    z = normalize(np.cross(x, y))
    y = normalize(np.cross(z, x))

    R = np.stack([x, y, z], axis=0)
    return ref, R


def generate_spatial_proximity_mask(
    structure,
    protein_tree,
    distance_threshold=5.0,
):
    """
    Returns one boolean per DNA base-pair.

    Length == number of PWM columns.
    """

    model = structure[0]

    dna_pairs = find_base_pairs(model)

    mask = np.zeros(len(dna_pairs), dtype=bool)

    if protein_tree is None:
        return mask

    for i, (forward_res, reverse_res) in enumerate(dna_pairs):

        coords = []

        for residue in (forward_res, reverse_res):

            if residue is None:
                continue

            for atom in residue:
                coords.append(atom.coord)

        coords = np.asarray(coords)

        dists, _ = protein_tree.query(coords, k=1)

        if np.min(dists) <= distance_threshold:
            mask[i] = True

    return mask


parser = PDBParser(QUIET=True)

print("Building Offline JASPAR Index for Fallback Search...")
jaspar_indices = build_jaspar_index()

# Initialize the JASPAR index ONCE before the loop begins
print("Loading Motif Annotations.")
annotations = load_motif_annotations(
    [
        "../data/specificity_train.json",
        "../data/specificity_train_2.json",
        "../data/specificity_train_3.json",
        "../data/specificity_evaluation_valid2.json",
        "../data/specificity_evaluation_valid1.json",
    ]
)
print(f"Loaded {len(annotations)} annotated structures.\n")

for pdb_file in sorted(os.listdir(PDB_DIR)):

    if not pdb_file.endswith(".pdb"):
        continue

    out_file = os.path.join(
        OUT_DIR,
        pdb_file.replace(".pdb", ".pt")
    )

    if os.path.exists(out_file):
        print("Skipping", pdb_file)
        continue

    print("Processing", pdb_file)

    try:

        pdb_path = os.path.join(PDB_DIR, pdb_file)
        dssr_feature_map = extract_dssr_features(pdb_path)

        structure = parser.get_structure("X", os.path.join(PDB_DIR, pdb_file))

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
                            dists, _ =protein_tree.query(dna_atoms, k=1)
                            if np.min(dists) <= INTERFACE_RADIUS:
                                interface = 1.0

                        resnum = res.id[1]
                        key = (chain.id, resnum)

                        dssr_feat = dssr_feature_map.get(
                            key,
                            np.zeros(18, dtype=np.float32)
                        )

                        dna_nodes.append({
                            "ref": ref,
                            "R": R,
                            "type": NODE_TYPE["DNA"],
                            "res_type": DNA_BASE[resname],
                            "interface": interface,
                            "dssr_feat": dssr_feat,
                            "chain": chain.id,
                            "resnum": res.id[1]
                        })

                    # PROTEIN
                    elif resname in PROTEIN_RES:

                        out = build_protein_frame(res)
                        if out is None:
                            continue

                        ref, R = out

                        protein_nodes.append({
                            "ref": ref,
                            "R": R,
                            "type": NODE_TYPE["PROTEIN"],
                            "res_type": PROTEIN_RES[resname],
                            "interface": 0,
                            "dssr_feat": np.zeros(18, dtype=np.float32),
                            "chain": chain.id,
                            "resnum": res.id[1]
                        })
        nodes = dna_nodes + protein_nodes

        proximity_mask = generate_spatial_proximity_mask(
            structure,
            protein_tree,
            distance_threshold=4.0,
        )

        if len(nodes) == 0:
            continue

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
        radius_neighbours = tree.query_ball_point(pos, r=RADIUS)

        edge_src, edge_dst = [], []
        edge_attr = []

        for i in range(len(nodes)):

            nbrs = set(idx_knn[i][1:])

            for j in radius_neighbours[i]:
                if j!=i:
                    nbrs.add(j)

            for j in nbrs:

                diff = pos[j] - pos[i]
                d = np.linalg.norm(diff)

                local_vec = R[i] @ diff
                local_vec = normalize(local_vec)

                rbf = rbf_expand(d)

                R_rel = R[i]@R[j].T
                rot_feat = R_rel.reshape(-1)

                edge_type = np.zeros(4, dtype=np.float32)

                src_is_dna = nodes[i]["type"] == NODE_TYPE["DNA"]
                dst_is_dna = nodes[j]["type"] == NODE_TYPE["DNA"]

                if src_is_dna and dst_is_dna:
                    edge_type[0] = 1.0          # DNA → DNA
                elif src_is_dna and not dst_is_dna:
                    edge_type[1] = 1.0          # DNA → Protein
                elif (not src_is_dna) and dst_is_dna:
                    edge_type[2] = 1.0          # Protein → DNA
                else:
                    edge_type[3] = 1.0          # Protein → Protein

                feat = np.concatenate([rbf, local_vec, rot_feat])

                edge_src.append(i)
                edge_dst.append(j)
                edge_attr.append(feat)

                # reverse edge
                local_vec_rev = R[j] @ (-diff)
                local_vec_rev = normalize(local_vec_rev)

                R_rel_rev = R[j]@R[i].T
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

                feat_rev = np.concatenate([rbf, local_vec_rev, rot_feat_rev])

                edge_src.append(j)
                edge_dst.append(i)
                edge_attr.append(feat_rev)

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        num_nodes = len(nodes)

        adj_mat = torch.zeros(
            (num_nodes, num_nodes),
            dtype=torch.bool,
        )

        edges = torch.zeros(
            (num_nodes, num_nodes, edge_attr.size(1)),
            dtype=torch.float32,
        )

        for k in range(edge_index.size(1)):

            src = edge_index[0, k].item()
            dst = edge_index[1, k].item()

            adj_mat[src, dst] = True
            edges[src, dst] = edge_attr[k]

        chains = [n["chain"] for n in nodes]
        resnums = [n["resnum"] for n in nodes]

        (
            pwm_forward,
            pwm_reverse,
            pwm_mask_forward,
            pwm_mask_reverse,
            proximity_forward,
            proximity_reverse,
            pwm_present,
        ) = generate_training_pwm(
            structure,
            pdb_file[:4].lower(),
            proximity_mask,
            annotations,
            jaspar_indices,
        )

        data = Data(
            pos=pos_t,
            frame=R_t,
            edge_index=edge_index,
            edge_attr=edge_attr,
            adj_mat=adj_mat,
            edges=edges,
            node_type=node_type,
            res_type=res_type,
            interface=interface,
            dssr_feat=dssr_feat,
        )

        data.chain = chains
        data.resnum = torch.tensor(resnums, dtype=torch.long)
        data.num_dna_nodes = len(dna_nodes)
        data.num_base_pairs = len(proximity_mask)
        data.num_protein_nodes = len(protein_nodes)

        data.pwm_forward = torch.tensor(pwm_forward, dtype=torch.float32)
        data.pwm_reverse = torch.tensor(pwm_reverse, dtype=torch.float32)

        data.pwm_mask_forward = torch.tensor(pwm_mask_forward, dtype=torch.bool)
        data.pwm_mask_reverse = torch.tensor(pwm_mask_reverse, dtype=torch.bool)

        data.proximity_forward = torch.tensor(proximity_forward, dtype=torch.bool)
        data.proximity_reverse = torch.tensor(proximity_reverse, dtype=torch.bool)

        data.pwm_present = pwm_present

        torch.save(data, out_file)

    except Exception as e:
        print(f"ERROR {pdb_file}: {e}")
        continue

print("Graph construction complete")
