import os
import requests
import numpy as np
from pyjaspar import jaspardb
import sys


def parse_raw_fallback_file(file_path, is_cisbp=False):
    matrix = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                parts = line_str.split()
                if is_cisbp and parts[0].isalpha():
                    continue
                vals = [float(x) for x in parts]
                if is_cisbp and len(vals) == 5:
                    matrix.append(vals[1:])
                elif not is_cisbp and len(vals) == 4:
                    matrix.append(vals)
    except Exception:
        return None
    mat_arr = np.array(matrix, dtype=np.float32)
    if mat_arr.size == 0:
        return None
    row_sums = mat_arr.sum(axis=1, keepdims=True)
    ppm = np.divide(mat_arr, row_sums, out=np.zeros_like(mat_arr), where=row_sums != 0)
    ppm = (ppm * 100 + 0.5) / (100 + 2.0)
    return np.log2(ppm / 0.25)


def parse_uniprobe_file(file_path):
    matrix_rows = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                parts = line_str.split()
                if not parts:
                    continue
                if parts[0].rstrip(":").upper() in ["A", "C", "G", "T"]:
                    parts = parts[1:]
                try:
                    matrix_rows.append([float(x) for x in parts if x])
                except ValueError:
                    continue
    except Exception:
        return None
    mat_arr = np.array(matrix_rows, dtype=np.float32)
    if mat_arr.size == 0:
        return None
    if mat_arr.shape[0] == 4 and mat_arr.shape[1] != 4:
        mat_arr = mat_arr.T
    elif mat_arr.shape[0] != 4 and mat_arr.shape[1] == 4:
        pass
    else:
        return None
    row_sums = mat_arr.sum(axis=1, keepdims=True)
    ppm = np.divide(mat_arr, row_sums, out=np.zeros_like(mat_arr), where=row_sums != 0)
    ppm = (ppm * 100 + 0.5) / (100 + 2.0)
    return np.log2(ppm / 0.25)


def get_pwm_matrix_from_annotations(
    pdb_id,
    annotations,
    hocomoco_dir="../data/motifs/hocomoco",
    cisbp_dir="../data/motifs/cisbp",
    uniprobe_dir="../data/motifs/uniprobe",
):
    pdb_id = pdb_id.lower()

    if pdb_id not in annotations or not annotations[pdb_id]:
        return None

    jdb_obj = jaspardb(release="JASPAR2024")
    for site_motifs in annotations[pdb_id]:
        for motif_info in site_motifs:
            if len(motif_info) != 2:
                continue
            db_name, motif_id = motif_info
            if db_name == "JASPAR":
                try:
                    motif = jdb_obj.fetch_motif_by_id(motif_id.replace(".jaspar", ""))
                    if motif:
                        ppm = motif.counts.normalize(pseudocounts=0.5)
                        pwm_dict = ppm.log_odds()
                        return np.array(
                            [
                                pwm_dict["A"],
                                pwm_dict["C"],
                                pwm_dict["G"],
                                pwm_dict["T"],
                            ],
                            dtype=np.float32,
                        ).T
                except Exception:
                    continue
            elif db_name == "HOCOMOCO":
                mat = parse_raw_fallback_file(
                    os.path.join(hocomoco_dir, f"{motif_id}.pwm"), False
                )
                if mat is not None:
                    return mat
            elif db_name == "CIS-BP":
                mat = parse_raw_fallback_file(
                    os.path.join(cisbp_dir, "pwms", f"{motif_id}.txt"), True
                )
                if mat is not None:
                    return mat
            elif db_name == "UniPROBE":
                if os.path.exists(uniprobe_dir):
                    for root, _, files in os.walk(uniprobe_dir):
                        for fname in files:
                            if (
                                motif_id in fname
                                and fname.upper().endswith(".PWM")
                                and ".RC." not in fname.upper()
                            ):
                                mat = parse_uniprobe_file(os.path.join(root, fname))
                                if mat is not None:
                                    return mat
    return None


def get_metadata_from_pdb(pdb_id):
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
    uniprot_ids, gene_symbols = [], []
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and response.json():
            for up_id, info in (
                response.json()[list(response.json().keys())[0]]
                .get("UniProt", {})
                .items()
            ):
                uniprot_ids.append(up_id)
                if "identifier" in info:
                    gene_symbols.append(info["identifier"].split("_")[0].upper())
    except Exception:
        pass
    return list(set(uniprot_ids)), list(set(gene_symbols))


def build_jaspar_index(release="JASPAR2024"):
    jdb_obj = jaspardb(release=release)
    all_motifs = jdb_obj.fetch_motifs(all=True, all_versions=True)
    uniprot_to_motifs, gene_to_motifs = {}, {}
    for motif in all_motifs:
        if motif.acc:
            for up_id in motif.acc:
                uniprot_to_motifs.setdefault(up_id, []).append(motif)
        if motif.name:
            gene_to_motifs.setdefault(motif.name.upper(), []).append(motif)
    return {"by_uniprot": uniprot_to_motifs, "by_gene": gene_to_motifs}


def check_local_database_fallbacks(
    pdb_id,
    hocomoco_dir="../data/motifs/hocomoco",
    cisbp_dir="../data/motifs/cisbp",
    uniprobe_dir="../data/motifs/uniprobe",
):
    _, gene_symbols = get_metadata_from_pdb(pdb_id)
    if not gene_symbols:
        return None

    if os.path.exists(hocomoco_dir):
        for gene in gene_symbols:
            for fname in os.listdir(hocomoco_dir):
                if fname.upper().startswith(f"{gene}_"):
                    return parse_raw_fallback_file(
                        os.path.join(hocomoco_dir, fname), False
                    )

    if os.path.exists(cisbp_dir):
        for gene in gene_symbols:
            for fname in os.listdir(cisbp_dir):
                if fname.upper().startswith(gene):
                    return parse_raw_fallback_file(os.path.join(cisbp_dir, fname), True)

    if os.path.exists(uniprobe_dir):
        for root, _, files in os.walk(uniprobe_dir):
            for gene in gene_symbols:
                for fname in files:
                    fname_upper = fname.upper()
                    if ".RC." in fname_upper:
                        continue
                    if fname_upper.startswith(f"{gene}_") or fname_upper.startswith(
                        gene
                    ):
                        if fname_upper.endswith(".PWM"):
                            mat = parse_uniprobe_file(os.path.join(root, fname))
                            if mat is not None:
                                return mat
    return None


def get_hybrid_pwm(
    pdb_id,
    annotations,
    jaspar_indices,
    hoco="../data/motifs/hocomoco",
    cis="../data/motifs/cisbp",
    uni="../data/motifs/uniprobe",
):
    """Tries strict JSON first. If it fails, hunts down the motif via API & local scan."""
    mat = get_pwm_matrix_from_annotations(pdb_id, annotations, hoco, cis, uni)
    if mat is not None:
        return mat

    uniprot_ids, gene_symbols = get_metadata_from_pdb(pdb_id)
    matched_motifs = []
    for up_id in uniprot_ids:
        if up_id in jaspar_indices["by_uniprot"]:
            matched_motifs.extend(jaspar_indices["by_uniprot"][up_id])
    if not matched_motifs:
        for gene in gene_symbols:
            if gene in jaspar_indices["by_gene"]:
                matched_motifs.extend(jaspar_indices["by_gene"][gene])
    if matched_motifs:
        best_motif = sorted(
            matched_motifs, key=lambda m: (abs(len(m) - 18), m.matrix_id)
        )[0]
        try:
            ppm = best_motif.counts.normalize(pseudocounts=0.5)
            pwm_dict = ppm.log_odds()
            return np.array(
                [pwm_dict["A"], pwm_dict["C"], pwm_dict["G"], pwm_dict["T"]],
                dtype=np.float32,
            ).T
        except Exception:
            pass

    return check_local_database_fallbacks(pdb_id, hoco, cis, uni)



