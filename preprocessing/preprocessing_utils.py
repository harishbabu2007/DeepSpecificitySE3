import json
import numpy as np


def load_motif_annotations(json_paths):
    # accept either a single path or a list
    if isinstance(json_paths, str):
        json_paths = [json_paths]

    annotations = {}
    for json_path in json_paths:
        with open(json_path, "r") as f:
            data = json.load(f)

        for item in data:
            if isinstance(item, list) and len(item) >= 2:
                pdb_id = str(item[0]).lower()
                motifs = item[1]
                if len(pdb_id) == 4 and motifs:
                    # merge: if pdb_id already exists, extend its motif list
                    if pdb_id in annotations:
                        annotations[pdb_id].extend(motifs)
                    else:
                        annotations[pdb_id] = motifs

    return annotations


def compute_ic(pwm_col, epsilon=1e-9):
    """Compute normalized Information Content (0.0 to 2.0 bits)."""
    p = np.clip(pwm_col, epsilon, 1.0)
    p = p / np.sum(p)
    entropy = -np.sum(p * np.log2(p))
    return max(0.0, 2.0 - entropy)


def trim_pwm(pwm, ic_threshold=0.5):
    """Trims uninformative flanks where IC < threshold (DeepPBS Supp Section 1)."""
    s = pwm.shape[0]
    start, end = 0, s
    for i in range(s):
        if compute_ic(pwm[i]) >= ic_threshold:
            start = i
            break
    for i in range(s - 1, -1, -1):
        if compute_ic(pwm[i]) >= ic_threshold:
            end = i + 1
            break
    return pwm[start:end] if start < end else pwm


def ic_weighted_pcc(col_pwm, col_dna, epsilon=1e-9):
    """Calculates IC-weighted PCC between a PWM column and 1-hot DNA base."""
    diff_pwm = col_pwm - 0.25
    diff_dna = col_dna - 0.25
    num = np.sum(diff_pwm * diff_dna)
    den = np.sqrt(np.sum(diff_pwm**2) * np.sum(diff_dna**2))
    pcc = num / den if den > epsilon else 0.0
    ic = compute_ic(col_pwm, epsilon)
    return pcc * 0.5 * ic


def ungapped_align(seq_one_hot, pwm, min_overlap=5):
    """Performs an ungapped sliding-window local alignment (Supp Section 2)."""
    l, s = seq_one_hot.shape[0], pwm.shape[0]
    if l < min_overlap or s < min_overlap:
        return 0, 0, 0, -9999.0

    max_score = -9999.0
    opt_i, opt_j, opt_k = 0, 0, 0

    # Precompute pairwise column similarity to prevent intensive looping
    pairwise_scores = np.zeros((s, l))
    for i in range(s):
        for j in range(l):
            pairwise_scores[i, j] = ic_weighted_pcc(pwm[i], seq_one_hot[j])

    # Slide window configurations
    for i in range(s):
        for k in range(min_overlap, s - i + 1):
            for j in range(l - k + 1):
                # Trace diagonal representing un-gapped sequence alignment match
                score = np.sum(pairwise_scores[i : i + k, j : j + k].diagonal())
                if score > max_score:
                    max_score, opt_i, opt_j, opt_k = score, i, j, k

    return opt_i, opt_j, opt_k, max_score


def get_sequence_one_hot(dna_labels):
    """Converts labels to 5'-to-3' forward and reverse one-hot representations."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    l = len(dna_labels)
    seq_fwd = np.zeros((l, 4), dtype=np.float32)

    for idx, label in enumerate(dna_labels):
        f_base = label[0]
        if f_base in mapping:
            seq_fwd[idx, mapping[f_base]] = 1.0

    # Reverse complement strand sequence (Flipped and complemented)
    seq_rev = seq_fwd[::-1, [3, 2, 1, 0]]
    return seq_fwd, seq_rev


def build_protein_labels(protein_residues):
    labels = []
    for residue in protein_residues:
        aa = residue.get_resname()
        residue_id = residue.id[1]
        labels.append(f"{aa}{residue_id}")
    return labels


def build_dna_labels(dna_pairs):
    labels = []
    for forward, reverse in dna_pairs:
        forward_name = forward.get_resname().replace("D", "")
        if reverse is None:
            labels.append(f"{forward_name}-")
        else:
            reverse_name = reverse.get_resname().replace("D", "")
            labels.append(f"{forward_name}{reverse_name}")
    return labels
