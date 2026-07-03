import numpy as np

from get_pwm import (
    get_hybrid_pwm,
)

from preprocessing_utils import (
    trim_pwm,
    ungapped_align,
    get_sequence_one_hot,
    build_dna_labels
)

from Bio.PDB.Residue import Residue

from constants import (
    DNA_RESIDUES,
)

from dna_definitions import BASE_NAME_MAP
from dna_definitions import WC_PAIR_ATOMS
from constants import WC_PAIR_DISTANCE_THRESHOLD

from geometry import distance

DNA_COMPLEMENT = {
    "A": "T",
    "T": "A",
    "G": "C",
    "C": "G",
    "DA": "DT",
    "DT": "DA",
    "DG": "DC",
    "DC": "DG",
}


class StructureRejected(Exception):
    pass


def extract_dna_chains(model):
    """
    Returns DNA chains sorted by size.
    """

    dna_chains = []

    for chain in model:

        dna_residues = []

        for residue in chain:

            resname = residue.get_resname().strip()

            if resname in DNA_RESIDUES:
                dna_residues.append(residue)

        if dna_residues:
            dna_chains.append((chain, len(dna_residues)))

    dna_chains.sort(key=lambda x: x[1], reverse=True)

    return [x[0] for x in dna_chains]


def compute_wc_pair_score(residue_a, residue_b):
    """
    Lower score = better Watson-Crick pair.
    """

    base_a = get_base_letter(residue_a)

    base_b = get_base_letter(residue_b)

    pair_key = (base_a, base_b)

    if pair_key not in WC_PAIR_ATOMS:
        return None

    distances = []

    for atom_a, atom_b in WC_PAIR_ATOMS[pair_key]:

        if atom_a not in residue_a:
            return None

        if atom_b not in residue_b:
            return None

        coord_a = residue_a[atom_a].coord

        coord_b = residue_b[atom_b].coord

        distances.append(distance(coord_a, coord_b))

    if not distances:
        return None

    return sum(distances) / len(distances)


def get_base_letter(residue):

    resname = residue.get_resname().strip()

    return BASE_NAME_MAP.get(resname, None)


def validate_pairing(base_pairs):
    """
    Allow terminal overhangs only.
    """

    unpaired = []

    for idx, pair in enumerate(base_pairs):

        if pair[1] is None:
            unpaired.append(idx)

    if not unpaired:
        return

    n = len(base_pairs)

    valid = set()

    i = 0

    while i < n and base_pairs[i][1] is None:
        valid.add(i)
        i += 1

    i = n - 1

    while i >= 0 and base_pairs[i][1] is None:
        valid.add(i)
        i -= 1

    for idx in unpaired:

        if idx not in valid:

            raise StructureRejected("Internal unpaired DNA base")


def find_base_pairs(model):
    """
    Pair DNA bases using Watson-Crick geometry.
    """

    dna_chains = extract_dna_chains(model)

    if len(dna_chains) < 2:

        raise StructureRejected("Less than two DNA chains")

    chain_a = list(dna_chains[0])
    chain_b = list(dna_chains[1])

    used_b = set()

    pairs = []

    for residue_a in chain_a:

        base_a = get_base_letter(residue_a)

        if base_a is None:
            continue

        best_score = float("inf")
        best_match = None

        for idx, residue_b in enumerate(chain_b):

            if idx in used_b:
                continue

            score = compute_wc_pair_score(residue_a, residue_b)

            if score is None:
                continue

            if score < best_score:

                best_score = score
                best_match = (idx, residue_b)

        if best_match is None or best_score > WC_PAIR_DISTANCE_THRESHOLD:

            pairs.append((residue_a, None))

            continue

        idx, residue_b = best_match

        used_b.add(idx)

        pairs.append((residue_a, residue_b))

    paired_count = sum(pair[1] is not None for pair in pairs)

    if paired_count < 5:
        raise StructureRejected("Insufficient paired DNA")

    validate_pairing(pairs)

    return pairs


def generate_training_pwm(
    structure,
    pdb_id,
    proximity_mask,
    annotations,
    jaspar_indices,
):
    """
    Returns

    pwm_forward          (Nd,4)
    pwm_reverse          (Nd,4)

    mask_forward         (Nd,)
    mask_reverse         (Nd,)

    proximity_forward    (Nd,)
    proximity_reverse    (Nd,)

    pwm_present          bool
    """

    model = structure[0]

    dna_pairs = find_base_pairs(model)

    dna_labels = build_dna_labels(dna_pairs)

    pwm_matrix = get_hybrid_pwm(
        pdb_id,
        annotations,
        jaspar_indices,
    )

    seq_fwd, seq_rev = get_sequence_one_hot(dna_labels)

    Nd = len(dna_labels)

    proximity_forward = proximity_mask.copy()

    proximity_reverse = proximity_mask[::-1].copy()

    ####################################################################
    # No experimental PWM
    ####################################################################
    
    if pwm_matrix is None:

        pwm_forward = seq_fwd.copy()
        pwm_reverse = seq_rev.copy()

        pwm_forward[~proximity_forward] = 0.25
        pwm_reverse[~proximity_reverse] = 0.25

        return (
            pwm_forward.astype(np.float32),
            pwm_reverse.astype(np.float32),
            proximity_forward.astype(bool),
            proximity_reverse.astype(bool),
            proximity_forward.astype(bool),
            proximity_reverse.astype(bool),
            False,
        )

    ####################################################################
    # Experimental PWM exists
    ####################################################################

    ppm = 0.25 * np.power(2.0, pwm_matrix)
    ppm /= ppm.sum(axis=1, keepdims=True)

    ppm = trim_pwm(
        ppm,
        ic_threshold=0.5,
    )

    opt_i_fwd, opt_j_fwd, opt_k_fwd, score_fwd = ungapped_align(
        seq_fwd,
        ppm,
    )

    opt_i_rev, opt_j_rev, opt_k_rev, score_rev = ungapped_align(
        seq_rev,
        ppm,
    )

    if max(score_fwd, score_rev) == -9999:

        pwm_forward = seq_fwd.copy()
        pwm_reverse = seq_rev.copy()

        pwm_forward[~proximity_forward] = 0.25
        pwm_reverse[~proximity_reverse] = 0.25

        return (
            pwm_forward.astype(np.float32),
            pwm_reverse.astype(np.float32),
            proximity_forward.astype(bool),
            proximity_reverse.astype(bool),
            proximity_forward.astype(bool),
            proximity_reverse.astype(bool),
            False,
        )

    pwm_forward = np.full(
        (Nd, 4),
        0.25,
        dtype=np.float32,
    )

    pwm_reverse = np.full(
        (Nd, 4),
        0.25,
        dtype=np.float32,
    )

    mask_forward = np.zeros(
        Nd,
        dtype=bool,
    )

    mask_reverse = np.zeros(
        Nd,
        dtype=bool,
    )

    ##########################################################
    # Forward alignment
    ##########################################################

    if score_fwd >= score_rev:

        pwm_forward = np.full(
            (Nd, 4),
            0.25,
            dtype=np.float32,
        )

        # copy the ENTIRE aligned PWM
        start_pwm = opt_i_fwd - opt_j_fwd
        end_pwm = start_pwm + Nd

        src_start = max(0, start_pwm)
        src_end = min(ppm.shape[0], end_pwm)

        dst_start = max(0, -start_pwm)
        dst_end = dst_start + (src_end - src_start)

        pwm_forward[dst_start:dst_end] = ppm[src_start:src_end]

        # motif mask only
        motif_start = opt_j_fwd
        motif_end = opt_j_fwd + opt_k_fwd

        mask_forward[motif_start:motif_end] = True

        pwm_reverse = pwm_forward[::-1, [3, 2, 1, 0]]

        mask_reverse = mask_forward[::-1]

    ##########################################################
    # Reverse alignment
    ##########################################################

    else:
        reverse_tmp = np.full(
            (Nd, 4),
            0.25,
            dtype=np.float32,
        )

        start_pwm = opt_i_rev - opt_j_rev
        end_pwm = start_pwm + Nd

        src_start = max(0, start_pwm)
        src_end = min(ppm.shape[0], end_pwm)

        dst_start = max(0, -start_pwm)
        dst_end = dst_start + (src_end - src_start)

        reverse_tmp[dst_start:dst_end] = ppm[src_start:src_end]

        reverse_mask = np.zeros(
            Nd,
            dtype=bool,
        )

        motif_start = opt_j_rev
        motif_end = opt_j_rev + opt_k_rev

        reverse_mask[motif_start:motif_end] = True

        pwm_reverse = reverse_tmp
        mask_reverse = reverse_mask

        pwm_forward = reverse_tmp[::-1, [3, 2, 1, 0]]
        mask_forward = reverse_mask[::-1]

    return (
        pwm_forward.astype(np.float32),
        pwm_reverse.astype(np.float32),
        mask_forward.astype(bool),
        mask_reverse.astype(bool),
        proximity_forward.astype(bool),
        proximity_reverse.astype(bool),
        True,
    )
