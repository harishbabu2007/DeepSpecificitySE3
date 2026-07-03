# BASE_HEAVY_ATOMS = {
#     "A": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
#     "G": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
#     "C": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
#     "T": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"],
# }

BASE_HEAVY_ATOMS = {
    "A": ["N9"],
    "G": ["N9"],
    "C": ["N1"],
    "T": ["N1"],
}

BASE_NAME_MAP = {
    "DA": "A",
    "DG": "G",
    "DC": "C",
    "DT": "T",
    "A": "A",
    "G": "G",
    "C": "C",
    "T": "T",
}


CANONICAL_PAIRS = {("A", "T"), ("T", "A"), ("G", "C"), ("C", "G")}


DNA_DONOR_ATOMS = {"A": ["N6"], "G": ["N2"], "C": ["N4"], "T": []}


DNA_ACCEPTOR_ATOMS = {
    "A": ["N1", "N7"],
    "G": ["O6", "N7", "N3"],
    "C": ["O2", "N3"],
    "T": ["O2", "O4", "N3"],
}


# Used later for hydrogen placement

DNA_DONOR_REFERENCE_ATOMS = {
    # amino group attached to C6
    "A": {"N6": ("C6", "C5")},
    # amino group attached to C2
    "G": {"N2": ("C2", "N1")},
    # amino group attached to C4
    "C": {"N4": ("C4", "N3")},
}

WC_PAIR_ATOMS = {
    ("A", "T"): [("N6", "O4"), ("N1", "N3")],
    ("T", "A"): [("O4", "N6"), ("N3", "N1")],
    ("G", "C"): [("O6", "N4"), ("N1", "N3"), ("N2", "O2")],
    ("C", "G"): [("N4", "O6"), ("N3", "N1"), ("O2", "N2")],
}
