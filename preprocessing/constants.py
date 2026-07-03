# H-Bond Geometry

HBOND_MIN_DISTANCE = 2.2
HBOND_MAX_DISTANCE = 3.9 

# Donor-Hydrogen-Acceptor angle
HBOND_MIN_ANGLE = 120  # degrees

# Feature Dimensions

MAX_DNA_BASE_HEAVY_ATOMS = 1
MAX_PROTEIN_SIDECHAIN_HEAVY_ATOMS = 10
MAX_PROTEIN_LENGTH = 1000

DNA_FEATURE_SIZE = 88
PROTEIN_FEATURE_SIZE = 64


# Dataset Rules

ALLOW_TERMINAL_OVERHANGS = True

REJECT_MODIFIED_RESIDUES = True
REJECT_ALTERNATE_LOCATIONS = True
REJECT_MULTI_MODEL = True

STORE_DTYPE_FEATURES = "float32"
STORE_DTYPE_BOND_MATRIX = "uint8"


# DNA Backbone Atom Order

DNA_BACKBONE_ORDER = [
    "OP1",
    "OP2",
    "P",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "C1'"
]

# Canonical DNA Bases

DNA_BASES = ["A", "C", "G", "T"]

BASE_TO_INDEX = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3
}

COMPLEMENT = {
    "A": "T",
    "T": "A",
    "G": "C",
    "C": "G"
}


# Protein Residues
# MUST NEVER CHANGE ONCE DATASET IS GENERATED

AMINO_ACIDS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]

AA_TO_INDEX = {
    aa: idx
    for idx, aa in enumerate(AMINO_ACIDS)
}

# Canonical Protein Backbone=

PROTEIN_BACKBONE_ORDER = [
    "N",
    "CA",
    "C",
    "O",
]


# DNA Donor / Acceptor Atoms

DNA_DONORS = {
    "A": ["N6"],
    "G": ["N2"],
    "C": ["N4"],
    "T": []
}

DNA_ACCEPTORS = {
    "A": ["N1", "N7"],
    "G": ["O6", "N7", "N3"],
    "C": ["O2", "N3"],
    "T": ["O2", "O4", "N3"]
}


# Protein Donor / Acceptor Atoms

PROTEIN_DONORS = {
    "ARG": ["NE", "NH1", "NH2"],
    "LYS": ["NZ"],
    "HIS": ["ND1", "NE2"],
    "ASN": ["ND2"],
    "GLN": ["NE2"],
    "SER": ["OG"],
    "THR": ["OG1"],
    "TYR": ["OH"],
    "TRP": ["NE1"],
}

PROTEIN_ACCEPTORS = {
    "HIS": ["ND1", "NE2"],
    "ASP": ["OD1", "OD2"],
    "ASN": ["OD1"],
    "GLU": ["OE1", "OE2"],
    "GLN": ["OE1"],
    "SER": ["OG"],
    "THR": ["OG1"],
    "TYR": ["OH"],
}

COORDINATE_SCALE_FACTOR = 100.0
RESIDUE_BASE_CUTOFF = 8.0
PROTEIN_BACKBONE_DONOR = "N"


# Allowed DNA Residues

DNA_RESIDUES = {
    "DA", "DG", "DC", "DT",
    "A", "G", "C", "T"
}
WC_PAIR_DISTANCE_THRESHOLD = 4.5

# Modified Residues To Reject

REJECT_RESIDUES = {
    "MSE",
}
