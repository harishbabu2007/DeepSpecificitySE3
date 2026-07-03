import numpy as np


def distance(coord1, coord2):
    """
    Euclidean distance between two points.
    """
    return np.linalg.norm(coord1 - coord2)


def squared_distance(coord1, coord2):
    """
    Squared Euclidean distance.
    Faster when only comparisons are needed.
    """
    diff = coord1 - coord2
    return np.dot(diff, diff)


def unit_vector(vector):
    """
    Normalize a vector.
    """
    norm = np.linalg.norm(vector)

    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)

    return vector / norm


def angle_between(v1, v2):
    """
    Returns angle in degrees.
    """

    u1 = unit_vector(v1)
    u2 = unit_vector(v2)

    dot = np.clip(np.dot(u1, u2), -1.0, 1.0)

    return np.degrees(np.arccos(dot))


def calculate_hbond_angle(donor_coord, hydrogen_coord, acceptor_coord):
    """
    Donor-Hydrogen-Acceptor angle.
    """

    dh = donor_coord - hydrogen_coord
    ah = acceptor_coord - hydrogen_coord

    return angle_between(dh, ah)


def centroid(coords):
    """
    Mean coordinate.
    """

    coords = np.asarray(coords)

    if len(coords) == 0:
        return np.zeros(3, dtype=np.float32)

    return np.mean(coords, axis=0)


def pad_coordinate_list(coordinates, target_atoms):
    """
    Converts coordinate list to fixed size vector.

    Example:
        8 atoms -> 24 values
        target = 11 atoms

        returns 33 values
    """

    output = []

    for coord in coordinates:
        output.extend(coord.tolist())

    required_length = target_atoms * 3

    if len(output) < required_length:
        output.extend([0.0] * (required_length - len(output)))

    return np.asarray(output, dtype=np.float32)


def atom_exists(residue, atom_name):
    """
    Convenience helper.
    """
    return atom_name in residue


def get_atom_coord(residue, atom_name):
    """
    Returns atom coordinates.

    Raises KeyError if atom missing.
    """

    return residue[atom_name].coord.astype(np.float32)

def get_dna_c1_atom(residue):
    """
    Supports both modern (C1') and legacy (C1*) PDB naming.
    """

    if "C1'" in residue:
        return residue["C1'"]

    if "C1*" in residue:
        return residue["C1*"]

    return None


def get_dna_atom(residue, atom_name):

    if atom_name in residue:
        return residue[atom_name]

    if "'" in atom_name:

        legacy_name = atom_name.replace("'", "*")

        if legacy_name in residue:
            return residue[legacy_name]

    return None

def safe_get_atom_coord(residue, atom_name):
    """
    Returns coordinate or None.
    """

    if atom_name not in residue:
        return None

    return residue[atom_name].coord.astype(np.float32)


def minimum_atom_distance(atom_coords_a, atom_coords_b):
    """
    Smallest distance between
    two atom sets.
    """

    min_dist = float("inf")

    for a in atom_coords_a:
        for b in atom_coords_b:

            d = distance(a, b)

            if d < min_dist:
                min_dist = d

    return min_dist


def project_point_onto_line(point, line_start, line_end):
    """
    Used later during hydrogen placement.
    """

    line = line_end - line_start
    line_unit = unit_vector(line)

    projection_length = np.dot(point - line_start, line_unit)

    return line_start + projection_length * line_unit


def normalize_coordinates(coordinates, center):
    """
    Translate coordinates around center.

    Useful if later you decide
    to center structures before training.
    """

    return coordinates - center
