"""Shared heavy-atom distance calculations for RNA and protein structures."""

import numpy as np


def _heavy_atom_coords(residue):
    return np.asarray([atom.coord for atom in residue.get_atoms() if atom.element != 'H']).reshape(-1, 3)


def _minimum_distance(first, second):
    if first.size == 0 or second.size == 0:
        return float('inf')
    differences = first[:, None, :] - second[None, :, :]
    return float(np.sqrt(np.sum(differences * differences, axis=-1).min()))


def calc_residue_dist(residue_one, residue_two):
    """Return the minimum distance between non-hydrogen atoms of two residues."""
    return _minimum_distance(_heavy_atom_coords(residue_one), _heavy_atom_coords(residue_two))


def calc_dist_matrix(structure):
    """Return symmetric residue distances across every chain in the structure."""
    coords = [_heavy_atom_coords(residue) for chain in structure.get_chains() for residue in chain.get_residues()]
    matrix = np.zeros((len(coords), len(coords)), dtype=float)
    for i, first in enumerate(coords):
        for j in range(i, len(coords)):
            matrix[i, j] = matrix[j, i] = _minimum_distance(first, coords[j])
    return matrix
