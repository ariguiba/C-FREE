# Source: https://github.com/KemperNiklas/FragNet/blob/view/publication/data/fragmentations/fragmentations.py

from itertools import permutations
from typing import Literal

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

fragment2type = {"ring": 0, "path": 1, "junction": 2}
ATOM_LIST = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Cu", "Zn", 'Co', "Mn", 'As', 'Al', 'Ni', 'Se', 'Si', 'H', 'He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Fe', 'Ga', 'Ge', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
             'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Fl', 'Uup', 'Lv', 'Uus', 'Uuo']


def is_leaf(node_id, graph):

    neighbors = get_neighbors(node_id, graph)
    if len(neighbors) == 1:
        neighbor = neighbors[0]
        if graph.mol.GetAtomWithIdx(neighbor).IsInRing():
            return True
        nns = get_neighbors(neighbor, graph)
        degree_nn = [get_degree(nn, graph) for nn in nns]
        if len([degree for degree in degree_nn if degree >= 2]) >= 2:
            return True
        # one neighbor neighbor with degree one is not a leaf
        potential_leafs = [nn for nn in nns if get_degree(nn, graph) == 1]
        atom_types = [(ATOM_LIST.index(graph.mol.GetAtomWithIdx(
            nn).GetSymbol()), nn) for nn in potential_leafs]
        sorted_idx = np.sort(atom_types)
        if sorted_idx[-1][1] == node_id:
            # node at end of path
            return False
        else:
            return True
    return False

def get_neighbors(node_id, graph):
    return (graph.edge_index[1, graph.edge_index[0, :] == node_id]).tolist()

def get_degree(node_id, graph):
    return len(get_neighbors(node_id, graph))

class Rings(BaseTransform):
    def __init__(self, vocab_size=15) -> None:
        self.max_vocab_size = vocab_size
        self.max_ring_size = vocab_size + 2

    def __call__(self, graph):
        mol = graph.mol
        rings = Chem.GetSymmSSSR(mol)
        node_substructures = [[] for _ in range(graph.num_nodes)]
        fragment_types = []
        fragment_id = 0
        for i in range(len(rings)):
            ring = list(rings[i])
            fragment_types.append([fragment2type["ring"], len(ring)])
            if len(ring) <= self.max_ring_size:
                for atom in ring:
                    fragment_type = len(ring) - 3
                    node_substructures[atom].append(
                        (fragment_id, fragment_type))
                fragment_id += 1
            else:
                for atom in ring:
                    fragment_type = self.max_vocab_size - 1  # max fragment_type number
                    node_substructures[atom].append(
                        (fragment_id, fragment_type))
                fragment_id += 1
        graph.substructures = node_substructures
        if fragment_types:
            graph.fragment_types = torch.tensor(
                fragment_types, dtype=torch.long)
        else:
            graph.fragment_types = torch.empty((0, 2), dtype=torch.long)
        return graph


class RingsPaths(BaseTransform):
    def __init__(self, vocab_size=30, max_ring=15, cut_leafs=False):
        self.max_ring = max_ring
        assert (vocab_size > max_ring)
        self.max_path = vocab_size - max_ring
        self.rings = Rings(max_ring)
        self.cut_leafs = cut_leafs
        self.vocab_size = vocab_size

    def get_frag_type(self, type: Literal["ring", "path"], size):
        if type == "ring":
            return size - 3 if size - 3 < self.max_ring else self.max_ring - 1
        else:  # type == "path"
            offset = self.max_ring
            return offset + size - 2 if size - 2 < self.max_path else offset + self.max_path - 1

    def __call__(self, graph):
        # first find rings
        self.rings(graph)

        # now find paths
        max_frag_id = max([frag_id for frag_infos in graph.substructures for (
            frag_id, _) in frag_infos], default=-1)
        fragment_id = max_frag_id + 1

        fragment_types = []

        # find paths
        visited = set()
        for bond in graph.mol.GetBonds():

            if not bond.IsInRing() and bond.GetIdx() not in visited:
                if self.cut_leafs and is_leaf(bond.GetBeginAtomIdx(), graph) and is_leaf(bond.GetEndAtomIdx(), graph):
                    continue
                visited.add(bond.GetIdx())
                in_path = []
                to_do = set([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                while to_do:
                    next_node = to_do.pop()
                    in_path.append(next_node)
                    neighbors = [neighbor for neighbor in get_neighbors(
                        next_node, graph) if not is_leaf(neighbor, graph) or not self.cut_leafs]
                    if not graph.mol.GetAtomWithIdx(next_node).IsInRing() and not len(neighbors) > 2:
                        # not in ring and not a junction
                        new_neighbors = [
                            neighbor for neighbor in neighbors if neighbor not in in_path]
                        visited.update([graph.mol.GetBondBetweenAtoms(
                            next_node, neighbor).GetIdx() for neighbor in new_neighbors])
                        to_do.update(new_neighbors)

                path_info = (fragment_id, self.get_frag_type(
                    "path", len(in_path)))
                fragment_types.append([fragment2type["path"], len(in_path)])
                fragment_id += 1
                for node_id in in_path:
                    graph.substructures[node_id].append(path_info)

        graph.fragment_types = torch.concat(
            [graph.fragment_types, torch.tensor(fragment_types, dtype=torch.long)], dim=0)

        return graph