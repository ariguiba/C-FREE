
import logging
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from typing import Any, Dict, List

from torch_geometric.utils import (
    to_smiles,
)

x_map: Dict[str, List[Any]] = {
    "atomic_num": list(range(0, 119)),
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
        "CHI_TETRAHEDRAL",
        "CHI_ALLENE",
        "CHI_SQUAREPLANAR",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
    ],
    "degree": list(range(0, 11)),
    "formal_charge": list(range(-5, 7)),
    "num_hs": list(range(0, 9)),
    "num_radical_electrons": list(range(0, 5)),
    "hybridization": [
        "UNSPECIFIED",
        "S",
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "OTHER",
    ],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}

e_map: Dict[str, List[Any]] = {
    "bond_type": [
        "UNSPECIFIED",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "QUADRUPLE",
        "QUINTUPLE",
        "HEXTUPLE",
        "ONEANDAHALF",
        "TWOANDAHALF",
        "THREEANDAHALF",
        "FOURANDAHALF",
        "FIVEANDAHALF",
        "AROMATIC",
        "IONIC",
        "HYDROGEN",
        "THREECENTER",
        "DATIVEONE",
        "DATIVE",
        "DATIVEL",
        "DATIVER",
        "OTHER",
        "ZERO",
    ],
    "stereo": [
        "STEREONONE",
        "STEREOANY",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
    ],
    "is_conjugated": [False, True],
}


def to_rdmol(
    data: Data,
    kekulize: bool = False,
) -> Any:
    """Converts a :class:`torch_geometric.data.Data` instance to a
    :class:`rdkit.Chem.Mol` instance.

    Args:
        data (torch_geometric.data.Data): The molecular graph data.
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem

    mol = Chem.RWMol()

    assert data.x is not None
    assert data.num_nodes is not None
    assert data.edge_index is not None
    assert data.edge_attr is not None
    for i in range(data.num_nodes):
        atom = Chem.Atom(int(data.x[i, 0]))
        atom.SetChiralTag(Chem.rdchem.ChiralType.values[int(data.x[i, 1])])
        atom.SetFormalCharge(x_map["formal_charge"][int(data.x[i, 3])])
        atom.SetNumExplicitHs(x_map["num_hs"][int(data.x[i, 4])])
        atom.SetNumRadicalElectrons(x_map["num_radical_electrons"][int(data.x[i, 5])])
        atom.SetHybridization(Chem.rdchem.HybridizationType.values[int(data.x[i, 6])])
        atom.SetIsAromatic(bool(data.x[i, 7]))
        mol.AddAtom(atom)

    edges = [tuple(i) for i in data.edge_index.t().tolist()]
    visited = set()

    for i in range(len(edges)):
        src, dst = edges[i]
        if tuple(sorted(edges[i])) in visited:
            continue

        bond_type = Chem.BondType.values[int(data.edge_attr[i, 0])]
        mol.AddBond(src, dst, bond_type)

        # Set stereochemistry:
        stereo = Chem.rdchem.BondStereo.values[int(data.edge_attr[i, 1])]
        if stereo != Chem.rdchem.BondStereo.STEREONONE:
            db = mol.GetBondBetweenAtoms(src, dst)
            db.SetStereoAtoms(dst, src)
            db.SetStereo(stereo)

        # Set conjugation:
        is_conjugated = bool(data.edge_attr[i, 2])
        mol.GetBondBetweenAtoms(src, dst).SetIsConjugated(is_conjugated)

        visited.add(tuple(sorted(edges[i])))

    mol = mol.GetMol()

    if kekulize:
        Chem.Kekulize(mol)

    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol)

    return mol


i_conformers = 0


class AugmentWithConformers:
    def __init__(self, num_conformers=200):
        super(AugmentWithConformers, self).__init__()
        self.num_conformers = num_conformers

    def __call__(self, graph: Data):
        global i_conformers
        pos = []
        max_attempts = 3
        attempt = 0
        
        while len(pos) == 0 and attempt < max_attempts:
            attempt += 1
            mol = Chem.MolFromSmiles(graph.smiles)
            molH = mol #Chem.AddHs(mol)
            
            # Removed for SIDER as the molecules are too big!
            
            # params.pruneRmsThresh = 0.9  # prune aggressively
            # params.randomSeed = 42
            # params.maxAttempts = 5   # don’t retry forever
            # params.numThreads = 0     # let RDKit auto-manage threads
            # params.useExpTorsionAnglePrefs = True
            # params.useBasicKnowledge = True

            # confIds = AllChem.EmbedMultipleConfs(
            #     molH, numConfs=self.num_conformers, params=params
            # )
            z = torch.tensor(
                [atom.GetAtomicNum() for atom in molH.GetAtoms()], dtype=torch.long
            )
            pos = [
                torch.tensor(conf.GetPositions(), dtype=torch.float)
                for conf in molH.GetConformers()
            ]
            if len(pos) == 0:
                logging.warn(f"Failed to generate conformers, attempt {attempt}/{max_attempts}")
        
        # If all attempts failed, create zeros tensor as fallback
        if len(pos) == 0:
            logging.warn(f"Failed to generate conformers after {max_attempts} attempts, using zeros tensor")
            # Create a zeros tensor with appropriate shape (num_atoms, 3)
            num_atoms = len([atom.GetAtomicNum() for atom in Chem.MolFromSmiles(graph.smiles).GetAtoms()])
            zeros_conformer = torch.zeros((num_atoms, 3), dtype=torch.float)
            pos = [zeros_conformer] * self.num_conformers
            
            # Also need to set z for the fallback case
            mol = Chem.MolFromSmiles(graph.smiles)
            molH = mol #Chem.AddHs(mol)
            z = torch.tensor(
                [atom.GetAtomicNum() for atom in molH.GetAtoms()], dtype=torch.long
            )
        
        if len(pos) != self.num_conformers and len(pos) != 0:
            logging.warn(
                f"Number of conformers generated is not equal to {self.num_conformers} ({len(pos)}); duplicating."
            )
            repeats = (self.num_conformers // len(pos)) + 1
            pos = (pos * repeats)[: self.num_conformers]
        
        graph.z = z
        graph.pos = pos
        i_conformers += 1
        #logging.warn(f"{i_conformers} molecules processed")
        return graph


class PygWithConformers:
    def __init__(self, num_conformers=200):
        super(PygWithConformers, self).__init__()
        self.num_conformers = num_conformers

    def __call__(self, graph: Data):

        # mol = to_rdmol(graph)
        # smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        # graph.smiles = smiles

        mol = graph  # your RDKit Mol
        try:
            Chem.SanitizeMol(mol)  # tries to adjust valences, charges etc.
            smiles = Chem.MolToSmiles(mol)
        except Chem.rdchem.KekulizeException:
            print("Kekulize failed")
        except ValueError as e:
            print("Invalid molecule:", e)
        # smiles = to_smiles(graph, kekulize=False)
        mol = Chem.MolFromSmiles(smiles)

        molH = Chem.AddHs(mol)
        confIds = AllChem.EmbedMultipleConfs(molH, numConfs=self.num_conformers)
        z = torch.tensor(
            [atom.GetAtomicNum() for atom in molH.GetAtoms()], dtype=torch.long
        )
        pos = [
            torch.tensor(conf.GetPositions(), dtype=torch.float)
            for conf in molH.GetConformers()
        ]
        graph.z = z
        graph.pos = pos
        return graph
