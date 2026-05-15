# allowable_features = {
#     "possible_atomic_num_list": list(range(1, 119)) + ['misc'],
#     "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
#     "possible_chirality_list": [
#         Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
#         Chem.rdchem.ChiralType.CHI_OTHER,
#         Chem.rdchem.ChiralType.CHI_SQUAREPLANAR,
#         Chem.rdchem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
#         Chem.rdchem.ChiralType.CHI_OCTAHEDRAL,
        
#     ],
#     "possible_hybridization_list": [
#         Chem.rdchem.HybridizationType.S,
#         Chem.rdchem.HybridizationType.SP,
#         Chem.rdchem.HybridizationType.SP2,
#         Chem.rdchem.HybridizationType.SP3,
#         Chem.rdchem.HybridizationType.SP3D,
#         Chem.rdchem.HybridizationType.SP3D2,
#         Chem.rdchem.HybridizationType.UNSPECIFIED,
#     ],
#     "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
#     "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
#     'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
#     "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6, 7],
#     "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'possible_is_aromatic_list': [False, True],
#     'possible_is_in_ring_list': [False, True],
#     "possible_bonds": [
#         Chem.rdchem.BondType.SINGLE,
#         Chem.rdchem.BondType.DOUBLE,
#         Chem.rdchem.BondType.TRIPLE,
#         Chem.rdchem.BondType.AROMATIC,
#         Chem.rdchem.BondType.DATIVE
#     ],
#     "possible_bond_dirs": [  # only for double bond stereo information
#         Chem.rdchem.BondDir.NONE,
#         Chem.rdchem.BondDir.ENDUPRIGHT,
#         Chem.rdchem.BondDir.ENDDOWNRIGHT,
#     ],
#     'possible_bond_stereo_list': [
#         Chem.rdchem.BondStereo.STEREONONE,
#         Chem.rdchem.BondStereo.STEREOZ,
#         Chem.rdchem.BondStereo.STEREOE,
#         Chem.rdchem.BondStereo.STEREOCIS,
#         Chem.rdchem.BondStereo.STEREOTRANS,
#         Chem.rdchem.BondStereo.STEREOANY,
#     ],
#     'possible_is_conjugated_list': [False, True],
# }

# allowable multiple choice node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list': [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    "possible_bond_dirs": [  # only for double bond stereo information
        'NONE',
        'ENDUPRIGHT',
        'ENDDOWNRIGHT',
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def atom_to_feature_vector(atom, simple=False):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    if simple:
        atom_feature = [
        safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
        allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
    ]
    else:
        atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
        ]
    return atom_feature


def get_atom_feature_dims(simple=False):
    if simple:
        return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
    ]))
    else:
        return list(map(len, [
            allowable_features['possible_atomic_num_list'],
            allowable_features['possible_chirality_list'],
            allowable_features['possible_degree_list'],
            allowable_features['possible_formal_charge_list'],
            allowable_features['possible_numH_list'],
            allowable_features['possible_number_radical_e_list'],
            allowable_features['possible_hybridization_list'],
            allowable_features['possible_is_aromatic_list'],
            allowable_features['possible_is_in_ring_list']
        ]))


def bond_to_feature_vector(bond, simple=False):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    if simple:
        bond_feature = [
        safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
        safe_index(allowable_features['possible_bond_dirs'], str(bond.GetBondDir())),
    ]
    else:
        bond_feature = [
            safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
            allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
            allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
        ]
    return bond_feature


def get_bond_feature_dims(simple=False):
    if simple:
        return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_dirs']
    ]))
    else:
        return list(map(len, [
            allowable_features['possible_bond_type_list'],
            allowable_features['possible_bond_stereo_list'],
            allowable_features['possible_is_conjugated_list']
        ]))

def atom_feature_vector_to_dict(atom_feature, simple=False):
    if simple:
        [atomic_num_idx,
        chirality_idx,
        ] = atom_feature
        feature_dict = {
            'atomic_num': allowable_features['possible_atomic_num_list'][atomic_num_idx],
            'chirality': allowable_features['possible_chirality_list'][chirality_idx],
        }
    else:
        [atomic_num_idx,
        chirality_idx,
        degree_idx,
        formal_charge_idx,
        num_h_idx,
        number_radical_e_idx,
        hybridization_idx,
        is_aromatic_idx,
        is_in_ring_idx] = atom_feature

        feature_dict = {
            'atomic_num': allowable_features['possible_atomic_num_list'][atomic_num_idx],
            'chirality': allowable_features['possible_chirality_list'][chirality_idx],
            'degree': allowable_features['possible_degree_list'][degree_idx],
            'formal_charge': allowable_features['possible_formal_charge_list'][formal_charge_idx],
            'num_h': allowable_features['possible_numH_list'][num_h_idx],
            'num_rad_e': allowable_features['possible_number_radical_e_list'][number_radical_e_idx],
            'hybridization': allowable_features['possible_hybridization_list'][hybridization_idx],
            'is_aromatic': allowable_features['possible_is_aromatic_list'][is_aromatic_idx],
            'is_in_ring': allowable_features['possible_is_in_ring_list'][is_in_ring_idx]
        }

    return feature_dict


def bond_feature_vector_to_dict(bond_feature, simple=False):
    if simple:
        [bond_type_idx,
         bond_dir_idx] = bond_feature

        feature_dict = {
            'bond_type': allowable_features['possible_bond_type_list'][bond_type_idx],
            'bond_dir': allowable_features['possible_bond_dirs'][bond_dir_idx],
        }
    else:
        [bond_type_idx,
        bond_stereo_idx,
        is_conjugated_idx] = bond_feature

        feature_dict = {
            'bond_type': allowable_features['possible_bond_type_list'][bond_type_idx],
            'bond_stereo': allowable_features['possible_bond_stereo_list'][bond_stereo_idx],
            'is_conjugated': allowable_features['possible_is_conjugated_list'][is_conjugated_idx]
        }

    return feature_dict