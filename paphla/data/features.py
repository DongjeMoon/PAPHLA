import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem

FEATURES = {
    'atom': {
        'symbol': ['C', 'N', 'O', 'S', 'P', 'Se', 'I'],                                               # 7 
        'aromaticity': [False, True],                                                                 # 2
        'ring': [False, True],                                                                        # 2
        'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],                                      # 11
        'total_degree': [0, 1, 2, 3, 4, 5, 6, 'OTHER'],                                               # 8
        'total_numHs': [0, 1, 2, 3, 4, 5, 6, 'OTHER'],                                                # 8
        'hybridization': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER'],                              # 6
        'chirality': [
            'CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW',
            'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'
        ]    # 4
    },
                                                                                                      ## 48
    'bond': {
        'type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],                                           # 4
        'conjugated': [False, True],                                                                  # 2
        'stereo': [
            'STEREONONE', 'STEREOZ', 'STEREOE',
            'STEREOCIS', 'STEREOTRANS', 'STEREOANY'
        ]                                                                                                     # 6
    },
                                                                                                      ## 12
    # 'residue': {
    #     'symbol': [
    #         'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
    #         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
    #     ],
    #     'type': [
    #         "logp", "total_charge", "mol_weight", "tpsa",
    #          "num_h_donors", "num_h_acceptors", "num_rot_bonds"
    #     ]
    # },

    'ptm': {
        'type': ['0', 'METH', 'SULF', 'FORM', 'DIMETH', 'PHOS', 'NAc', 'OX', 'HYL',
                'PYRE', 'CITR', 'CBX', 'SUCC', 'CRTNL', 'PALM', 'BIOT', 'ACET',
                'ADPR', 'AMID', 'MAL', "OTHER"
        ]
    }
}


def safe_index(l, e):
    try:
        return l.index(e)
    except:
        return len(l) - 1


def get_atom_features(atom):
    atom_feature_dict = FEATURES['atom']
    atom_feature = [
        safe_index(atom_feature_dict['symbol'], atom.GetSymbol()),
        atom_feature_dict['aromaticity'].index(atom.GetIsAromatic()),
        atom_feature_dict['ring'].index(atom.IsInRing()),
        atom_feature_dict['formal_charge'].index(atom.GetFormalCharge(),),
        safe_index(atom_feature_dict['total_degree'], atom.GetTotalDegree()),
        safe_index(atom_feature_dict['total_numHs'], atom.GetTotalNumHs()),
        safe_index(atom_feature_dict['hybridization'], str(atom.GetHybridization())),
        safe_index(atom_feature_dict['chirality'], str(atom.GetChiralTag()))
        ]
    return atom_feature


def get_bond_features(bond):
    bond_feature_dict = FEATURES['bond']
    bond_feature = [
        bond_feature_dict['type'].index(str(bond.GetBondType())),
        bond_feature_dict['conjugated'].index(bond.GetIsConjugated()),
        bond_feature_dict['stereo'].index(str(bond.GetStereo()))
    ]
    return bond_feature


def get_ptm_features(ptm):
    ptm_feature_dict = FEATURES['ptm']
    return safe_index(ptm_feature_dict['type'], ptm)


def get_feature_dims(type):
    if type == 'atom':
        atom_feature_dict = FEATURES['atom']
        return [len(atom_feature_dict[key]) for key in atom_feature_dict.keys()]
    elif type == 'bond':
        bond_feature_dict = FEATURES['bond']
        return [len(bond_feature_dict[key]) for key in bond_feature_dict.keys()]
    elif type == 'ptm':
        ptm_feature_dict = FEATURES['ptm']
        return [len(ptm_feature_dict['type'])]


DEFAULT_PTM_JSON = Path(__file__).parent / "aa_to_smiles.json"


def _load_aa_to_smiles(path=DEFAULT_PTM_JSON):
    with open(path, "r") as f:
        return json.load(f)

_AA_TO_SMILES = _load_aa_to_smiles()

@lru_cache(maxsize=200_000)
def _num_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveHs(mol)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles}")
    return mol.GetNumAtoms()

def build_fragment_table(aa_to_smiles: dict) -> dict:
    """(aa, ptm, is_terminal) -> (smiles, n_atoms)"""
    table = {}
    for aa, ptm_dict in aa_to_smiles.items():
        for ptm, base_smiles in ptm_dict.items():
            for is_terminal in (False, True):
                s = base_smiles if is_terminal else base_smiles.rstrip("O")
                table[(aa, ptm, is_terminal)] = (s, _num_atoms(s))
    return table

_FRAG = build_fragment_table(_AA_TO_SMILES)


def parse_ptm_loc(ptm_location):
    if str(ptm_location) == '0':
        return []
    return [int(s.strip()[1:]) for s in ptm_location.split(',')]


def peptide_to_smiles(sequence: str, ptm: str, ptm_location):
    ptm_pos = parse_ptm_loc(ptm_location)
    L = len(sequence)

    smiles_parts = []
    atom_counts = []

    for pos, aa in enumerate(sequence, start=1):
        is_terminal = (pos == L)

        use_ptm = (pos in ptm_pos) and (ptm in _AA_TO_SMILES.get(aa, {}))
        ptm_key = ptm if use_ptm else "0"

        frag_smiles, n_atoms = _FRAG[(aa, ptm_key, is_terminal)]
        smiles_parts.append(frag_smiles)
        atom_counts.append(n_atoms)

    peptide_smiles = "".join(smiles_parts)
    mol = Chem.MolFromSmiles(peptide_smiles)
    mol = Chem.RemoveHs(mol)
    if mol is None:
        raise ValueError(f"Bad peptide SMILES: {peptide_smiles}")

    res_idx = np.repeat(np.arange(L, dtype=np.int64), atom_counts)

    # Optional sanity check:
    assert res_idx.size == mol.GetNumAtoms()

    return peptide_smiles, res_idx, mol.GetNumAtoms(), mol.GetNumBonds() * 2