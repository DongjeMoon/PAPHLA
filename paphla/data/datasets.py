import argparse
import json
import os
import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter, to_dense_adj
from torch_geometric.utils.num_nodes import maybe_num_nodes
from tqdm import tqdm

from paphla.data.features import *


class PeptideData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'res_idx':
            return self.num_residues
        else:
            return super().__inc__(key, value, *args, **kwargs)


class MemMapDataset(Dataset):
    """
    Memory-mapped dataset for large PyG graphs.
    Data is stored as concatenated arrays with boundaries marking each sample.
    """
    
    def __init__(self, path="data/train/dataset.csv", root=""):
        script_dir = Path(__file__).resolve().parent.parent

        if root == "":
            root = script_dir
        
        self.path = osp.join(root, path)
        self.name = path.split("/")[-1].replace(".csv", "")
        self.processed_dir = osp.join(root, "data", "processed", self.name)
        
        # initialization
        df = pd.read_csv(self.path)
        cols = ["peptide","hla","label","ptm","ptm_location"]
        self.__dict__.update({ col: df[col].to_numpy() for col in cols })

        # Load HLA embeddings
        hla_index_path = osp.join(root, 'data/esm/hla_esm_index.json')
        with open(hla_index_path, 'r') as f:
            index_data = json.load(f)
        hla_to_idx = index_data.get('hla_to_idx', index_data)
        self.hla_idx = np.array([hla_to_idx.get(hla, -1) for hla in self.hla], 
                                 dtype=np.int16)
        missing_mask = self.hla_idx == -1
        num_missing = missing_mask.sum()
        if num_missing > 0:
            missing_hlas = np.unique(np.array(self.hla)[missing_mask])
            raise ValueError(f"Missing ESM2 for: {missing_hlas}")

        # PTM groups
        self.ptm_flags = np.array([get_ptm_features(str(p)) for p in self.ptm], 
                                  dtype=np.int16)
        self.n_groups = sum(get_feature_dims("ptm"))
        counts = torch.zeros(self.n_groups, dtype=torch.long)
        try:
            for s in self.ptm:
                counts[get_ptm_features(str(s))] += 1
        except:
            pass
        self._counts = counts

        # Get dimensions
        self.node_dim = len(get_feature_dims('atom'))
        self.edge_dim = len(get_feature_dims('bond'))
        self.max_len  = max(len(pep) for pep in self.peptide)
        self.rwse_dim = len(list(range(1, 16 + 1)))

        if osp.exists(self.processed_dir) and os.listdir(self.processed_dir):
            self._load_memmaps()
        else:
            print(f"Processing {osp.relpath(self.path)}...")
            self._process()
            self._load_memmaps()

    def peptide2graph(self, idx):
        """Convert SMILES to PyG Data object"""
        smiles = self.smiles_list[idx]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES {smiles}")

        # Extract molecular features
        node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        num_nodes = len(node_features)
        
        src_nodes, dst_nodes, edge_features = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feat = get_bond_features(bond)
            src_nodes.extend([i, j])
            dst_nodes.extend([j, i])
            edge_features.extend([bond_feat, bond_feat])
        
        if not src_nodes:
            src_nodes, dst_nodes, edge_features = [0], [0], [[0, 0, 0]]
        
        # Create PyG Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.long),
            edge_index=torch.tensor([src_nodes, dst_nodes], dtype=torch.long),
            edge_attr=torch.tensor(edge_features, dtype=torch.long),
            num_nodes=num_nodes
        )
        
        # Compute PE
        rw_landing = get_rw_landing_probs(ksteps=list(range(1, 16 + 1)), # steps = 16
                                          edge_index=data.edge_index,
                                          num_nodes=num_nodes)
        data.pestat_RWSE = rw_landing
        data.ptm_flag = torch.as_tensor(get_ptm_features(str(self.ptm[idx])), dtype=torch.long)

        return data

    def _process(self):
        """Process dataset to memory-mapped format"""
        os.makedirs(self.processed_dir, exist_ok=True)
        
        total_nodes = total_edges = 0
        node_bounds = [0]
        edge_bounds = [0]
        valid_indices = []
        self.smiles_list = []
        self.res_idx_list = []

        for idx in tqdm(range(len(self.peptide)), desc="Processing data"):
            try:
                smiles, res_idx, num_nodes, num_edges = peptide_to_smiles(
                            self.peptide[idx], self.ptm[idx], self.ptm_location[idx])

                total_nodes += num_nodes
                total_edges += num_edges
                
                self.smiles_list.append(smiles)
                self.res_idx_list.append(res_idx)
                node_bounds.append(total_nodes)
                edge_bounds.append(total_edges)
                valid_indices.append(idx)

            except Exception as e:
                print(f"Skip {idx}: {e}")
        
        num_samples = len(valid_indices)
        print(f"Valid: {num_samples}, Nodes: {total_nodes}, Edges: {total_edges}")
        
        # Allocate memory-mapped arrays
        node_features = np.memmap(
            osp.join(self.processed_dir, 'node_features.npy'),
            dtype='int64', mode='w+', shape=(total_nodes, self.node_dim)
        )
        edge_indices = np.memmap(
            osp.join(self.processed_dir, 'edge_indices.npy'),
            dtype='int64', mode='w+', shape=(total_edges, 2)
        )
        edge_attrs = np.memmap(
            osp.join(self.processed_dir, 'edge_attrs.npy'),
            dtype='int64', mode='w+', shape=(total_edges, self.edge_dim)
        )
        res_indices = np.memmap(
            osp.join(self.processed_dir, 'res_indices.npy'),
            dtype='int64', mode='w+', shape=(total_nodes,)
        )
        rwse_features = np.memmap(
            osp.join(self.processed_dir, 'rwse_features.npy'),
            dtype='float32', mode='w+', shape=(total_nodes, self.rwse_dim)
        )

        res_indices[:] = np.concatenate(self.res_idx_list)

        for i, idx in enumerate(tqdm(valid_indices, desc="Writing data")):
            try:
                data = self.peptide2graph(idx)
                
                ns, ne = node_bounds[i], node_bounds[i + 1]
                es, ee = edge_bounds[i], edge_bounds[i + 1]
                
                node_features[ns:ne] = data.x.numpy()
                edge_indices[es:ee] = data.edge_index.t().numpy()
                edge_attrs[es:ee] = data.edge_attr.numpy()
                rwse_features[ns:ne] = data.pestat_RWSE.numpy().astype('float32')
            except Exception as e:
                print(f"Error {idx}: {e}")
        
        # Flush
        node_features.flush()
        edge_indices.flush()
        edge_attrs.flush()
        rwse_features.flush()
        res_indices.flush()

        # Save
        np.save(osp.join(self.processed_dir, 'node_boundaries.npy'), node_bounds)
        np.save(osp.join(self.processed_dir, 'edge_boundaries.npy'), edge_bounds)
        
        print(f"Processed {num_samples:,} samples ({total_nodes:,} nodes)")

    def _load_memmaps(self):
        """Load all memory-mapped arrays and metadata"""
        
        self.node_boundaries = np.load(osp.join(self.processed_dir, 'node_boundaries.npy'))
        self.edge_boundaries = np.load(osp.join(self.processed_dir, 'edge_boundaries.npy'))
        
        total_nodes = self.node_boundaries[-1]
        total_edges = self.edge_boundaries[-1]
        
        # Load memmap arrays
        self.node_features = np.memmap(
            osp.join(self.processed_dir, 'node_features.npy'),
            dtype='int64', mode='r', shape=(total_nodes, self.node_dim)
        )
        self.edge_indices = np.memmap(
            osp.join(self.processed_dir, 'edge_indices.npy'),
            dtype='int64', mode='r', shape=(total_edges, 2)
        )
        self.edge_attrs = np.memmap(
            osp.join(self.processed_dir, 'edge_attrs.npy'),
            dtype='int64', mode='r', shape=(total_edges, self.edge_dim)
        )
        self.res_indices = np.memmap(
            osp.join(self.processed_dir, 'res_indices.npy'),
            dtype='int64', mode='r', shape=(total_nodes,)
        )
        self.rwse_features = np.memmap(
            osp.join(self.processed_dir, 'rwse_features.npy'),
            dtype='float32', mode='r', shape=(total_nodes, self.rwse_dim)
        )

        self.num_samples = len(self.label)

        # print(f"Loaded {self.num_samples} samples from {osp.relpath(self.processed_dir)}")
    
    def __len__(self):
        return self.num_samples

    def n_groups(self):
        return self.n_groups

    def group_counts(self):
        return self._counts

    def group_str(self, idx):
        return self.ptm[idx]
    
    def __getitem__(self, idx):
        """Zero-copy slicing from memory-mapped arrays"""
        if idx >= self.num_samples or idx < 0:
            raise IndexError(f"Index {idx} out of range")
        
        node_start, node_end = self.node_boundaries[idx], self.node_boundaries[idx + 1]
        edge_start, edge_end = self.edge_boundaries[idx], self.edge_boundaries[idx + 1]
        
        x = torch.from_numpy(self.node_features[node_start:node_end])
        edge_index = torch.from_numpy(self.edge_indices[edge_start:edge_end])
        edge_attr = torch.from_numpy(self.edge_attrs[edge_start:edge_end])
        res_idx = torch.from_numpy(self.res_indices[node_start:node_end])
        
        data = PeptideData(
            x=x,
            edge_index=edge_index.t().contiguous(),  # [2, num_edges]
            edge_attr=edge_attr,
            num_nodes=node_end - node_start
        )
        
        data.res_idx = res_idx
        data.pestat_RWSE = torch.from_numpy(self.rwse_features[node_start:node_end])
        data.hla_idx = torch.tensor(self.hla_idx[idx], dtype=torch.long)
        data.y = torch.tensor(self.label[idx], dtype=torch.long)
        data.num_residues = res_idx.max().item() + 1
        data.ptm_flag = torch.tensor(self.ptm_flags[idx], dtype=torch.long)
        
        return data


class CustomCollate:
    def __init__(self, hla_path='data/esm/hla_esm.mmap',
                 hla_index_path='data/esm/hla_esm_index.json', root=""):
        script_dir = Path(__file__).resolve().parent.parent

        if root == "":
            root = script_dir

        hla_path = osp.join(root, hla_path)
        hla_index_path = osp.join(root, hla_index_path)
        
        with open(hla_index_path, 'r') as f:
            index_data = json.load(f)
        
        shape = tuple(index_data['shape'])
        dtype = np.dtype(index_data['dtype'])
            
        hla_memmap = np.memmap(hla_path, mode='r', dtype=dtype, shape=shape)
        self.hla_embeddings = torch.from_numpy(hla_memmap[:]).float()
    
    def __call__(self, data_list):
        hla_indices = torch.tensor([data.hla_idx for data in data_list], dtype=torch.long)
        batch = Batch.from_data_list(data_list)
        batch.hla = self.hla_embeddings[hla_indices]

        return batch

def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.
    Code adapted from: https://github.com/rampasek/GraphGPS/blob/main/graphgps/transform/posenc_stats.py (MIT license)

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    edge_index = torch.as_tensor(edge_index)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing


def main(): 
    parser = argparse.ArgumentParser(description='Process dataset')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the data file')
    args = parser.parse_args()
    if not os.path.isabs(args.data):
        data_path = os.path.abspath(args.data)
    else:
        data_path = args.data
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    dataset = MemMapDataset(data_path, root="")

if __name__ == "__main__":
    main()