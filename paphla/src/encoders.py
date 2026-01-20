import torch
import torch.nn as nn

from paphla.data.features import get_feature_dims


class Encoder(nn.Module):
    """Encoder for the input features.

    Args:
        emb_dim: Size of the embedding
        type: Type of the feature

    Returns:
        x_embedding: Embedding of the input features
    """
    def __init__(self, emb_dim, type='atom'):
        super().__init__()

        self.type = type
        self.embedding_list = nn.ModuleList()
        
        if type == 'atom':
            full_dims = get_feature_dims(type)
        elif type == 'bond':
            full_dims = get_feature_dims(type)
        else:
            raise ValueError("Invalid type. Choose 'atom' or 'bond'.")

        for dim in full_dims:
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.embedding_list.append(emb)

    def forward(self, x):
        x_embedding = torch.zeros_like(self.embedding_list[0](x[:, 0]))
        for i in range(x.shape[1]):
            x_embedding += self.embedding_list[i](x[:, i])

        return x_embedding

class RWSENodeEncoder(nn.Module):
    """Random Walk Structural Encoding node encoder.
    Code adapted from: https://github.com/rampasek/GraphGPS/blob/main/graphgps/encoder/kernel_pos_encoder.py (MIT License)

    Args:
        dim_in: Original input node features dimension
        dim_emb: Size of final node embedding
        dim_pe: Size of the positional embedding
        num_rw_steps: Number of random walk steps
    """

    def __init__(self, dim_in, dim_emb, dim_pe, num_rw_steps):
        super().__init__()
        
        if dim_emb - dim_pe < 0:
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        self.pe_encoder = nn.Sequential(
            nn.Linear(num_rw_steps, 2 * dim_pe),
            nn.ReLU(),
            nn.Linear(2 * dim_pe, dim_pe),
            nn.ReLU()
        )

    def forward(self, batch):
        if not hasattr(batch, 'pestat_RWSE'):
            raise ValueError("Precomputed 'pestat_RWSE' variable is required")

        pos_enc = batch.pestat_RWSE  # (Num nodes) x (Num kernel times)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        h = batch.x
        batch.x = torch.cat((h, pos_enc), 1)
        return batch


class PeptideEncoder(nn.Module):
    """Peptide graph encoder.

    Args:
        emb_dim: Size of final node embedding
        dim_pe: Size of the positional embedding
    """
    def __init__(self, dim_emb, dim_pe, steps):
        super().__init__()

        if dim_emb - dim_pe < 0:
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        # atoms and bonds
        dim_h = dim_emb - dim_pe
        self.encoder_h = Encoder(dim_h, type='atom')
        self.encoder_e = Encoder(dim_emb, type='bond')

        # PE
        self.encoder_hpe = RWSENodeEncoder(dim_h, dim_emb, dim_pe, num_rw_steps=steps)

    def forward(self, batch):
        batch.x = self.encoder_h(batch.x)
        batch.edge_attr = self.encoder_e(batch.edge_attr)
        batch = self.encoder_hpe(batch)

        return batch