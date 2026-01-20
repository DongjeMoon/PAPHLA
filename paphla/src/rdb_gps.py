import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter


class GatedGCNLayer(pyg_nn.conv.MessagePassing):
    """GatedGCN layer.
    Code adapted from: https://github.com/rampasek/GraphGPS/blob/main/graphgps/layer/gatedgcn_layer.py (MIT license)
        
    Residual Gated Graph ConvNets
    https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, in_dim, out_dim, dropout, residual, **kwargs):
        super().__init__(**kwargs)
        self.A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.B = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.C = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.E = pyg_nn.Linear(in_dim, out_dim, bias=True)

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.act_fn_x = nn.ReLU()
        self.act_fn_e = nn.ReLU()
        self.dropout = dropout
        self.residual = residual
        self.e = None

    def forward(self, batch):
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index

        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        if self.residual:
            x_in = x
            e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        x, e = self.propagate(edge_index,
                              Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce,
                              e=e, Ax=Ax)

        x = self.bn_node_x(x)
        e = self.bn_edge_e(e)

        x = self.act_fn_x(x)
        e = self.act_fn_e(e)

        x = F.dropout(x, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        if self.residual:
            x = x_in + x
            e = e_in + e

        batch.x = x
        batch.edge_attr = e

        return batch

    def message(self, Dx_i, Ex_j, Ce):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)

        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size,
                                   reduce='sum')

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size,
                                     reduce='sum')

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out


class RDBGPSLayer(nn.Module):
    """GatedGCN + residue-distance-biased attention layer.
    Code adapted from: https://github.com/rampasek/GraphGPS/blob/main/graphgps/layer/gps_layer.py (MIT license)
    """

    def __init__(self, dim_h, num_heads, dropout=0.0, attn_dropout=0.0, use_rdb=True):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.use_rdb = use_rdb

        # Message-passing model.
        self.local_model = GatedGCNLayer(dim_h, dim_h,
                                         dropout=dropout,
                                         residual=True
                                        )

        # Residue-distance-biased attention.
        self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)

        self.norm1_local = pyg_nn.norm.LayerNorm(dim_h)
        self.norm1_attn = pyg_nn.norm.LayerNorm(dim_h)
        self.norm2 = pyg_nn.norm.LayerNorm(dim_h)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_block = nn.Sequential(
            nn.Linear(dim_h, dim_h * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_h * 2, dim_h),
            nn.Dropout(dropout)
        )

        # ICLR 2022: Train short, test long:
        # attention with linear biases enables
        # input length extrapolation
        # https://openreview.net/forum?id=R8sQPpGCv0
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2**math.floor(math.log2(n))
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

        if use_rdb:
            self.register_buffer('slopes', torch.Tensor(get_slopes(num_heads)))                

    def compute_residue_bias(self, batch):
        """
        Compute ALiBi bias based on residue distances
        
        Args:
            res_idx: [num_atoms] residue index for each atom
            batch_idx: [num_atoms] batch assignment for each atom
            mask: [batch_size, max_atoms] padding mask from to_dense_batch
        
        Returns:
            rdb: [batch_size, num_heads, max_atoms, max_atoms]
        """
        # Convert to dense batch
        res_idx_dense, _ = to_dense_batch(batch.res_idx, batch.batch, fill_value=-1)

        # Compute pairwise residue distance matrix
        # res_diff[b, i, j] = |res_idx[b,i] - res_idx[b,j]|
        res_i = res_idx_dense.unsqueeze(2)  # [B, max_atoms, 1]
        res_j = res_idx_dense.unsqueeze(1)  # [B, 1, max_atoms]
        res_diff = (res_i - res_j).abs().to(dtype=torch.float32)     # [B, max_atoms, max_atoms]
        
        # bias = -slope * distance
        slopes = self.slopes.view(1, self.num_heads, 1, 1)
        rdb = -slopes * res_diff.unsqueeze(1)  # [B, num_heads, max_atoms, max_atoms]
        
        return rdb

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        local_out = self.local_model(Batch(batch=batch,
                                           x=h,
                                           edge_index=batch.edge_index,
                                           edge_attr=batch.edge_attr))
        # GatedGCN does residual connection and dropout internally.
        h_local = local_out.x
        batch.edge_attr = local_out.edge_attr
        h_local = self.norm1_local(h_local, batch.batch)

        h_out_list.append(h_local)

        # global attention.
        h_dense, mask = to_dense_batch(h, batch.batch)
        if self.use_rdb:
            B, N, _ = h_dense.shape
            alibi = self.compute_residue_bias(batch)
            alibi = alibi.view(B * self.num_heads, N, N)
            h_attn = self.self_attn(h_dense, h_dense, h_dense,
                                    key_padding_mask=~mask, attn_mask=alibi)[0][mask]
        else:
            h_attn = self.self_attn(h_dense, h_dense, h_dense,
                                    key_padding_mask=~mask)[0][mask]

        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in1 + h_attn  # Residual connection.
        h_attn = self.norm1_attn(h_attn, batch.batch)
        h_out_list.append(h_attn)

        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self.ff_block(h)
        h = self.norm2(h, batch.batch)

        batch.x = h
        return batch