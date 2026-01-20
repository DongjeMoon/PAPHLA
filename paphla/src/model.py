import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch

from paphla.src.attention import PMA, CrossAttention
from paphla.src.encoders import PeptideEncoder
from paphla.src.rdb_gps import RDBGPSLayer


class PAPHLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_heads = cfg.num_heads
        self.pool = cfg.pool

        # Peptide
        self.pep_encoder = PeptideEncoder(cfg.dim_h, cfg.dim_pe, cfg.steps)
        self.RDBGPS = nn.ModuleList([
            RDBGPSLayer(cfg.dim_h, cfg.num_heads, cfg.dropout, cfg.attn_dropout, cfg.use_rdb)
            for _ in range(cfg.n_gps_layers)
        ])
        self.pep_proj = nn.Sequential(
            nn.LayerNorm(cfg.dim_h),
            nn.Linear(cfg.dim_h, cfg.dim_h * 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout)
        )

        # HLA
        self.hla_proj = nn.Sequential(
            nn.LayerNorm(cfg.dim_esm),
            nn.Linear(cfg.dim_esm, cfg.dim_h * 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

        # cross-modal alignment
        self.mutual_proj = nn.Linear(cfg.dim_h * 2, cfg.dim_h)
        self.cross_attn = nn.ModuleList([
            CrossAttention(
                cfg.dim_h, cfg.num_heads, 
                cfg.attn_dropout, cfg.dropout)
                for _ in range(cfg.n_crs_layers)
        ])

        # pooling
        if self.pool == "PMA":
            self.pma = PMA(cfg.dim_h, cfg.num_heads, cfg.num_seeds)
        
        # classification head
        self.head = nn.Sequential(
            nn.Linear(cfg.num_seeds * cfg.dim_h, cfg.dim_h),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_h, cfg.dim_h // 2),
            nn.ReLU(),
            nn.Linear(cfg.dim_h // 2, 1)
        )

    def forward(self, batch):
        # peptide
        batch = self.pep_encoder(batch)
        
        for layer in self.RDBGPS:
            batch = layer(batch)

        # cross-modal alignment
        pep_proj = self.mutual_proj(self.pep_proj(batch.x))
        hla_proj = self.mutual_proj(self.hla_proj(batch.hla))

        batch.x   = pep_proj
        batch.hla = hla_proj

        attns = []
        for ca in self.cross_attn:
            batch, crs_attn = ca(batch)
            attns.append(crs_attn)

        if self.pool == "PMA":
            phla, pma_attn = self.pma(batch)
            attns.append(pma_attn)
        elif self.pool == "mean":        
            phla = global_mean_pool(batch.x, batch.batch)
        else:
            raise ValueError("Either mean or PMA accepted for pooling")
        
        logit = self.head(phla)
        logit = logit.view(-1)

        if self.training:
            mask = batch.ptm_flag == 0
            canonical_loss = F.binary_cross_entropy_with_logits(
                logit[mask],
                batch.y[mask].float()
            ) if mask.any() else logit.new_tensor(0.0)

            return logit, canonical_loss
        else:
            return logit, attns

    @torch.no_grad()
    def predict(self, batch, return_logit=False):
        self.eval()
        logit, attns = self.forward(batch)
        pred = torch.sigmoid(logit)
        # attns = torch.stack(attns, dim=0)

        if return_logit:
            return logit, attns
        return pred, attns