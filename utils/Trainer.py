"""Legacy trainer API for models without inter-residue features."""

from dataclasses import dataclass
from typing import Tuple

import torch

from utils.Trainer_with_inter_edge_feat import (
    Trainer as _Trainer, TrainConfig as _TrainConfig,
    CSVLogger, MetricTracker, remove_padded_rows_cols, print_metrics_table,
)


@dataclass
class TrainConfig(_TrainConfig):
    topk_list: Tuple[int, ...] = (1, 3, 5, 10, 20)


class Trainer(_Trainer):
    def read_batch(self, data):
        batch = super().read_batch(data)
        return batch[:-5] + batch[-4:]

    def forward_logits(self, batch) -> torch.Tensor:
        (
            pdbname,
            nuc_x, nuc_evo_x, nuc_coord_x, nuc_edge_index, nuc_edge_weight, RNA_local_R, RNA_local_t, RNA_pred_site,
            residue_x, residue_evo_x, residue_coord_x, residue_edge_index, residue_edge_weight, prot_local_R, prot_local_t, prot_pred_site, protein_domain_mask, protein_domain_lbl,
            RNA_batch, prot_batch,
            _, _, _, _
        ) = batch
        # print(f"Processing {pdbname} with RNA size {nuc_x.shape[0]} and Protein size {residue_x.shape[0]}")
        final_logits = self.model(
            nuc_x, nuc_evo_x, nuc_coord_x, nuc_edge_index, nuc_edge_weight, RNA_local_R, RNA_local_t, RNA_pred_site,
            residue_x, residue_evo_x, residue_coord_x, residue_edge_index, residue_edge_weight, prot_local_R, prot_local_t, prot_pred_site, protein_domain_mask, 
            RNA_batch, prot_batch
        )  # [B, h, H, W]

        return final_logits
