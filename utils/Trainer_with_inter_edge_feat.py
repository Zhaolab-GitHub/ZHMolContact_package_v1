import os
import sys
import time
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from utils.metrics import Animator
from utils.contact_metrics import precision_topk
from utils.Predictor_inter import read_batch


# ========================================================
# Helpers
# ========================================================
def remove_padded_rows_cols(matrix: torch.Tensor, pad_value: int = -100) -> torch.Tensor:
    """Remove rows/cols that are entirely pad_value."""
    non_padded_rows = torch.where(torch.any(matrix != pad_value, dim=1))[0]
    non_padded_cols = torch.where(torch.any(matrix != pad_value, dim=0))[0]
    return matrix[non_padded_rows][:, non_padded_cols]


@dataclass
class TrainConfig:
    lr: float = 1e-5
    epochs: int = 20
    eval_every: int = 1
    batch_size: int = 16

    optim_name: str = "AdamW"      # AdamW | Adam | SGD
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0

    num_classes: int = 37
    pad_value: int = -100

    # K values for precision@K.
    topk_list: Tuple[int, ...] = (1, 3, 5, 10)

    # For 37 distance classes, classes 0..15 indicate contacts within 10 angstroms.
    contact_class_cutoff: int = 16

    save_all_epochs: bool = True
    save_best: bool = True                # Enable best-model saving when desired.
    best_key: str = "valid_Top10_pre"      # Metric used to select the best model.


# ========================================================
# Logging and output helpers
# ========================================================

class CSVLogger:
    'Persist training metrics in a local CSV file.'
    def __init__(self, path: str, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        # Create the directory if needed.
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        # Create the file and write its header.
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, row: Dict):
        'Append known fields only, preventing extra keys from changing the CSV structure.'
        clean_row = {k: row.get(k, None) for k in self.fieldnames}
        # Round numeric values to four decimal places.
        for k, v in clean_row.items():
            if isinstance(v, (float, torch.Tensor)):
                clean_row[k] = round(float(v), 4)
                
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(clean_row)


class MetricTracker:
    def __init__(self):
        self.sums: Dict[str, float] = {}
        self.count: int = 0

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v) * n
        self.count += n

    def average(self) -> Dict[str, float]:
        if self.count == 0:
            return {k: 0.0 for k in self.sums}
        return {k: v / self.count for k, v in self.sums.items()}


# ========================================================
# Trainer Class
# ========================================================

class Trainer:
    def __init__(self, model: nn.Module, train_loader, device: torch.device, cfg: TrainConfig):
        
        # ========================= 1) Initialize the model, optimizer, loss and output paths. ======================== #
        
        # model and cfg
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg

        # optimizer
        if cfg.optim_name == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=cfg.lr, betas=(0.9, 0.999),
                eps=1e-8, weight_decay=cfg.weight_decay
            )
        elif cfg.optim_name == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=cfg.lr, betas=(0.9, 0.999),
                eps=1e-8, weight_decay=cfg.weight_decay
            )
        elif cfg.optim_name == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        else:
            raise ValueError(f"Unknown optim_name: {cfg.optim_name}")

        total_steps = int(len(train_loader) * cfg.epochs)
        if total_steps <= 0 or cfg.eval_every <= 0:
            raise ValueError('Training requires a nonempty loader, positive epochs and eval_every')
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=0.0)

        # loss
        label_weight = torch.tensor([0.1] * cfg.num_classes, device=device)
        if cfg.num_classes == 37:
            label_weight[:cfg.contact_class_cutoff] = 10
        else:
            label_weight[:1] = 10
        self.criterion = nn.CrossEntropyLoss(weight=label_weight, ignore_index=cfg.pad_value)

        # save dirs
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = f"Result/Train_{timestamp}_{cfg.lr}_{cfg.epochs}_{cfg.batch_size}"
        self.model_dir = os.path.join(self.run_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)


        # ========================= 2) Logger instance ======================== #
        res_keys = [f"Res_Top{k}" for k in cfg.topk_list]
        res_keys.append('Res_TopL/2')
        res_keys.append('Res_TopL')
        # dom_keys = [f"Dom_Top{k}" for k in cfg.topk_list]

        # 2. Assemble column headers.
        # Log loss for every stage.
        fieldnames = ["epoch", "train_loss", "valid_loss", "test_loss"]

        # Log precision for validation and test stages.
        for stage in ["valid", "test"]:
            # Residue metrics
            for rk in res_keys:
                fieldnames.append(f"{stage}_{rk}")
            # Domain metrics
            # for dk in dom_keys:
            #     fieldnames.append(f"{stage}_{dk}")

        # 3. Create the logger.
        self.logger = CSVLogger(os.path.join(self.run_dir, "log.csv"), fieldnames=fieldnames)

        # optional: curves
        self.animator_loss = Animator(xlabel="epoch", xlim=[1, cfg.epochs], legend=["train loss"])

        # best tracking
        self.best_score = -1e9

    # ========================================================
    # Main loop for train
    # ========================================================
    def train(self, train_loader, valid_loader=None, test_loader=None):
        for epoch in tqdm(range(1, self.cfg.epochs + 1), desc="epoch"):
            # --- 1. Training ---
            train_metrics = self.train_one_epoch(train_loader)
            self.animator_loss.add(epoch, [train_metrics["loss"]])

            valid_m = {}
            test_m = {}

            # --- 2. Evaluation ---
            if epoch % self.cfg.eval_every == 0:
                if valid_loader is not None:
                    valid_m = self.evaluate_one_epoch(valid_loader)
                if test_loader is not None:
                    test_m = self.evaluate_one_epoch(test_loader)

                print_metrics_table(train_metrics, valid_m, test_m)

                # --- 3. Write logs ---
                row = {
                    "epoch": epoch,
                    "train_loss": train_metrics.get("loss", 0.0),
                    "valid_loss": valid_m.get("loss", 0.0),
                    "test_loss": test_m.get("loss", 0.0),
                }

                for stage, m_dict in [("valid", valid_m), ("test", test_m)]:
                    for k in self.cfg.topk_list:
                        row[f"{stage}_Res_Top{k}"] = m_dict.get(f"Res_Top{k}", 0.0)
                        row[f"{stage}_Dom_Top{k}"] = m_dict.get(f"Dom_Top{k}", 0.0)
                    row[f"{stage}_Res_TopL/2"] = m_dict.get(f"Res_TopL/2", 0.0)
                    row[f"{stage}_Res_TopL"] = m_dict.get(f"Res_TopL", 0.0)

                self.logger.log(row)

                # --- 4. Save model ---
                if self.cfg.save_best and valid_loader is not None:
                    # Use the configured validation metric to select the best model.
                    metric_key = self.cfg.best_key.removeprefix('valid_')
                    if metric_key == 'Top10_pre':
                        metric_key = 'Res_Top10'
                    if metric_key not in valid_m:
                        raise KeyError(f'Unknown best-model metric: {self.cfg.best_key}')
                    current_score = valid_m[metric_key]
                    if metric_key == 'loss':
                        current_score = -current_score
                    if current_score > self.best_score:
                        self.best_score = current_score
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(self.model_dir, "best.pth")
                        )

            if self.cfg.save_all_epochs:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.model_dir, f"epoch_{epoch}.pth")
                )

    # ========================================================
    # each loop of each loop
    # ========================================================
    def train_one_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        tracker = MetricTracker()

        for data in tqdm(loader, desc="train", leave=False):
            # print(data.pdbname)  
            batch = self.read_batch(data)

            std_target = batch[-4]  # [B,H,W]
            # print(f'std_target shape: {std_target.shape}')
            lbl_target = batch[-3]  # [B,H,W]
            # print(f'lbl_target shape: {lbl_target.shape}')

            # domain_target = batch[-3]  # [B,H,W] 
            # print(f'domain_target shape: {domain_target.shape}')

            RNA_site_truth = batch[-2]  # [B, H]
            prot_site_truth = batch[-1] # [B, W]
            # print(f'RNA_site_truth shape: {RNA_site_truth.shape}, prot_site_truth shape: {prot_site_truth.shape}')

            # domain_logits = self.forward_logits(batch)
            final_logits = self.forward_logits(batch)
        
            loss = self.compute_loss(
                final_logits=final_logits, 
                std_target=std_target, 
                lbl_target=lbl_target,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()  # per-iteration
            self.optimizer.zero_grad(set_to_none=True)

            tracker.update({"loss": float(loss.item())}, n=1)

        return tracker.average()

    @torch.no_grad()
    def evaluate_one_epoch(self, loader) -> Dict[str, float]:
        self.model.eval()
        tracker = MetricTracker()

        for data in tqdm(loader, desc="eval", leave=False):
            batch = self.read_batch(data)
            std_target = batch[-4]  # [B,H,W]
            # print(f'std_target shape: {std_target.shape}')
            lbl_target = batch[-3]  # [B,H,W]
            # print(f'lbl_target shape: {lbl_target.shape}')

            # domain_target = batch[-3]  # [B,H,W] 
            # print(f'domain_target shape: {domain_target.shape}')

            RNA_site_truth = batch[-2]  # [B, H]
            prot_site_truth = batch[-1] # [B, W]
            # print(f'RNA_site_truth shape: {RNA_site_truth.shape}, prot_site_truth shape: {prot_site_truth.shape}')

            final_logits = self.forward_logits(batch)

            # Compute loss.
            loss = self.compute_loss(
                final_logits=final_logits, 
                std_target=std_target, 
                lbl_target=lbl_target,
            )


            # Prepare prediction scores.
            res_scores = self.logits_to_contact_score(final_logits) 

            B = res_scores.shape[0]
            res_prec_accum = {k: 0.0 for k in self.cfg.topk_list}
            res_prec_accum['L/2'] = 0.0
            res_prec_accum['L'] = 0.0

            for b in range(B):
                # Compute residue-level metrics.
                res_lbl = lbl_target[b]
                res_true_binary = self.label_to_contact_binary(res_lbl)
                
                res_p = self.compute_precision_topk(res_scores[b], res_true_binary, self.cfg.topk_list)
                for k, v in res_p.items():
                    res_prec_accum[k] += v
                

            # Aggregate metrics.
            batch_metrics = {"loss": float(loss.item())}
            for k in self.cfg.topk_list:
                batch_metrics[f"Res_Top{k}"] = res_prec_accum[k] / B
            

            batch_metrics[f"Res_TopL/2"] = res_prec_accum['L/2'] / B
            batch_metrics[f"Res_TopL"] = res_prec_accum['L'] / B

            tracker.update(batch_metrics, n=B)

        return tracker.average()

    # ========================================================
    # compute loss
    # ========================================================
    @torch.no_grad()
    def sample_pixels_for_loss(
            self, 
            lbl_target: torch.Tensor,
            logits: torch.Tensor = None,
            pad_value: int = -100,
            contact_cutoff: int = 16,
            neg_pos_ratio: int = 95,
            min_neg: int = 1024,
            hard_neg_frac: float = 0.5,
            max_samples: int = 20000,
        ):
            device = lbl_target.device
            B, H, W = lbl_target.shape
            # Validate that logit and label spatial dimensions match.
            if logits is not None:
                assert logits.shape[2:] == (H, W), f"Shape Mismatch! Logits: {logits.shape} vs Label: {lbl_target.shape}"

            mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)

            valid = (lbl_target != pad_value)
            pos = valid & (lbl_target < contact_cutoff)
            neg = valid & (lbl_target >= contact_cutoff)

            for b in range(B):
                # Get flat indices for the current sample.
                # Flatten with view(-1) so indices refer to one-dimensional positions.
                pos_idx = torch.nonzero(pos[b].reshape(-1)).squeeze(-1)
                neg_idx = torch.nonzero(neg[b].reshape(-1)).squeeze(-1)

                P, N = pos_idx.numel(), neg_idx.numel()
                
                # Keep positives when no negatives are available.
                if N == 0:
                    if P > 0:
                        mask[b].view(-1)[pos_idx] = True
                    continue

                # Determine the sample count.
                k_neg = min(N, max(min_neg, neg_pos_ratio * max(P, 1)))
                if max_samples:
                    k_neg = min(k_neg, max(0, max_samples - P))

                if k_neg > 0 and logits is not None and hard_neg_frac > 0:
                    # Score hard negatives by the summed contact-class probabilities.
                    probs = torch.softmax(logits[b], dim=0) # [37, H, W]
                    score = probs[:contact_cutoff].sum(dim=0).view(-1)
                    
                    # Gather negative-pair scores.
                    neg_scores = score[neg_idx]

                    k_hard = min(N, int(round(k_neg * hard_neg_frac)))
                    k_easy = max(0, k_neg - k_hard)

                    # 1. Select hard negatives with the highest contact scores.
                    _, hard_in_neg = neg_scores.topk(k_hard)
                    chosen_hard = neg_idx[hard_in_neg]

                    # 2. Sample easy negatives from the remaining candidates.
                    if k_easy > 0:
                        # Exclude hard negatives with a boolean mask.
                        is_hard = torch.zeros(N, dtype=torch.bool, device=device)
                        is_hard[hard_in_neg] = True
                        easy_indices = neg_idx[~is_hard]
                        
                        if easy_indices.numel() > 0:
                            perm = torch.randperm(easy_indices.numel(), device=device)
                            chosen_easy = easy_indices[perm[:min(k_easy, easy_indices.numel())]]
                        else:
                            chosen_easy = torch.empty(0, dtype=torch.long, device=device)
                    else:
                        chosen_easy = torch.empty(0, dtype=torch.long, device=device)
                    
                    chosen_neg = torch.cat([chosen_hard, chosen_easy])
                else:
                    # Sample negatives uniformly.
                    perm = torch.randperm(N, device=device)
                    chosen_neg = neg_idx[perm[:k_neg]]

                # Assign selected positions through the flattened mask.
                all_idx = torch.cat([pos_idx, chosen_neg])
                if all_idx.numel() > 0:
                    mask[b].view(-1)[all_idx] = True
                    
            return mask
    

    def compute_loss(self, final_logits, std_target, lbl_target, lambda_bin=0.5, pad=-100):
        """Compute sampled distance and binary-contact losses on valid pairs."""
        pad = self.cfg.pad_value if pad == -100 else pad
        if final_logits.shape[1] != self.cfg.num_classes:
            raise ValueError('Logit channels do not match configured num_classes')
        if std_target.shape != lbl_target.shape:
            raise ValueError('Strict contact targets and distance labels must have the same shape')
        cutoff = self.cfg.contact_class_cutoff if self.cfg.num_classes == 37 else 1
        sample_mask = self.sample_pixels_for_loss(
            lbl_target, logits=final_logits, pad_value=pad, contact_cutoff=cutoff,
            neg_pos_ratio=95, min_neg=1024, hard_neg_frac=0.5,
        )
        valid = sample_mask & (lbl_target != pad)
        if not valid.any():
            raise ValueError('Cannot compute loss without valid distance labels')
        selected_logits = final_logits.permute(0, 2, 3, 1)[valid]
        ce_loss = self.criterion(selected_logits, lbl_target[valid].long())
        # Evaluate softmax only for sampled pairs and reuse it for binary loss.
        p_contact = selected_logits.softmax(dim=-1)[:, :cutoff].sum(dim=-1)
        y = std_target[valid].float()
        if not ((y == 0) | (y == 1)).all():
            raise ValueError('Strict contact targets must contain binary values at sampled positions')
        pos = y.sum().clamp(min=1.0)
        neg = (1.0 - y).sum().clamp(min=1.0)
        weights = 1.0 + y * ((neg / pos).detach() - 1.0)
        bce_loss = F.binary_cross_entropy(p_contact.clamp(1e-6, 1 - 1e-6), y, weight=weights)
        return ce_loss + lambda_bin * bce_loss

    # ========================================================
    # Unpack the batch
    # ========================================================

    def read_batch(self, data):
        batch = read_batch(data, self.device)
        if batch[-4] is None or batch[-3] is None:
            raise ValueError('Training requires strict contact targets and distance labels')
        return batch

    # ========================================================
    # forward
    # ========================================================
    def forward_logits(self, batch) -> torch.Tensor:
        (
            pdbname,
            nuc_x, nuc_evo_x, nuc_coord_x, nuc_edge_index, nuc_edge_weight, RNA_local_R, RNA_local_t, RNA_pred_site,
            residue_x, residue_evo_x, residue_coord_x, residue_edge_index, residue_edge_weight, prot_local_R, prot_local_t, prot_pred_site, protein_domain_mask, protein_domain_lbl,
            RNA_batch, prot_batch, 
            inter_residue_feat_pad,
            _, _, _, _
        ) = batch
        # print(f"Processing {pdbname} with RNA size {nuc_x.shape[0]} and Protein size {residue_x.shape[0]}")
        final_logits = self.model(
            nuc_x, nuc_evo_x, nuc_coord_x, nuc_edge_index, nuc_edge_weight, RNA_local_R, RNA_local_t, RNA_pred_site,
            residue_x, residue_evo_x, residue_coord_x, residue_edge_index, residue_edge_weight, prot_local_R, prot_local_t, prot_pred_site, protein_domain_mask, 
            RNA_batch, prot_batch,
            inter_residue_feat_pad
        )  # [B, h, H, W]

        return final_logits

    # ==========================================================
    # metrics
    # ==========================================================

    def logits_to_contact_score(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert multiclass distance logits -> contact score per pixel.
        Return: score [B, H, W] (higher = more likely contact)
        """
        probs = torch.softmax(logits.permute(0, 2, 3, 1), dim=-1)  # [B,H,W,C]
        if self.cfg.num_classes == 37:
            score = probs[..., :self.cfg.contact_class_cutoff].sum(-1)
        else:
            score = probs[..., :1].sum(-1)
        return score  # [B,H,W]


    def label_to_contact_binary(self, lbl_pad: torch.Tensor) -> torch.Tensor:
        """Convert valid class indices to contacts while preserving padding."""
        cutoff = self.cfg.contact_class_cutoff if self.cfg.num_classes == 37 else 1
        return torch.where(lbl_pad == self.cfg.pad_value,
                           torch.full_like(lbl_pad, self.cfg.pad_value, dtype=torch.float32),
                           (lbl_pad < cutoff).float())


    def compute_precision_topk(self, score, target, topk_list):
        """Compute precision using the unpadded RNA length for L and L/2."""
        length = int((target != self.cfg.pad_value).any(dim=1).sum())
        results = precision_topk(score, target, list(topk_list) + [length // 2, length], self.cfg.pad_value)
        return {**{k: results[k] for k in topk_list}, 'L/2': results[length // 2], 'L': results[length]}
              

def print_metrics_table(train_metrics, valid_metrics, test_metrics):
    'Print aligned training, validation and test metrics with four decimal places.'
    # Define metric keys and display names.
    # Display loss before residue metrics.
    metrics_to_show = [
        ('loss', 'Loss (损失值)'),
        # Group residue-level metrics.
        ('Res_Top1', 'Res_Top1'),
        ('Res_Top3', 'Res_Top3'),
        ('Res_Top5', 'Res_Top5'),
        ('Res_Top10', 'Res_Top10'),
        ('Res_Top20', 'Res_Top20'),
        ('Res_TopL/2', 'Res_TopL/2'),
        ('Res_TopL', 'Res_TopL'),
    ]

    # Table header
    header = f"{'指标项目':<15} | {'训练集 (Train)':>15} | {'验证集 (Valid)':>15} | {'测试集 (Test)':>15}"
    separator = "-" * len(header)

    print("\n" + separator)
    print(header)
    print(separator)

    # Print one row per metric.
    for key, display_name in metrics_to_show:
        train_val = train_metrics.get(key, 0.0)
        valid_val = valid_metrics.get(key, 0.0)
        test_val = test_metrics.get(key, 0.0)

        # Left-align names and right-align values with four decimal places.
        print(f"{display_name:<15} | {train_val:15.4f} | {valid_val:15.4f} | {test_val:15.4f}")

    print(separator + "\n")
