#!/usr/bin/env python
# coding: utf-8
"""
RNA–Protein Contact Prediction Script
-------------------------------------
This script predicts contacts between RNA nucleotides and protein residues
using a trained ZHMolContact model. 

Author: Haoquan Liu, Yunjie Zhao
Date: 2026-10-18
"""


import json
import pickle
import argparse
import torch
import pandas as pd
from utils.protein_init import *
from utils.RNA_init import *
from utils.dataset import *
from utils.Predictor_inter import Predictor, read_batch
import os
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from torch_geometric.loader import DataLoader as GeoDataLoader

from utils.dataset import RNAProteinMoleculeDataset
from utils.utils_with_inter_edge_feat import pad_conmap_truth_lbl, rigid_transform_perturb_batch
from utils.contact_metrics import precision_topk as _precision_topk
from utils.Predictor_inter import Predictor, read_batch
from models.ZHMolContact_with_refine import ZHMolContact
from utils.prior_init import prior_init


def logits_to_contact_score(
    logits: torch.Tensor,
    num_classes: int,
    contact_class_cutoff: int = 16
) -> torch.Tensor:
    """
    Convert multi-class distance logits into contact probability scores.

    Args:
        logits: Input tensor of shape [B, C, H, W].
        num_classes: Total number of distance classes.
        contact_class_cutoff: Number of classes considered as contacts
            when num_classes == 37.

    Returns:
        torch.Tensor: Contact probability scores of shape [B, H, W],
            where higher values indicate higher contact probabilities.
    """
    # Move the channel dimension to the last axis: [B, C, H, W] -> [B, H, W, C]
    probs = torch.softmax(logits.permute(0, 2, 3, 1), dim=-1)

    if num_classes == 37:
        # Sum the probabilities of the first contact_class_cutoff classes
        score = probs[..., :contact_class_cutoff].sum(-1)
    else:
        # Use the probability of the first class by default
        score = probs[..., :1].sum(-1)

    return score  # [B, H, W]

# -----------------------------
# Argument Parser and Config Loader
# -----------------------------
def parse_args():
    """
    Parse command-line arguments and optionally merge them with a JSON config file.
    Command-line arguments take precedence over JSON file values.
    """

    parser = argparse.ArgumentParser(description="Run RNA-protein contact prediction with JSON config")
    
    # JSON configuration file
    parser.add_argument('--config', type=str, default='config.json', help='Path to JSON config file')
    
     # Optional overrides for config values
    parser.add_argument('--work_dir', type=str, help='Override work_dir')
    parser.add_argument('--project_name', type=str, help='Override project name')
    parser.add_argument('--RNA_fasta_file', type=str, help='Override RNA fasta file')
    parser.add_argument('--prot_fasta_file', type=str, help='Override protein fasta file')
    parser.add_argument('--RNA_pdb_file', type=str, help='Override RNA pdb file')
    parser.add_argument('--prot_pdb_file', type=str, help='Override protein pdb file')
    parser.add_argument('--model_path', type=str, help='Override model path')

    args = parser.parse_args()

    # Load configuration from JSON file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Overwrite JSON config with command-line arguments (if provided)
    for k, v in vars(args).items():
        if v is not None and k != 'config':
            config[k] = v

    return config

# -----------------------------
# FASTA Parser
# -----------------------------
def parse_fasta(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip()
    sequence = ''.join(line.strip() for line in lines[1:])
    return header, sequence

# -----------------------------
# Main Prediction Pipeline
# -----------------------------
def main():
    # Load configuration (from JSON + CLI)
    config = parse_args()
    seed = 2024
    # Extract parameters
    data_path = config['work_dir']
    project_name = config['project_name']
    RNA_fasta_file = config['RNA_fasta_file']
    prot_fasta_file = config['prot_fasta_file']
    RNA_pdb_file = config['RNA_pdb_file']
    prot_pdb_file = config['prot_pdb_file']
    model_path = config['model_path']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Load sequences from FASTA
    # -----------------------------
    _, RNA_fa = parse_fasta(RNA_fasta_file)
    _, prot_fa = parse_fasta(prot_fasta_file)

    # Build input DataFrame for dataset creation
    records = [{
        'PDB_code': project_name,
        'RNA_seq': RNA_fa,
        'prot_seq': prot_fa,
        'RNA_PDB': RNA_pdb_file,
        'protein_PDB': prot_pdb_file,
        'RNA_fa': RNA_fasta_file,
        'protein_fa': prot_fasta_file
    }]
    test_df = pd.DataFrame(records)
    # print(test_df)

    # -----------------------------
    # Initialize or Load RNA/Protein Graphs
    # -----------------------------
    RNA_path = os.path.join(f'{data_path}', f'RNA_{project_name}.pt')
    protein_path = os.path.join(f'{data_path}', f'protein_{project_name}.pt')
    prior_path = os.path.join(f'{data_path}', f'prior_{project_name}.pt')

    # Load or create RNA graph
    if os.path.exists(RNA_path):
        print('Loading RNA Graph data...')
        RNA_dict = torch.load(RNA_path)
    else:
        print('Initialising RNA Sequence to RNA Graph...')
        RNA_pdb_files = (
            [RNA_pdb_file]
            if RNA_pdb_file is not None
            else None
        )
        RNA_dict = RNA_init([RNA_fa], pdb_files=RNA_pdb_files,
                            fa_files=[RNA_fasta_file], pdbcodes=[project_name])
        torch.save(RNA_dict, RNA_path)

    # Load or create protein graph
    if os.path.exists(protein_path):
        print('Loading Protein Graph data...')
        prot_dict = torch.load(protein_path)
    else:
        print('Initialising Protein Sequence to Protein Graph...')
        prot_pdb_files = (
            [prot_pdb_file]
            if prot_pdb_file is not None
            else None
        )
        prot_dict = protein_init([prot_fa], seq_files=[prot_fasta_file],
                                 pdb_files=prot_pdb_files, pdbcodes=[project_name])
        torch.save(prot_dict, protein_path)

    # Load prior information 
    
    if os.path.exists(prior_path):
        print('Loading prior information...')
        prior_dict = torch.load(prior_path)
    else:
        print('Generating prior information...')
        prior_dict = prior_init(RNA_path=RNA_path, prot_path=protein_path,
                                RNA_fa_files=[RNA_fasta_file], prot_fa_files=[prot_fasta_file],
                                pdbcodes=[project_name], rna_site_dir=data_path, prot_site_dir=data_path)
        torch.save(prior_dict, prior_path)

    # -----------------------------
    # Create Dataset and DataLoader
    # -----------------------------
    test_dataset = RNAProteinMoleculeDataset(test_df, RNA_dict, prot_dict, prior_dict=prior_dict, device=device)
    test_loader = GeoDataLoader(test_dataset, batch_size=1, follow_batch=['RNA_node_aa', 'prot_node_aa'])
    # print(test_loader)
    # -----------------------------
    # Load Pre-trained Model
    # -----------------------------
    print(f"Loading model from: {model_path}")

    dropout = 0.05
    heads = 4
    H = 48
    n_ls = 2
    n_tri_ls = 2

    # ==================== Initialize Model ====================
    zhmolcontact = ZHMolContact(
        dropout=dropout,
        node_feat_dropout=dropout,
        evo_feat_dropout=dropout,
        heads=heads,
        hidden_channels=H,
        num_layers=n_ls,
        num_triangle_layers=n_tri_ls
    )

    # ==================== Load Pretrained Weights ====================
    checkpoint = torch.load(model_path, map_location='cpu')

    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )

    zhmolcontact.load_state_dict(state_dict)

    # ==================== Prepare Model for Inference ====================
    zhmolcontact = zhmolcontact.to(device)
    zhmolcontact.eval()

    # ==================== Initialize Predictor ====================
    predictor = Predictor(
        model=zhmolcontact,
        device=device
    )

    # -----------------------------
    # Run Inference
    # -----------------------------
    result_dir = os.path.join(data_path, "results")
    os.makedirs(result_dir, exist_ok=True)
    pred_dir = os.path.join(result_dir, "contact_probs")
    LFT_pred_dir = os.path.join(result_dir, "contact_probs_LFT")
    pred_logits_dir = os.path.join(result_dir, "contact_probs_logits")
    for directory in [pred_dir, LFT_pred_dir, pred_logits_dir]:
        os.makedirs(directory, exist_ok=True)

    zhmolcontact.eval()
    for data in test_loader:
        data = data.to(device)
        data = pad_conmap_truth_lbl(data, mode="blind")
        data = rigid_transform_perturb_batch(data, seed=seed, oprate_trans=False)
        test_id = data.pdbname[0]

        with torch.no_grad():
            final_logits = predictor.predict(data=data)
            contact_probs = logits_to_contact_score(
                final_logits, num_classes=37, contact_class_cutoff=16
            )
            distance_probs = torch.softmax(final_logits, dim=1)
            distance_probs = distance_probs.permute(0, 2, 3, 1).squeeze(0)

        # Save contact probability matrix: [H, W]
        contact_probs_np = contact_probs.squeeze(0).detach().cpu().numpy()
        pred_df = pd.DataFrame(contact_probs_np)
        pred_df.to_csv(
            os.path.join(pred_dir, f"{test_id}.csv"), index=False, header=False
        )

        # Save long-format contact probabilities: row, col, value
        long_df = pred_df.stack().reset_index()
        long_df.columns = ["row", "col", "value"]
        long_df.to_csv(os.path.join(LFT_pred_dir, f"{test_id}.csv"), index=False)

        # Save distance probability distributions: [H, W, 37]
        torch.save(
            distance_probs.detach().cpu(),
            os.path.join(pred_logits_dir, f"{test_id}.pt")
        )
        print(f"Three prediction files saved for: {test_id}")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == '__main__':
    main()
