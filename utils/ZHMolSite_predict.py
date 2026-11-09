#!/usr/bin/env python
# coding: utf-8
"""
RNA-Protein Binding Nucleotides Prediction Script
-----------------------------------------
This script runs RNA-protein binding nucleotides prediction using a trained model.
It supports both JSON-based configuration and command-line argument overrides.

Author: Haoquan Liu, Yunjie Zhao
Date: 2025-12-20
"""

import os
import json
import pickle
import argparse
from pathlib import Path
import torch
import pandas as pd
from utils.ZHMolSite_Predictor import Predictor
from utils.ZHMolSite_utils import DataLoader
from utils.ZHMolSite_dataset import *


def predict_binding_sites(RNA_path, protein_path):
    """Predict RNA-binding nucleotides from one pair of prepared graph files.

    Each file must contain a dictionary with one complex ID as its key and
    the complete RNA or protein graph as its value. The IDs must match.
    Set model_path below to the trained ZHMolSite model used by this project.

    Returns
    -------
    pandas.DataFrame
        Per-nucleotide pred_score and pred_label columns. Prediction CSV and
        cluster pickle files are also saved beside the RNA graph file.
    """
    # Configure the trained model once; callers supply only graph paths.
    model_path = Path(__file__).resolve().parent / "models" / "ZHMolSite.pth"
    RNA_thresh = 0.521667
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    RNA_path = Path(RNA_path).expanduser().resolve()
    protein_path = Path(protein_path).expanduser().resolve()
    for path in (RNA_path, protein_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"File does not exist: {path}")

    RNA_dict = torch.load(RNA_path, map_location="cpu")
    prot_dict = torch.load(protein_path, map_location="cpu")
    if not isinstance(RNA_dict, dict) or not isinstance(prot_dict, dict):
        raise ValueError("Each graph file must contain a dictionary keyed by complex ID.")
    if len(RNA_dict) != 1 or len(prot_dict) != 1:
        raise ValueError("Each graph file must contain exactly one complex.")

    project_name = next(iter(RNA_dict))
    if project_name not in prot_dict:
        raise ValueError("RNA and protein graph dictionaries must use the same complex ID.")
    if not isinstance(project_name, str) or not project_name or project_name in (".", "..") or any(
        separator in project_name for separator in ("/", "\\")
    ):
        raise ValueError("The complex ID must be a nonempty filename-safe string.")

    test_df = pd.DataFrame([{
        "PDB_code": project_name,
        "RNA_seq": RNA_dict[project_name]["seq"],
        "prot_seq": prot_dict[project_name]["seq"],
    }])
    dataset = RNAProteinMoleculeDataset(test_df, RNA_dict, prot_dict, device=device)
    loader = DataLoader(
        dataset, batch_size=1, follow_batch=["RNA_node_aa", "prot_node_aa"]
    )

    # This follows the existing pipeline: the checkpoint contains a full model.
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.device = device
    model.eval()
    predictor = Predictor(model=model)

    batch = next(iter(loader)).to(device)
    with torch.no_grad():
        scores, RNA_clusters, protein_clusters = predictor.predict(data=batch)
    scores = scores.detach().cpu().numpy().reshape(-1)
    if len(scores) != len(test_df.loc[0, "RNA_seq"]):
        raise ValueError("The number of predicted scores does not match the RNA length.")
    pred_df = pd.DataFrame({
        "pred_score": scores,
        "pred_label": (scores >= RNA_thresh).astype(int),
    })

    output_dir = RNA_path.parent
    pred_file = output_dir / f"{project_name}_rna_pred_sites.csv"
    pred_df.to_csv(pred_file, index=False)
    with open(output_dir / f"{project_name}_RNA_K.pkl", "wb") as f:
        pickle.dump(RNA_clusters, f)
    with open(output_dir / f"{project_name}_prot_K.pkl", "wb") as f:
        pickle.dump(protein_clusters, f)
    print(f"Prediction saved to: {pred_file}")
    return pred_df


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == '__main__':
    main()
