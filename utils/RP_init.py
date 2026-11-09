import numpy as np
import pandas as pd
import os
import sys
import torch
from skimage.transform import resize
# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from Bio import SeqIO
import random
import os
import torch
import gzip
import pickle
from utils.RP_domain_contact import build_domain_contact_map

def resize_image_skimage(image, target_size):
    # Resize the image.
    resized_image = resize(image, target_size, order=0, anti_aliasing=True)
    return resized_image


def RP_init(RNA_seqs, prot_seqs, dist_files=None, dist_cutoff=None, strict_dist_cutoff=None, target_size=[32,512], 
            RNA_fa_files=None, prot_fa_files=None, joint_MSA_files=None, MSA_feat_files=None, pdbcodes=None, Cot_model=None, device=None, rna_site_dir=None, prot_site_dir=None):

    if dist_cutoff is None:
        raise ValueError('dist_cutoff is required when constructing contact labels')
    if dist_files is None or pdbcodes is None or not (len(dist_files) == len(pdbcodes) == len(RNA_seqs) == len(prot_seqs)):
        raise ValueError('Sequences, distance files and identifiers must have matching lengths')
    label_config = {'dist_cutoff': dist_cutoff, 'strict_dist_cutoff': strict_dist_cutoff,
                    'target_size': tuple(target_size)}
    result_dict = {}
    for idx, file in enumerate(tqdm(dist_files)):
        pdbcode = pdbcodes[idx]  # 1C0A_C_A for native, 1C0A_C_A_model_1 for docking decoys
        print(pdbcode)
        # print(file)
        seq_file = RNA_fa_files[idx] if RNA_fa_files is not None else file

        dataset = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(seq_file))))

        out_path = os.path.dirname(file)
        out_file = os.path.join(out_path, f'{pdbcode}_lbl.pt')
        # os.remove(out_file) if os.path.exists(out_file) else None  # Remove the existing file.
        dist_file = dist_files[idx]

        cached = torch.load(out_file, map_location='cpu') if os.path.exists(out_file) else None
        if cached is None or cached.get('_label_config') != label_config:

            path = os.path.dirname(dist_file)

            residue_map_file = os.path.join(f'{path}', 'residue_mapping.csv')
            nuc_map_file = os.path.join(f'{path}', 'nucleotide_mapping.csv')
            conmap_truth = create_conmap_truth(file, residue_map_file, nuc_map_file)

            bins = np.arange(2.5, 20.5, 0.5)  # Bin edges: [2.5, 3.0, 3.5, ..., 20.0].

            # Map distances into bins using np.digitize.
            valid_distances = np.isfinite(conmap_truth)
            conmap_truth_lbl = np.where(valid_distances, np.digitize(conmap_truth, bins, right=True), -100)

            min_val = np.min(conmap_truth)
            max_val = np.max(conmap_truth)
            # conmap_truth_std = (conmap_truth - min_val) / (max_val - min_val)  # Normalize to [0, 1].
            # conmap_truth_std = 255*(conmap_truth - min_val) / (max_val - min_val)  # Normalize to [0, 255].

            # conmap_truth_std = 2 * (conmap_truth - min_val) / (max_val - min_val) - 1  # Normalize to [-1, 1].

            # Missing distances are unknown labels, rather than negative contacts.
            conmap_truth_std = np.where(valid_distances, (conmap_truth <= dist_cutoff).astype(int), -100)
            if strict_dist_cutoff is not None:
                conmap_truth_strict_std = np.where(valid_distances, (conmap_truth <= strict_dist_cutoff).astype(int), -100)

            target_size = target_size
            conmap_truth_resize = resize_image_skimage(conmap_truth_std.astype(float), target_size)

            RNA_site_truth = create_RNA_site_truth(file, nuc_map_file, dist_cutoff=dist_cutoff)
            prot_site_truth = create_prot_site_truth(file, residue_map_file, dist_cutoff=dist_cutoff)
            
            
            ## binary 0 1
            # conmap_truth = np.where(conmap_truth <= dist_cutoff, 1, 0)
            # RNA_site_truth = np.where(RNA_site_truth <= dist_cutoff, 1, 0)
            # prot_site_truth = np.where(prot_site_truth <= dist_cutoff, 1, 0)

            ## multi target
            # vectorized_distance_to_k = np.vectorize(distance_to_k)
            # conmap_truth = vectorized_distance_to_k(conmap_truth)

            ## Regression task: normalize when computing loss.
            # min_val = np.min(conmap_truth)
            # max_val = np.max(conmap_truth)
            # conmap_truth = (conmap_truth - min_val) / (max_val - min_val)

            RNA_seq = RNA_seqs[idx]
            prot_seq = prot_seqs[idx]




            result_dict[pdbcode] = {
                'RNA seq': RNA_seq,
                'prot_seq': prot_seq,
                'conmap_truth_origin': conmap_truth,
                'conmap_truth': conmap_truth_std, 
                'conmap_truth_strict': conmap_truth_strict_std if strict_dist_cutoff is not None else None,
                'conmap_truth_resize': conmap_truth_resize,
                'conmap_truth_lbl': conmap_truth_lbl,
                'RNA_site_truth': RNA_site_truth,
                'prot_site_truth': prot_site_truth, 
                'pdbname': pdbcode,
                '_label_config': label_config,
            }
            torch.save(result_dict[pdbcode], out_file)
        else:
            # Load cached features.

            result_subdict = cached
            result_subdict['pdbname'] = pdbcode
            # print(result_dict)

            result_dict[pdbcode] = result_subdict

        RNA_pred_site_file = os.path.join(rna_site_dir, f'{pdbcode}.csv') if rna_site_dir is not None else f'/home/quan/RNA_Protein_contact_prediction/result/ZHMolSite_result/{dataset}/{pdbcode}.csv'
        RNA_pred_site_df = pd.read_csv(RNA_pred_site_file, sep=',')
        RNA_pred_site_columns = RNA_pred_site_df[['pred_score']].rename(columns={'pred_score': 'RNA site probs'})
        RNA_pred_site = RNA_pred_site_columns.to_numpy()
        result_dict[pdbcode]['RNA_pred_site'] = RNA_pred_site

        prot_pred_site_file = os.path.join(prot_site_dir, f'{pdbcode}_protein.txt') if prot_site_dir is not None else f'/home/quan/RNA_Protein_contact_prediction/protein_binding_sites_prediction/GPSite-main/example/{dataset}/pred/{pdbcode}_protein.txt'
        prot_pred_site_df = pd.read_csv(prot_pred_site_file, sep='\t')
        prot_pred_site_columns = prot_pred_site_df[['RNA_binding']].rename(columns={'RNA_binding': 'prot site probs'})
        prot_pred_site = prot_pred_site_columns.to_numpy()
        result_dict[pdbcode]['prot_pred_site'] = prot_pred_site

    return result_dict


def create_prot_site_truth(distance_file, residue_map_file, dist_cutoff):
    """Build a protein-site label column from distances and residue mapping."""
    return _create_site_truth(distance_file, residue_map_file, 'residue', 'residue_number', dist_cutoff)


def create_RNA_site_truth(distance_file, nuc_map_file, dist_cutoff):
    """Build an RNA-site label column from distances and nucleotide mapping."""
    return _create_site_truth(distance_file, nuc_map_file, 'nucleotide', 'nucleotide_number', dist_cutoff)


def create_conmap_truth(distance_file, residue_map_file, nuc_map_file):
    """Map distance rows to an RNA-by-protein matrix; unknown pairs remain NaN."""
    distances = _read_distances(distance_file)
    rna_indices, rna_size = _map_indices(distances['nuc'], nuc_map_file, 'nucleotide', 'nucleotide_number')
    prot_indices, prot_size = _map_indices(distances['residue'], residue_map_file, 'residue', 'residue_number')
    matrix = np.full((rna_size, prot_size), np.nan)
    matrix[rna_indices, prot_indices] = distances['dist'].to_numpy()
    return matrix


def _read_distances(path):
    return pd.read_csv(path, sep='\t', header=None, names=['nuc', 'residue', 'dist'])


def _map_indices(values, mapping_file, key, number):
    mapping = pd.read_csv(mapping_file)
    ids = mapping[number]
    if mapping.empty or mapping[key].duplicated().any() or ids.duplicated().any():
        raise ValueError(f'{mapping_file}: mapping must be nonempty and unique')
    if ids.isna().any() or (ids < 1).any() or (ids != ids.astype(int)).any():
        raise ValueError(f'{mapping_file}: indices must be positive integers')
    mapped = values.map(dict(zip(mapping[key], ids)))
    if mapped.isna().any():
        raise ValueError(f'{mapping_file}: distance rows contain unmapped identifiers')
    return mapped.to_numpy(dtype=np.int64) - 1, int(ids.max())


def _create_site_truth(distance_file, mapping_file, key, number, dist_cutoff):
    distances = _read_distances(distance_file)
    indices, size = _map_indices(distances['nuc' if key == 'nucleotide' else 'residue'], mapping_file, key, number)
    truth = np.zeros((size, 1), dtype=np.int64)
    truth[indices[distances['dist'].to_numpy() <= dist_cutoff], 0] = 1
    return truth

def distance_to_k(v):
    if v <= 4:
        lbl = 0
    elif v <= 8:
        lbl = 1
    elif v <= 12:
        lbl = 2
    elif v <= 16:
        lbl = 3
    elif v <= 20:
        lbl = 4
    elif v <= 24:
        lbl = 5
    elif v <= 28:
        lbl = 6
    elif v <= 32:
        lbl = 7
    elif v <= 36:
        lbl = 8
    elif v <= 40:
        lbl = 9
    elif v <= 44:
        lbl = 10
    elif v <= 48:
        lbl = 11
    elif v <= 52:
        lbl = 12
    elif v <= 56:
        lbl = 13
    else:
        lbl = 14

    return lbl

def distance_to_37(v):
    if v <= 2.5:
        lbl = 0
    elif v <= 3.0:
        lbl = 1
    elif v <= 3.5:
        lbl = 2
    elif v <= 4.0:
        lbl = 3
    elif v <= 4.5:
        lbl = 4
    elif v <= 5.0:
        lbl = 5
    elif v <= 5.5:
        lbl = 6
    elif v <= 6.0:
        lbl = 7
    elif v <= 6.5:
        lbl = 8
    elif v <= 7.0:
        lbl = 9
    elif v <= 7.5:
        lbl = 10
    elif v <= 8.0:
        lbl = 11
    elif v <= 8.5:
        lbl = 12
    elif v <= 9.0:
        lbl = 13
    elif v <= 9.5:
        lbl = 14
    elif v <= 10.0:
        lbl = 15
    elif v <= 10.5:
        lbl = 16
    elif v <= 11.0:
        lbl = 17
    elif v <= 11.5:
        lbl = 18
    elif v <= 12.0:
        lbl = 19
    elif v <= 12.5:
        lbl = 20
    elif v <= 13.0:
        lbl = 21
    elif v <= 13.5:
        lbl = 22
    elif v <= 14.0:
        lbl = 23
    elif v <= 14.5:
        lbl = 24
    elif v <= 15.0:
        lbl = 25
    elif v <= 15.5:
        lbl = 26
    elif v <= 16.0:
        lbl = 27
    elif v <= 16.5:
        lbl = 28
    elif v <= 17.0:
        lbl = 29
    elif v <= 17.5:
        lbl = 30
    elif v <= 18.0:
        lbl = 31
    elif v <= 18.5:
        lbl = 32
    elif v <= 19.0:
        lbl = 33
    elif v <= 19.5:
        lbl = 34
    elif v <= 20.0:
        lbl = 35
    else:
        lbl = 36

    return lbl
