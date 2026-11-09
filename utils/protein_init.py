import numpy as np
from utils.geometry import calc_residue_dist, calc_dist_matrix
import pandas as pd
import sys
import Bio.PDB
# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import torch

from torch_geometric.utils import degree, add_self_loops, subgraph, to_undirected, remove_self_loops, coalesce

import math
import json
import os

def protein_init(seqs, seq_files=None, pdb_files=None, pdbcodes=None, feature_root=None):
    '''
    Encode protein graph features.

    :param seqs: List of protein sequences.
    :param seq_files: List of corresponding sequence file paths.
    :param pdb_files: List of corresponding PDB file paths.
    :param pdbcodes: List of unique identifiers for each sequence (e.g. "1C0A_C_A").

    :return: Dictionary containing protein features.
    '''
    if any(values is None or len(values) != len(seqs) for values in (seq_files, pdb_files, pdbcodes)):
        raise ValueError('Provide one sequence path, PDB path and identifier per protein sequence')
    result_dict = {}
    esm2_model = batch_converter = None

    for idx, seq in enumerate(tqdm(seqs)):
        pdbcode = pdbcodes[idx]
        print(pdbcode)
        seq_file = seq_files[idx]
        out_path = os.path.dirname(seq_file)
        os.makedirs(out_path, exist_ok=True)
        dataset = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(seq_file))))

        root = feature_root if feature_root is not None else os.path.join(os.path.dirname(seq_file), 'features')
        out_path = os.path.join(root, 'protein', f'{pdbcode}_protein')
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(out_path, f"{pdbcode}_protein_feature.pt" if pdb_files else f"{pdbcode}_protein_feature_seq.pt")
        
        # Reuse existing cached features.
        if os.path.exists(out_file):
            result_dict[pdbcode] = torch.load(out_file)
        
        else:
            #============================ 1 Extract sequence features. ==============================#
            seq_feat = seq_feature(seq)

            #============================ 2 construct graph ==============================#
            pdb_file = pdb_files[idx]
            edge_index, edge_weight = contact_map(pdb_file=pdb_file)
            structure = Bio.PDB.PDBParser().get_structure('', pdb_file)

            # Extract coordinates.
            xyz_coord = get_atom_coord(structure, atom='CA')

            assert xyz_coord.shape[0] == seq_feat.shape[0], f"{pdbcode} 坐标和序列特征长度不一致"

            result_dict[pdbcode] = {
                'seq': seq,
                'seq_feat': torch.from_numpy(seq_feat),
                'xyz_coord': torch.from_numpy(xyz_coord),
                'num_nodes': len(seq),
                'num_pos': torch.arange(len(seq)).reshape(-1, 1),
                'edge_index': edge_index,
                'edge_weight': edge_weight,
            }

            torch.save(result_dict[pdbcode], out_file)

        #==================== 4 local frame system coordinate ====================#

        local_frame_savefile = f"{out_path}/{pdbcode}_protein_local_frame.pt"
        pdb_file = pdb_files[idx]
        structure = Bio.PDB.PDBParser().get_structure('', pdb_file)
        pdb_struc = structure
        if os.path.exists(local_frame_savefile):
            local_frame = torch.load(local_frame_savefile)
            result_dict[pdbcode]['xyz_coord_CA'] = local_frame['xyz_coord_CA']
            result_dict[pdbcode]['xyz_coord_C'] = local_frame['xyz_coord_C']
            result_dict[pdbcode]['xyz_coord_N'] = local_frame['xyz_coord_N']

        else:

            result_dict[pdbcode]['xyz_coord_CA'] = torch.from_numpy(get_specific_atom_coord(pdb_struc, atom='CA'))
            result_dict[pdbcode]['xyz_coord_C'] = torch.from_numpy(get_specific_atom_coord(pdb_struc, atom='C'))
            result_dict[pdbcode]['xyz_coord_N'] = torch.from_numpy(get_specific_atom_coord(pdb_struc, atom='N'))
            local_frame = {
                'xyz_coord_CA': result_dict[pdbcode]['xyz_coord_CA'],
                'xyz_coord_C': result_dict[pdbcode]['xyz_coord_C'],
                'xyz_coord_N': result_dict[pdbcode]['xyz_coord_N']
            }
            # print(local_frame)
            torch.save(local_frame, local_frame_savefile)   

        seq_len = result_dict[pdbcode]['seq_feat'].shape[0]

        for atom_key in ['xyz_coord_CA', 'xyz_coord_C', 'xyz_coord_N']:
            coord_len = result_dict[pdbcode][atom_key].shape[0]
            assert coord_len == seq_len, (
                f"{pdb_file}, {atom_key} 的第一个维度 {coord_len} 和 seq_feat 的第一个维度 {seq_len} 不相同"
            )

        #==================== 5 evolutionary feature ====================#
        evo_feat_savefile = f"{out_path}/{pdbcode}_protein_evo_feat.pt"
        if os.path.exists(evo_feat_savefile):        
            evo_feat = torch.load(evo_feat_savefile)
            result_dict[pdbcode]['token_representation'] = evo_feat
        else:
            
            # Load the ESM2 model.
            print(f'Loading ESM2 model for {pdbcode}...')
            if esm2_model is None:
                import esm
                esm2_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                esm2_model.eval()
                batch_converter = alphabet.get_batch_converter()
            
            if torch.cuda.is_available():
                esm2_model = esm2_model.cuda()
            token_repr, contact_map_proba, _ = esm_extract(
                esm2_model, batch_converter, seq, layer=33, approach='last', dim=1280
            )
            result_dict[pdbcode]['token_representation'] = token_repr.half()
            torch.save(result_dict[pdbcode]['token_representation'], evo_feat_savefile)

    return result_dict

def predict_protein_esmfold(seqs, seq_files=None, pdb_files=None, pdbcodes=None):
    '''
    Encode protein graph features.

    :param seqs: List of protein sequences.
    :param seq_files: List of corresponding sequence file paths.
    :param pdb_files: List of PDB file paths (optional).
    :param pdbcodes: List of unique identifiers for each sequence (e.g. "1C0A_C_A").

    :return: Dictionary containing protein features.
    '''
    result_dict = {}


    # Load the ESMFold model.
    from esm import pretrained
    esmfold_model = pretrained.esmfold_v1()
    esmfold_model = esmfold_model.eval().cuda()  # Use a CUDA device when available.
    # esmfold_model = esmfold_model.eval().cpu().float()  # Move to CPU and convert to float32.
    esmfold_model.set_chunk_size(64)

    for idx, seq in enumerate(tqdm(seqs)):

        pdbcode = pdbcodes[idx]
        seq_file = seq_files[idx]
        out_path = os.path.dirname(seq_file)
        os.makedirs(out_path, exist_ok=True)

        if pdb_files is None:

            # Generate a predicted structure.
            pred_pdb_dir = os.path.join(out_path, 'esmfold_model')
            os.makedirs(pred_pdb_dir, exist_ok=True)
            pdb_file = os.path.join(pred_pdb_dir, f"{pdbcode}_protein_esmfold.pdb")

            if not os.path.exists(pdb_file):
                with torch.no_grad():
                    output = esmfold_model.infer_pdb(seq)

                with open(pdb_file, "w") as f:
                    f.write(output)

            # structure = Bio.PDB.PDBParser().get_structure('', pdb_file)

        else:
            # Build the graph from the supplied structure.
            pdb_file = pdb_files[idx]
            edge_index, edge_weight = contact_map(pdb_file=pdb_file)
            # structure = Bio.PDB.PDBParser().get_structure('', pdb_file)

# normalize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def seq_feature(pro_seq):
    if 'U' in pro_seq or 'B' in pro_seq:
        print('U or B in Sequence')
    pro_seq = pro_seq.replace('U', 'X').replace('B', 'X')
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)


def contact_map(contact_map_proba=None, pdb_file=None, contact_threshold=0.5, distance_cutoff=8):

    if pdb_file is None:
        num_residues = contact_map_proba.shape[0]
        prot_contact_adj = (contact_map_proba >= contact_threshold).long()
        edge_index = prot_contact_adj.nonzero(as_tuple=False).t().contiguous()
        row, col = edge_index
        edge_weight = contact_map_proba[row, col].float()

    else:
        structure = Bio.PDB.PDBParser().get_structure('', pdb_file)
        model = structure[0]
        # chains = []
        # for chain in model:
        #     chains.append(chain.get_id())
        # contact_map_proba = calc_dist_matrix(model[chains[0]], model[chains[0]])
        # contact_map_proba = -contact_map_proba/distance_cutoff + 1  # distance_cutoff = 0, dist 0 = 1
        # contact_map_proba = torch.tensor(contact_map_proba)

        # num_residues = contact_map_proba.shape[0]
        # prot_contact_adj = (contact_map_proba >= contact_threshold).long()
        # edge_index = prot_contact_adj.nonzero(as_tuple=False).t().contiguous()
        # row, col = edge_index
        # edge_weight = contact_map_proba[row, col].float()
        chains = []
        for chain in model:
            chains.append(chain.get_id())
        contact_map_proba = calc_dist_matrix(model)
        contact_map_proba = -contact_map_proba/distance_cutoff + 1  # distance_cutoff = 0, dist 0 = 1
        contact_map_proba = torch.tensor(contact_map_proba)

        num_residues = contact_map_proba.shape[0]
        prot_contact_adj = (contact_map_proba >= contact_threshold).long()
        edge_index = prot_contact_adj.nonzero(as_tuple=False).t().contiguous()
        row, col = edge_index
        edge_weight = contact_map_proba[row, col].float()

    ############### CONNECT ISOLATED NODES - Prevent Disconnected Residues ######################
    seq_edge_head1 = torch.stack([torch.arange(num_residues)[:-1], (torch.arange(num_residues) + 1)[:-1]])
    seq_edge_tail1 = torch.stack([(torch.arange(num_residues))[1:], (torch.arange(num_residues) - 1)[1:]])
    seq_edge_weight1 = torch.ones(seq_edge_head1.size(1) + seq_edge_tail1.size(1)) * contact_threshold
    edge_index = torch.cat([edge_index, seq_edge_head1, seq_edge_tail1], dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight1], dim=-1)

    seq_edge_head2 = torch.stack([torch.arange(num_residues)[:-2], (torch.arange(num_residues) + 2)[:-2]])
    seq_edge_tail2 = torch.stack([(torch.arange(num_residues))[2:], (torch.arange(num_residues) - 2)[2:]])
    seq_edge_weight2 = torch.ones(seq_edge_head2.size(1) + seq_edge_tail2.size(1)) * contact_threshold
    edge_index = torch.cat([edge_index, seq_edge_head2, seq_edge_tail2], dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight2], dim=-1)
    ############### CONNECT ISOLATED NODES - Prevent Disconnected Residues ######################

    edge_index, edge_weight = coalesce(edge_index, edge_weight, reduce='max')
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce='max')
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1)

    return edge_index, edge_weight



def esm_extract(model, batch_converter, seq, layer=36, approach='mean', dim=2560):
    """Extract ESM features, averaging overlapping windows for long sequences."""
    if not seq or approach not in ('last', 'sum', 'mean'):
        raise ValueError('Provide a nonempty sequence and approach last, sum or mean')
    layers = [layer] if approach == 'last' else list(range(1, layer + 1))
    length = len(seq)
    contacts = np.zeros((length, length), dtype=np.float32)
    contact_counts = np.zeros((length, length), dtype=np.int32)
    token_counts = np.zeros(length, dtype=np.int32)
    tokens = logits = None
    for start in range(0, length, 350):
        end = min(start + 700, length)
        _, _, batch_tokens = batch_converter([('A', seq[start:end])])
        batch_tokens = batch_tokens.to(next(model.parameters()).device, non_blocking=True)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=layers, return_contacts=True)
        representations = [results['representations'][i][0, 1:end - start + 1] for i in layers]
        representation = representations[0]
        if approach != 'last':
            representation = torch.stack(representations).sum(dim=0)
            if approach == 'mean':
                representation = representation / len(layers)
        representation = representation.cpu().numpy()
        window_logits = results['logits'][0, 1:end - start + 1].cpu().numpy()
        if tokens is None:
            tokens = np.zeros((length, representation.shape[-1]), dtype=representation.dtype)
            logits = np.zeros((length, window_logits.shape[-1]), dtype=window_logits.dtype)
        tokens[start:end] += representation
        logits[start:end] += window_logits
        token_counts[start:end] += 1
        contacts[start:end, start:end] += results['contacts'][0].cpu().numpy()
        contact_counts[start:end, start:end] += 1
        if end == length:
            break
    tokens /= token_counts[:, None]
    logits /= token_counts[:, None]
    np.divide(contacts, contact_counts, out=contacts, where=contact_counts > 0)
    return torch.from_numpy(tokens), torch.from_numpy(contacts), torch.from_numpy(logits)


def generate_ESM_structure(model, filename, sequence):
    model.set_chunk_size(256)
    chunk_size = 256
    output = None

    while output is None:
        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            with open(filename, "w") as f:
                f.write(output)
                print("saved", filename)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory on chunk_size', chunk_size)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                chunk_size = chunk_size // 2
                if chunk_size > 2:
                    model.set_chunk_size(chunk_size)
                else:
                    print("Not enough memory for ESMFold")
                    break
            else:
                raise e
    return output is not None


from Bio.PDB import PDBParser

biopython_parser = PDBParser()

one_to_three = {"A": "ALA",
                "C": "CYS",
                "D": "ASP",
                "E": "GLU",
                "F": "PHE",
                "G": "GLY",
                "H": "HIS",
                "I": "ILE",
                "K": "LYS",
                "L": "LEU",
                "M": "MET",
                "N": "ASN",
                "P": "PRO",
                "Q": "GLN",
                "R": "ARG",
                "S": "SER",
                "T": "THR",
                "V": "VAL",
                "W": "TRP",
                "Y": "TYR",
                "B": "ASX",
                "Z": "GLX",
                "X": "UNK",
                "*": " * "}

three_to_one = {}
for _key, _value in one_to_three.items():
    three_to_one[_value] = _key
three_to_one["SEC"] = "C"
three_to_one["MSE"] = "M"


def extract_pdb_seq(protein_path):
    structure = biopython_parser.get_structure('random_id', protein_path)[0]
    seq = ''
    chain_str = ''
    for i, chain in enumerate(structure):
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid and not
                try:
                    seq += three_to_one[residue.get_resname()]
                    chain_str += str(chain.id)
                except Exception as e:
                    seq += 'X'
                    chain_str += str(chain.id)
                    print("encountered unknown AA: ", residue.get_resname(),
                          ' in the complex. Replacing it with a dash X.')

    return seq, chain_str



# def calc_dist_matrix(chain_one, chain_two) :
#     """Returns a matrix of C-alpha distances between two chains"""
#     answer = np.zeros((len(chain_one), len(chain_two)), dtype=float)
#     for row, residue_one in enumerate(chain_one) :
#         for col, residue_two in enumerate(chain_two) :
#             answer[row, col] = calc_residue_dist(residue_one, residue_two)
#     return answer



def get_atom_coord(pdb_struc, atom=None):
    """
    Get the coordinates of the specified atom in the PDB structure.
    :param pdb_struc: The PDB structure object.
    :param atom: The atom name to extract coordinates for (default is None).
    :return: A numpy array of shape (num_atoms, 3) containing the coordinates.
    """
    coords = []
    for model in pdb_struc:
        for chain in model:
            for residue in chain:
                target_atom = atom
                # Fall back to the first atom if the requested atoms are missing.
                if not residue.has_id(target_atom):
                    first_atom = list(residue.get_atoms())[0]
                    target_atom = first_atom.get_name()
                    
                if residue.has_id(target_atom):
                    coords.append(residue[target_atom].get_coord())
    return np.array(coords)


def get_specific_atom_coord(pdb_struc, atom=None):
    """
    Get the coordinates of the specified atom in the PDB structure.

    Parameters
    ----------
    pdb_struc : Bio.PDB.Structure.Structure
        The PDB structure object.
    atom : str
        The atom name to extract coordinates for.
        Special case:
            if atom == 'N', then:
                C/U -> N1
                A/G -> N9

    Returns
    -------
    np.ndarray
        A numpy array of shape (num_atoms, 3) containing the coordinates.
    """
    coords = []

    for model in pdb_struc:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip().upper()
                target_atom = atom

                if residue.has_id(target_atom):
                    coords.append(residue[target_atom].get_coord())
                else:
                    first_atom = list(residue.get_atoms())[0]
                    # print(first_atom)
                    target_atom = first_atom.get_name()
                    coords.append(residue[target_atom].get_coord())

    return np.array(coords)
