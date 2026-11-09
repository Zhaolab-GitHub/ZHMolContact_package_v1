import numpy as np
from utils.geometry import calc_residue_dist, calc_dist_matrix
import pandas as pd
import sys
import Bio.PDB
from collections import defaultdict
# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
import torch
import math
from Bio.PDB import PDBParser
from torch_geometric.utils import degree, add_self_loops, subgraph, to_undirected, remove_self_loops, coalesce
import os
import json
import subprocess
from Bio import pairwise2
from Bio import SeqIO
from pathlib import Path
import tempfile
import gc

def RNA_init(seqs, pdb_files=None, fa_files=None, pdbcodes=None, feature_root=None):
    '''

    Encode RNA graph features.

    :param seqs: List of RNA sequences

    :param pdb_files: List of RNA PDB file paths 

    :param fa_files: List of RNA FASTA file paths 

    :param pdbcodes: List of PDB code identifiers 

    :return: Dictionary of RNA graph features keyed by pdbcode

    '''
    if any(values is None or len(values) != len(seqs) for values in (pdb_files, fa_files, pdbcodes)):
        raise ValueError('Provide one PDB path, FASTA path and identifier per RNA sequence')
    result_dict = {}
    model = batch_converter = None
    for idx, seq in enumerate(tqdm(seqs)):
        pdbcode = pdbcodes[idx]  # 1C0A_C_A for native, 1C0A_C_A_model_1 for docking decoys
        print(pdbcode)
        fa_file = fa_files[idx]
        pdb_file = pdb_files[idx]
        print(fa_file)
        dataset = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(fa_file))))
        
        root = feature_root if feature_root is not None else os.path.join(os.path.dirname(fa_file), 'features')
        out_path = os.path.join(root, 'RNA', f'{pdbcode}_RNA')
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(out_path, f"{pdbcode}_RNA_feature.pt" if pdb_files else f"{pdbcode}_RNA_feature_seq.pt")
        if not os.path.exists(out_file):
            structure = Bio.PDB.PDBParser().get_structure('', pdb_file)
            pdb_struc = structure

            #========================== 1 calculate sasa ======================== #
            SASA_result_save_file = f"{out_path}/{pdbcode}_RNA_SASA.pkl"
            if os.path.exists(SASA_result_save_file):
                sasa_result_df = pd.read_pickle(SASA_result_save_file)
            else:
                options = {'skip-unknown': False}  # Infer unknown atom parameters instead of skipping atoms.
                import freesasa
                freesasa_structure = freesasa.Structure(pdb_file, None, options)
                # calculate SASA from Bio.PDB structure #
                SASA_result = freesasa.calc(freesasa_structure)
                print(SASA_result)
                sasa_result = pd.DataFrame(columns=['res', 'total_sasa', 'polar_sasa', 'apolar_sasa'])
                ii = 0
                for chain_id, residues in SASA_result.residueAreas().items():
                    for residue_id, residue_area in sorted(residues.items(), key=lambda x: str(x[0])):
                        total_sasa = residue_area.total
                        polar_sasa = residue_area.polar
                        apolar_sasa = residue_area.apolar
                        new_row = {'res': ii, 'total_sasa': total_sasa, 'polar_sasa': polar_sasa, 'apolar_sasa': apolar_sasa}
                        sasa_result.loc[len(sasa_result)] = new_row
                        ii += 1
                sasa_result_df = add_rna_max_normalized_sasa(sasa_result)
                sasa_result_df.to_pickle(SASA_result_save_file)

            #=================== 2 calculate secondary structure ===================#
            ss_file_path = f"{out_path}/{pdbcode}_RNA.ss"
            final_ss = prepare_rna_ss(seq, pdb_file, fa_file, ss_file_path, pdbcode)

            #============ 3 calculate LN norm for each nucleotide in RNA =============#
            LN_result_save_file = f"{out_path}/{pdbcode}_RNA_LN.pkl"
            if os.path.exists(LN_result_save_file):
                LN_result = pd.read_pickle(LN_result_save_file)
            else:
                LN_result = cal_RNA_LN(pdb_file)
                columns = LN_result.iloc[:, 1:]
                spans = (columns.max() - columns.min()).replace(0, 1)
                LN_result.iloc[:, 1:] = (columns - columns.min()) / spans
                LN_result.to_pickle(LN_result_save_file)

            #========================== 4 encode sequence features ========================#
            seq_feat = RNA_feature(seq, sasa_result_df, LN_result, final_ss)

            #========================= 5 extract coordinate features =======================#
            xyz_coord = get_atom_coord(pdb_struc, atom='C4\'')
            # print('xyz_coord:', xyz_coord.shape)
            assert xyz_coord.shape[0] == seq_feat.shape[0], f"{pdb_file}, xyz_coord {xyz_coord.shape[0]} 和 seq_feat {seq_feat.shape[0]} 的第一个维度不相同"

            #========================= 6 construct graph  =======================#
            pdb_struc = pdb_struc[0]
            edge_index, edge_weight = contact_map(pdb=pdb_struc)
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
        else:
            # Load cached features.
            result_subdict = torch.load(out_file)
            # print(result_dict)
            result_dict[pdbcode] = result_subdict
            print(pdbcode)

        #========================= 7 domain mask  =======================#

        #==================== 8 local frame system coordinate ====================#
        local_frame_savefile = f"{out_path}/{pdbcode}_RNA_local_frame.pt"
        if os.path.exists(local_frame_savefile):
            local_frame = torch.load(local_frame_savefile)
            result_dict[pdbcode]['xyz_coord_P'] = local_frame['xyz_coord_P']
            result_dict[pdbcode]['xyz_coord_C4'] = local_frame['xyz_coord_C4']
            result_dict[pdbcode]['xyz_coord_N'] = local_frame['xyz_coord_N']
        else:
            structure = Bio.PDB.PDBParser().get_structure('', pdb_file)
            pdb_struc = structure
            result_dict[pdbcode]['xyz_coord_P'] = torch.from_numpy(get_specific_atom_coord(pdb_struc, atom='P'))
            result_dict[pdbcode]['xyz_coord_C4'] = torch.from_numpy(get_specific_atom_coord(pdb_struc, atom='C4\''))
            result_dict[pdbcode]['xyz_coord_N'] = torch.from_numpy(get_specific_atom_coord(pdb_struc, atom='N'))
            local_frame = {
                'xyz_coord_P': result_dict[pdbcode]['xyz_coord_P'],
                'xyz_coord_C4': result_dict[pdbcode]['xyz_coord_C4'],
                'xyz_coord_N': result_dict[pdbcode]['xyz_coord_N'],
            }
            torch.save(local_frame, local_frame_savefile)
        seq_len = result_dict[pdbcode]['seq_feat'].shape[0]
        print(seq_len)
        for atom_key in ['xyz_coord_P', 'xyz_coord_C4', 'xyz_coord_N']:
            coord_len = result_dict[pdbcode][atom_key].shape[0]
            assert coord_len == seq_len, (
                f"{pdb_file}, {atom_key} 的第一个维度 {coord_len} 和 seq_feat 的第一个维度 {seq_len} 不相同"
            )

        #==================== 9 evolutionary feature ====================#
        evo_feat_savefile = f"{out_path}/{pdbcode}_RNA_evo_feat.pt"
        if os.path.exists(evo_feat_savefile):
            evo_feat = torch.load(evo_feat_savefile)
            result_dict[pdbcode]['token_representation'] = evo_feat
        else:
            # Load RNA-FM model
            print("Loading RNA-FM model...")
            if model is None:
                import fm
                model, alphabet = fm.pretrained.rna_fm_t12()
                batch_converter = alphabet.get_batch_converter()
                model.eval()  # Disable dropout for deterministic embeddings.
            data = [
                ("RNA1", seq),
                ]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[12])
            token_repr = results["representations"][12][0][1:-1]
            result_dict[pdbcode]['token_representation'] = token_repr.half()
            torch.save(result_dict[pdbcode]['token_representation'], evo_feat_savefile)

        #==================== clear freesasa structure and result to save memory ====================#
        if 'freesasa_structure' in locals():
            del freesasa_structure
            del SASA_result
        gc.collect() # Run garbage collection.
    return result_dict

def add_rna_max_normalized_sasa(df, res_col='res'):
    'Normalize RNA total, polar and apolar SASA using the legacy global total-SASA maximum.'
    df = df.copy()
    df['total_max'] = df.groupby(res_col)['total_sasa'].transform('max')
    df['polar_max'] = df.groupby(res_col)['polar_sasa'].transform('max')
    df['apolar_max'] = df.groupby(res_col)['apolar_sasa'].transform('max')
    denominator = df['total_max'].max()
    denominator = denominator if denominator > 0 else 1.0
    df['total_sasa_norm'] = df['total_sasa'] / denominator
    df['polar_sasa_norm'] = df['polar_sasa'] / denominator
    df['apolar_sasa_norm'] = df['apolar_sasa'] / denominator
    df = df[[res_col, 'total_sasa_norm', 'polar_sasa_norm', 'apolar_sasa_norm']]
    return df
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
RNA_nuc_table = ['A','U','C','G','X']
RNA_ss_table = ['(', '.', ')', '[', ']']
RNA_nuc_pur_table = ['A', 'G']
RNA_nuc_pyr_table = ['U', 'C']
nuc_weight_table = {'A': 329.2, 'U': 306.2, 'C': 305.2, 'G': 345.2}
nuc_weight_table = dic_normalize(nuc_weight_table)
# nuc_ss_table = dic_normalize(nuc_ss_table)

def nuc_features(nuc, sasa_fea, ln_fea):
    nuc_property1 = [1 if nuc in RNA_nuc_pur_table else 0, 1 if nuc in RNA_nuc_pyr_table else 0]
    # nuc_ss = nuc_ss_dict[nuc]
    # nuc_property2 = [nuc_weight_table[nuc], nuc_ss_table[nuc_ss]]
    nuc_property2 = [nuc_weight_table[nuc]] + [x for x in sasa_fea] + [x for x in ln_fea]
    # print(np.array(nuc_property1 + nuc_property2).shape)
    return np.array(nuc_property1 + nuc_property2)
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

def RNA_feature(RNA_seq, sasa_result, LN_result, pred_ss):
    RNA_hot = np.zeros((len(RNA_seq), len(RNA_nuc_table)))
    RNA_ss_hot = np.zeros((len(RNA_seq), len(RNA_ss_table)))
    RNA_property = np.zeros((len(RNA_seq), 11))
    # if pdb_files is None:
    #     RNA_ss_dict = get_RNA_ss(pdb_file=pdb_file)
    # else:
    #     RNA_ss_dict = get_RNA_ss(pdb_file=pdb_files[idx])
    assert len(RNA_seq) == len(sasa_result) == len(LN_result) == len(pred_ss), \
        f"Length mismatch: len(RNA_seq)={len(RNA_seq)}, len(sasa_result)={len(sasa_result)}, len(LN_result)={len(LN_result)}, len(pred_ss)={len(pred_ss)}"
    for i in range(len(RNA_seq)):
        # if 'X' in RNA_seq:
        #     print(RNA_seq)
        RNA_hot[i,] = one_of_k_encoding(RNA_seq[i], RNA_nuc_table)
        RNA_ss_hot[i,] = one_of_k_encoding(pred_ss[i], RNA_ss_table)
        RNA_property[i,] = nuc_features(RNA_seq[i], sasa_result.iloc[i, 1:].to_numpy(), LN_result.iloc[i, 1:].to_numpy())
    return np.concatenate((RNA_hot, RNA_ss_hot, RNA_property), axis=1)

def contact_map(contact_map_proba=None, pdb=None, contact_threshold=0.5, distance_cutoff=8):
    if pdb is None:
        num_residues = contact_map_proba.shape[0]
        prot_contact_adj = (contact_map_proba >= contact_threshold).long()
        edge_index = prot_contact_adj.nonzero(as_tuple=False).t().contiguous()
        row, col = edge_index
        edge_weight = contact_map_proba[row, col].float()
    else:
        chains = []
        for chain in pdb:
            chains.append(chain.get_id())
        contact_map_proba = calc_dist_matrix(pdb)
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

def cal_RNA_LN(infile):
    """

    Processes a PDB file to calculate distance-weighted properties.

    Args:

        infile (str): Path to the PDB input file.

    Returns:

        pd.DataFrame: A DataFrame containing the calculated distances.

    """
    # Parse PDB file and filter relevant atoms
    pdb = defaultdict(list)
    previous_residue = None
    number = -1
    with open(infile, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            residue_id = (line[21:22], line[22:27])
            if residue_id != previous_residue:
                number += 1
                previous_residue = residue_id
            atom_name = line[12:16].strip()
            if atom_name.startswith(('C', 'O', 'N', 'P')):
                pdb[number].append(line)
    # Extract C3 atoms
    pdb_c3 = []
    for num in sorted(pdb, key=lambda x: int(x)):
        found_c3 = False
        for element_info in pdb[num]:
            element = element_info[13:15].strip()
            if element == 'C3':
                pdb_c3.append(element_info)
                found_c3 = True
                break
        if not found_c3:
            pdb_c3.append(pdb[num][0])
    # Calculate pairwise distances
    all_distance = []
    coordinates = {}
    # print(len(pdb_c3))
    for num, line in enumerate(pdb_c3):
        # num = line[22:27].strip()
        x, y, z = map(float, [line[30:38], line[38:46], line[46:54]])
        coordinates[num] = (x, y, z)
    for num1, coord1 in coordinates.items():
        for num2, coord2 in coordinates.items():
            if int(num1) > int(num2):
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))
                all_distance.append(dist)
    # Sort distances and calculate mean thresholds
    if not all_distance:
        columns = ['residue_number', 'LN_0', 'LN_1/4', 'LN_1/2', 'LN_3', '1']
        return pd.DataFrame([[i, 0., 0., 0., 0., 0.] for i in coordinates], columns=columns)
    all_distance.sort()
    mean = len(all_distance) // 4
    thresholds = [0, mean, mean * 2, mean * 3, len(all_distance) - 1]
    # Compute distance-weighted properties
    hash_map = defaultdict(lambda: defaultdict(dict))
    for num1, coord1 in coordinates.items():
        for num2, coord2 in coordinates.items():
            if abs(int(num1) - int(num2)) > 1:
                dist_squared = sum((a - b) ** 2 for a, b in zip(coord1, coord2))
                for threshold in thresholds:
                    cda = all_distance[threshold]
                    if cda != 0:
                        weight = math.exp(-dist_squared / (cda ** 2))
                    else:
                        weight = 1
                    hash_map[cda][num1][num2] = weight
    # Calculate distances and create DataFrame
    data = []
    for num1, coord1 in coordinates.items():
        x, y, z = coord1
        row = [num1]
        for threshold in thresholds:
            cda = all_distance[threshold]
            sumx = sumy = sumz = sumb = 0
            for num2, coord2 in coordinates.items():
                if abs(int(num1) - int(num2)) > 1:
                    x1, y1, z1 = coord2
                    weight = hash_map[cda][num1].get(num2, 0)
                    sumx += x1 * weight
                    sumy += y1 * weight
                    sumz += z1 * weight
                    sumb += weight
            if sumb == 0:
                meanx, meany, meanz = 0, 0, 0   # Alternatively use np.nan.
            else:
                meanx, meany, meanz = sumx / sumb, sumy / sumb, sumz / sumb
            distance = math.sqrt((x - meanx) ** 2 + (y - meany) ** 2 + (z - meanz) ** 2)
            row.append(distance)
        data.append(row)
    columns = ['residue_number'] + ['LN_0', 'LN_1/4', 'LN_1/2', 'LN_3', '1']
    df = pd.DataFrame(data, columns=columns)
    return df

######################################################################

######## RNA secondary structure prediction or calculation    ########

######################################################################

def _checked_rna_ss(ss, length):
    """Validate dot-bracket output and retain the graph's five-symbol encoding."""
    if not isinstance(ss, str):
        raise ValueError('Secondary structure must be a dot-bracket string')
    ss = ss.strip().replace('&', '')
    if len(ss) != length or not ss or set(ss) - set('.()[]{}<>'):
        raise ValueError(f'Invalid secondary structure: expected {length} dot-bracket symbols')
    opening = '([{<'
    closing = dict(zip(')]}>', opening))
    counts = dict.fromkeys(opening, 0)
    for symbol in ss:
        if symbol in opening:
            counts[symbol] += 1
        elif symbol in closing:
            bracket = closing[symbol]
            counts[bracket] -= 1
            if counts[bracket] < 0:
                raise ValueError('Secondary structure contains an unmatched closing bracket')
    if any(counts.values()):
        raise ValueError('Secondary structure contains unmatched opening brackets')
    return ss.translate(str.maketrans('{}<>', '[][]'))

def _ss_error_message(error):
    detail = getattr(error, 'stderr', None)
    if isinstance(detail, bytes):
        detail = detail.decode('utf-8', errors='replace')
    return f'{error}: {detail.strip()}' if detail else str(error)

def prepare_rna_ss(seq, pdb_file, fa_file, ss_file_path, pdbcode):
    """Reuse valid SS; try DSSR first and fall back to MXfold2 on failure.

    Saved files retain the three-line header/sequence/structure format.
    The header records whether DSSR or MXfold2 supplied the structure.
    Existing graph-feature caches still follow RNA_init's original reuse policy.
    """
    reference = str(seq).strip().upper().replace('T', 'U')
    if not reference:
        raise ValueError('RNA sequence cannot be empty')
    ss_path = Path(ss_file_path).expanduser()
    if ss_path.is_file():
        try:
            lines = ss_path.read_text(encoding='utf-8').splitlines()
            if len(lines) < 3 or lines[1].strip().upper().replace('T', 'U') != reference:
                raise ValueError('Cached SS sequence differs from the current RNA')
            return _checked_rna_ss(lines[2], len(reference))
        except (OSError, ValueError) as error:
            print(f'Rebuilding invalid RNA SS cache {ss_path}: {error}', flush=True)
    try:
        ss, seq_ss = get_native_x3dna_ss(pdb_file)
        if not isinstance(ss, str) or not isinstance(seq_ss, str) or not seq_ss:
            raise ValueError('DSSR returned no usable sequence/secondary structure')
        seq_ss = seq_ss.strip().replace('&', '').upper().replace('T', 'U')
        ss = _checked_rna_ss(ss, len(seq_ss))
        mapped = map_ss_structure_to_reference(reference, seq_ss, ss)
        final_ss = _checked_rna_ss(mapped, len(reference))
        source = 'DSSR'
    except (OSError, subprocess.SubprocessError, ValueError, TypeError) as dssr_error:
        print(f'DSSR failed for {pdbcode}: {_ss_error_message(dssr_error)}; '
              'switching to MXfold2.', flush=True)
        try:
            record = SeqIO.read(str(Path(fa_file).expanduser()), 'fasta')
            fasta_seq = str(record.seq).upper().replace('T', 'U')
            if fasta_seq != reference:
                raise ValueError('MXfold2 FASTA sequence differs from the current RNA sequence')
            final_ss = _checked_rna_ss(get_mxfold2_ss(fa_file), len(reference))
        except (OSError, subprocess.SubprocessError, ValueError, TypeError) as mxfold_error:
            raise RuntimeError(
                f'RNA secondary structure failed for {pdbcode}. '
                f'DSSR: {_ss_error_message(dssr_error)}. '
                f'MXfold2: {_ss_error_message(mxfold_error)}. '
                'Check that mxfold2 predict can run in the active environment.'
            ) from mxfold_error
        source = 'MXfold2'
    # Replace the cache only after a complete, validated prediction is available.
    ss_path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=ss_path.parent,
                                         prefix=f'.{ss_path.name}.', delete=False) as handle:
            pending = Path(handle.name)
            handle.write(f'>{pdbcode} source={source}\n{reference}\n{final_ss}\n')
        pending.replace(ss_path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
    return final_ss

def get_mxfold2_ss(fa_file):
    """Run MXfold2 and parse its sequence and dot-bracket/score output."""
    fasta = Path(fa_file).expanduser().resolve()
    expected = str(SeqIO.read(str(fasta), 'fasta').seq).upper().replace('T', 'U')
    if not expected:
        raise ValueError('MXfold2 requires a nonempty RNA sequence')
    result = subprocess.run(['mxfold2', 'predict', str(fasta)], check=True,
                            capture_output=True, text=True)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    headers = [i for i, line in enumerate(lines) if line.startswith('>')]
    if len(headers) != 1 or len(lines) < headers[0] + 3:
        raise ValueError('MXfold2 did not return one complete FASTA/dot-bracket result')
    start = headers[0]
    predicted_seq = lines[start + 1].upper().replace('T', 'U')
    if predicted_seq != expected:
        raise ValueError('MXfold2 output sequence does not match its input FASTA')
    pred_ss = lines[start + 2].split()[0]  # Ignore the score printed after the structure.
    return _checked_rna_ss(pred_ss, len(expected))

def get_native_x3dna_ss(pdb_file, timeout=180):
    """Extract DSSR SS; propagate tool/JSON errors to the MXfold2 fallback."""
    pdb_path = Path(pdb_file).expanduser().resolve()
    # A fresh working directory prevents stale JSON and auxiliary-file collisions.
    with tempfile.TemporaryDirectory(prefix='rna_dssr_') as work:
        json_file = Path(work) / 'dssr.json'
        command = ['x3dna-dssr', f'-i={pdb_path}', '--json', f'-o={json_file}']
        subprocess.run(command, cwd=work, check=True, capture_output=True,
                       text=True, timeout=timeout)
        with json_file.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not isinstance(data.get('dbn'), dict):
            raise ValueError('DSSR JSON has no dbn record')
        all_chains = data['dbn'].get('all_chains')
        if not isinstance(all_chains, dict):
            raise ValueError('DSSR JSON has no all_chains record')
        ss, sequence = all_chains.get('sstr'), all_chains.get('bseq')
        if not isinstance(ss, str) or not isinstance(sequence, str) or not ss or not sequence:
            raise ValueError('DSSR JSON has no usable sequence/secondary structure')
        return ss, sequence

def map_ss_structure_to_reference(seq_ref, seq_target, ss_target):
    "Map a target dot-bracket structure onto seq_ref.\n\nseq_target and ss_target must have equal lengths. Missing reference\npositions receive '.', and the result has exactly len(seq_ref) symbols."
    # 1. Align globally with globalxx: matches score 1; mismatches and gaps score 0.
    if len(seq_target) != len(ss_target):
        raise ValueError('Target sequence and secondary structure lengths differ')
    if not seq_ref:
        return ''
    if not seq_target:
        return '.' * len(seq_ref)
    alignments = pairwise2.align.globalxx(seq_ref, seq_target, one_alignment_only=True)
    # Use the highest-scoring alignment.
    aln_ref, aln_target, score, begin, end = alignments[0]
    mapped_ss = []
    target_idx = 0  # Track the position in the original ss_target.
    # 2. Traverse both aligned sequences using aln_ref as the reference.
    for char_ref, char_target in zip(aln_ref, aln_target):
        # Case A: the reference has an alignment gap.
        # Skip positions absent from seq_ref while advancing the target index.
        if char_ref == "-":
            if char_target != "-":
                target_idx += 1
            continue
        # Case B: the reference has a base but the target has a gap.
        # Insert '.' for missing target positions.
        elif char_target == "-":
            mapped_ss.append(".")
        # Case C: both sequences have a base, matching or substituted.
        else:
            mapped_ss.append(ss_target[target_idx])
            target_idx += 1
    final_ss = "".join(mapped_ss)
    # 3. Validate the mapped length.
    if len(final_ss) != len(seq_ref):
        raise ValueError(f"补齐失败！生成长度 {len(final_ss)} 与参考长度 {len(seq_ref)} 不符。")
    return final_ss

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
                if not residue.has_id(target_atom):
                    if target_atom != 'C4\'' and residue.has_id('C5\''):
                        target_atom = 'C5\''
                    elif not residue.has_id('C5\'') and residue.has_id('C4'):
                        target_atom = 'C4'
                    # Fall back to the first atom if the requested atoms are missing.
                    else:
                        first_atom = list(residue.get_atoms())[0]
                        # print(first_atom)
                        target_atom = first_atom.get_name()
                if residue.has_id(target_atom):
                    coords.append(residue[target_atom].get_coord())
    # print(target_atom)
    # print(next(iter(residue)))
    # print(residue, target_atom, residue[target_atom].get_coord())
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
                if atom == 'N':
                    if resname in ['C', 'U', 'CYT', 'URA']:
                        target_atom = 'N1'
                    elif resname in ['A', 'G', 'ADE', 'GUA']:
                        target_atom = 'N9'
                if residue.has_id(target_atom):
                    coords.append(residue[target_atom].get_coord())
                else:
                    first_atom = list(residue.get_atoms())[0]
                    # print(first_atom)
                    target_atom = first_atom.get_name()
                    coords.append(residue[target_atom].get_coord())
    return np.array(coords)
