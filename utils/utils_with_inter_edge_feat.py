import os
import numpy as np
import sys

# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

    
from itertools import repeat
import pandas as pd 


import torch

from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.utils import degree


import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Dataset, DataLoader


class InfiniteDataLoader(GeoDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.Domain_contact
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

def create_custom_loader(type='epoch'):
    if type == 'epoch':
        return DataLoader
    elif type =='infinite':
        return InfiniteDataLoader
    else:
        raise Exception('Not Implemented')
        
class CustomWeightedRandomSampler(torch.utils.data.WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

def sampler_from_weights(weights):
    sampler = CustomWeightedRandomSampler(weights, len(weights), replacement=True)
    
    return sampler 
def create_custom_sampler(class_list, specified_weight={}):
    assert isinstance(specified_weight,dict)
    class_list = np.array(class_list)
    class_weight = {
         t: 1./len(np.where(class_list == t)[0]) for t in np.unique(class_list)
    }
    
    samples_weight = np.array([class_weight[t] for t in class_list])
    
    if specified_weight:
        specified_weight = np.array([specified_weight[i] for i in class_list])
        samples_weight *= specified_weight
        
    sampler = CustomWeightedRandomSampler(samples_weight, len(samples_weight))
    
    return sampler 

def compute_pna_degrees(train_loader):
    RNA_max_degree = -1
    prot_max_degree = -1

    for data in tqdm(train_loader):

        # RNA
        RNA_d = degree(data.RNA_edge_index[1], num_nodes=data.RNA_node_aa.shape[0], dtype=torch.long)
        RNA_max_degree = max(RNA_max_degree, int(RNA_d.max()))

        # protein
        prot_d = degree(data.prot_edge_index[1], num_nodes=data.prot_node_aa.shape[0], dtype=torch.long)
        prot_max_degree = max(prot_max_degree, int(prot_d.max()))

    # Compute the in-degree histogram tensor
    RNA_deg = torch.zeros(RNA_max_degree + 1, dtype=torch.long)
    prot_deg = torch.zeros(prot_max_degree + 1, dtype=torch.long)

    for data in tqdm(train_loader):

        # RNA
        RNA_d = degree(data.RNA_edge_index[1], num_nodes=data.RNA_node_aa.shape[0], dtype=torch.long)
        RNA_deg += torch.bincount(RNA_d, minlength=RNA_deg.numel())

        # Protein
        prot_d = degree(data.prot_edge_index[1], num_nodes=data.prot_node_aa.shape[0], dtype=torch.long)
        prot_deg += torch.bincount(prot_d, minlength=prot_deg.numel())

    return RNA_deg, prot_deg


def unbatch(src, batch, dim: int = 0):
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)

    :rtype: :class:`List[Tensor]`

    Example:

        >>> src = torch.arange(7)
        >>> batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
        >>> unbatch(src, batch)
        (tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


def unbatch_nodes(data_tensor, index_tensor):
    """
    Unbatch a data tensor based on an index tensor.

    Args:
    data_tensor (torch.Tensor): The tensor to be unbatched.
    index_tensor (torch.Tensor): A tensor of the same length as data_tensor's first dimension, 
                                 indicating the batch index for each element in data_tensor.

    Returns:
    list[torch.Tensor]: A list of tensors, where each tensor corresponds to a separate batch.
    """
    return [data_tensor[index_tensor == i] for i in index_tensor.unique()]


def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

def printline(line):
    sys.stdout.write(line + "\x1b[K\r")
    sys.stdout.flush()


def protein_degree_from_dict(protein_dict):
    protein_max_degree = -1 
    for k, v in protein_dict.items():
        node_num = len(v['seq'])
        edge_index = v['edge_index']
        protein_degree = degree(edge_index[1], num_nodes=node_num, dtype=torch.long)
        protein_max_degree = max(protein_max_degree, protein_degree.max())

    protein_deg = torch.zeros(protein_max_degree + 1, dtype=torch.long)
    for k, v in protein_dict.items():
        node_num = len(v['seq'])
        edge_index = v['edge_index']
        protein_degree = degree(edge_index[1], num_nodes=node_num, dtype=torch.long)
        protein_deg += torch.bincount(protein_degree, minlength=protein_deg.numel())   

    return protein_deg


def ligand_degree_from_dict(ligand_dict):
    mol_max_degree = -1 
    clique_max_degree = -1 
    
    for k, v in tqdm(ligand_dict.items()):
         # mol
        mol_x = v['atom_feature']
        adj = v['bond_feature']
        mol_edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        mol_d = degree(mol_edge_index[1], num_nodes=mol_x.shape[0], dtype=torch.long)
        mol_max_degree = max(mol_max_degree, int(mol_d.max()))
        # clique
        clique_x = v['x_clique']
        clique_edge_index = v['tree_edge_index'].long()
        clique_d = degree(clique_edge_index[1], num_nodes=clique_x.shape[0], dtype=torch.long)
        clique_max_degree = max(clique_max_degree, int(clique_d.max()))
    
    mol_deg = torch.zeros(mol_max_degree + 1, dtype=torch.long)
    clique_deg = torch.zeros(clique_max_degree + 1, dtype=torch.long)

    for k, v in tqdm(ligand_dict.items()):
        # mol
        mol_x = v['atom_feature']
        adj = v['bond_feature']
        mol_edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        mol_d = degree(mol_edge_index[1], num_nodes=mol_x.shape[0], dtype=torch.long)
        
        mol_deg += torch.bincount(mol_d, minlength=mol_deg.numel())
        # clique
        clique_x = v['x_clique']
        clique_edge_index = v['tree_edge_index'].long()
        clique_d = degree(clique_edge_index[1], num_nodes=clique_x.shape[0], dtype=torch.long)
        clique_deg += torch.bincount(clique_d, minlength=clique_deg.numel())
       
    return mol_deg, clique_deg


def minmax_norm(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return arr.astype(float)
    span = arr.max() - arr.min()
    return (arr - arr.min()) / span if span != 0 else np.zeros_like(arr, dtype=float)

def percentile_rank(arr):
    if len(arr) < 2:
        return np.zeros(len(arr), dtype=float)
    return np.argsort(np.argsort(arr)) / (len(arr)-1)

# from rdkit import Chem
# from rdkit.Chem import PropertyPickleOptions
# import pickle
        
def store_ligand_score(ligand_smiles, atom_types, atom_scores, ligand_path):
    # Create a molecule from a SMILES string
    mol = Chem.MolFromSmiles(ligand_smiles)
    # Add an atom-level property to the first atom
    for i, atom in enumerate(mol.GetAtoms()):
        
        if atom_types[i] == atom.GetSymbol():
            atom.SetProp("PSICHIC_Atom_Score", str(atom_scores[i]))
        else:
            return False
    # Configure RDKit to pickle all properties
    Chem.SetDefaultPickleProperties(PropertyPickleOptions.AllProps)

    # Serialize molecule to a pickle file
    with open(ligand_path, 'wb') as f:
        pickle.dump(mol, f)

    return True
    
def store_result(df, attention_dict, interaction_keys,  ligand_dict, 
                 reg_pred=None, cls_pred=None, mcls_pred=None, 
                 result_path='', save_interpret=True):
    if save_interpret:
        unbatched_residue_score = unbatch(attention_dict['residue_final_score'],attention_dict['protein_residue_index'])
        unbatched_atom_score = unbatch(attention_dict['atom_final_score'], attention_dict['drug_atom_index'])
        unbatched_residue_layer_score = unbatch(attention_dict['residue_layer_scores'],attention_dict['protein_residue_index'])
        unbatched_clique_layer_score = unbatch(attention_dict['clique_layer_scores'], attention_dict['drug_clique_index'])
    
    for idx, key in enumerate(interaction_keys):
        matching_row = (df['Protein'] == key[0]) & (df['Ligand'] == key[1])
        if reg_pred is not None:
            if 'predicted_binding_affinity' in df.columns:
                df.loc[matching_row, 'predicted_binding_affinity'] = reg_pred[idx]
            else:
                df['predicted_binding_affinity'] = None
                df.loc[matching_row, 'predicted_binding_affinity'] = reg_pred[idx]
        if cls_pred is not None:
            if 'predicted_binary_interaction' in df.columns:
                df.loc[matching_row, 'predicted_binary_interaction'] = cls_pred[idx]
            else:
                df['predicted_binary_interaction'] = None
                df.loc[matching_row, 'predicted_binary_interaction'] = cls_pred[idx]

        if mcls_pred is not None:
            if 'predicted_antagonist' in df.columns and 'predicted_nonbinder' in df.columns and 'predicted_agonist' in df.columns:
                df.loc[matching_row, ['predicted_antagonist','predicted_nonbinder','predicted_agonist']] = mcls_pred[idx].tolist()
            else:
                df['predicted_antagonist'] = None
                df['predicted_nonbinder'] = None
                df['predicted_agonist'] = None
                df.loc[matching_row, ['predicted_antagonist','predicted_nonbinder','predicted_agonist']] = mcls_pred[idx].tolist()
        

        if all([idx in attention_dict['cluster_s'] for idx in range(3)]):
            unbatched_cluster_s0 = unbatch_nodes(attention_dict['cluster_s'][0].softmax(dim=-1), attention_dict['protein_residue_index'])
            unbatched_cluster_s1 = unbatch_nodes(attention_dict['cluster_s'][1].softmax(dim=-1), attention_dict['protein_residue_index'])
            unbatched_cluster_s2 = unbatch_nodes(attention_dict['cluster_s'][2].softmax(dim=-1), attention_dict['protein_residue_index'])

        if save_interpret:
            for pair_id in df[matching_row]['ID']:
                pair_path = os.path.join(result_path,pair_id)
                if not os.path.exists(pair_path):
                    os.makedirs(pair_path)
                ## STORE Protein Interpretation
                protein_interpret = pd.DataFrame({
                    'Residue_Type':list(key[0]),
                    'PSICHIC_Residue_Score':minmax_norm(unbatched_residue_score[idx].cpu().flatten().numpy())
                    })

                protein_interpret['Residue_ID'] = protein_interpret.index + 1
                protein_interpret['PSICHIC_Residue_Percentile'] = percentile_rank(protein_interpret['PSICHIC_Residue_Score'])
                protein_interpret = protein_interpret[['Residue_ID','Residue_Type','PSICHIC_Residue_Score','PSICHIC_Residue_Percentile']]

                if all([id_ in attention_dict['cluster_s'] for id_ in range(3)]):
                    for ci in range(5):
                        protein_interpret['Layer0_Cluster'+str(ci)] = unbatched_cluster_s0[idx][:,ci].cpu().flatten().numpy()

                    for ci in range(10):
                        protein_interpret['Layer1_Cluster'+str(ci)] = unbatched_cluster_s1[idx][:,ci].cpu().flatten().numpy()
                    
                    for ci in range(20):
                        protein_interpret['Layer2_Cluster'+str(ci)] = unbatched_cluster_s2[idx][:,ci].cpu().flatten().numpy()

                protein_interpret.to_csv(os.path.join(pair_path,'protein.csv'),index=False)

                ## STORE Ligand Interpretation
                ligand_path = os.path.join(pair_path,'ligand.pkl')

                successful_ligand = store_ligand_score(key[1], ligand_dict[key[1]]['atom_types'].split('|'), 
                                                       minmax_norm(unbatched_atom_score[idx].cpu().flatten().numpy()),
                                                       ligand_path)
                if not successful_ligand:
                    print('Ligand Intepretation for {} failed due to not matching atom order.'.format(pair_id))
                ## STORE Fingerprint
                np.save(os.path.join(pair_path,'fingerprint.npy'),
                        attention_dict['interaction_fingerprint'][idx].detach().cpu().numpy()
                        )
    return df

def virtual_screening(screen_df, model, data_loader, result_path, save_interpret=True, ligand_dict=None, device='cpu',
                        save_cluster=False):
    if "ID" in screen_df.columns:
        # Iterate through the DataFrame check any empty pairs 
        for i, row in screen_df.iterrows():
            if pd.isna(row['ID']):
                screen_df.at[i, 'ID'] = f"PAIR_{i}"
    else:
        screen_df['ID'] = 'PAIR_' 
        screen_df['ID'] += screen_df.index.astype(str)
    reg_preds = []
    cls_preds = []
    mcls_preds = []

    model.eval()
    
    with torch.no_grad():
        for data in tqdm(data_loader):
            data = data.to(device)
            reg_pred, cls_pred, mcls_pred, sp_loss, o_loss, cl_loss, attention_dict = model(
                    # Molecule
                    mol_x=data.mol_x, mol_x_feat=data.mol_x_feat, bond_x=data.mol_edge_attr,
                    atom_edge_index=data.mol_edge_index, clique_x=data.clique_x, 
                    clique_edge_index=data.clique_edge_index, atom2clique_index=data.atom2clique_index,
                    # Protein
                    residue_x=data.prot_node_aa, residue_evo_x=data.prot_node_evo,
                    residue_edge_index=data.prot_edge_index,
                    residue_edge_weight=data.prot_edge_weight,
                    # Mol-Protein Interaction batch
                    mol_batch=data.mol_x_batch, prot_batch=data.prot_node_aa_batch, clique_batch=data.clique_x_batch,
                    # save_cluster
                    save_cluster=save_cluster
            )
            interaction_keys = list(zip(data.prot_key, data.mol_key))

            if reg_pred is not None:
                reg_pred = reg_pred.squeeze().reshape(-1).cpu().numpy()
                reg_preds.append(reg_pred)
                
            if cls_pred is not None:
                cls_pred = torch.sigmoid(cls_pred).squeeze().reshape(-1).cpu().numpy()
                cls_preds.append(cls_pred)

            if mcls_pred is not None:
                mcls_pred = torch.softmax(mcls_pred,dim=-1).cpu().numpy()
                mcls_preds.append(mcls_pred)

            screen_df = store_result(screen_df, attention_dict, interaction_keys, ligand_dict, 
                                     reg_pred, cls_pred, mcls_pred, 
                                     result_path=result_path, save_interpret = save_interpret)

    return screen_df



def pad_conmap_truth_lbl(data_batch, mode="eval"):
    """blind: pad input features; eval: additionally pad ground-truth labels."""
    if mode not in {"blind", "eval"}:
        raise ValueError("mode must be 'blind' or 'eval'")
    rna_lengths = [int(n) for n in data_batch.RNA_num_nodes]
    prot_lengths = [int(n) for n in data_batch.prot_num_nodes]
    if not rna_lengths or len(rna_lengths) != len(prot_lengths):
        raise ValueError("RNA and protein graph counts must match and be nonempty")
    max_rows, max_cols = max(rna_lengths), max(prot_lengths)

    def pad_field(name, pair=False, site_lengths=None, fill=-100):
        values = getattr(data_batch, name, None)
        if values is None:
            return
        if len(values) != len(rna_lengths):
            raise ValueError(f"{name}: expected one array per graph")
        padded = []
        for i, value in enumerate(values):
            matrix = torch.as_tensor(value)
            if site_lengths is not None:
                # Site arrays: [N] or [N, C]
                if matrix.ndim == 1:
                    matrix = matrix.unsqueeze(-1)
                if matrix.ndim != 2 or matrix.shape[0] != site_lengths[i]:
                    raise ValueError(
                        f"{name}: site length does not match graph {i}: "
                        f"{tuple(matrix.shape)}"
                    )
                padding = (0, 0, 0, max(site_lengths) - matrix.shape[0])
            else:
                # Contact labels: [H, W]; pair features: [H, W, C]
                rank = 3 if pair else 2
                expected_shape = (rna_lengths[i], prot_lengths[i])
                if matrix.ndim != rank or tuple(matrix.shape[:2]) != expected_shape:
                    raise ValueError(
                        f"{name}: pair shape does not match graph {i}: "
                        f"{tuple(matrix.shape)}"
                    )
                padding = (
                    0, max_cols - matrix.shape[1],
                    0, max_rows - matrix.shape[0]
                )
                if pair:
                    padding = (0, 0) + padding
            # Ensure integer inputs support negative padding.
            if fill < 0 and not matrix.is_floating_point():
                matrix = matrix.long()
            padded.append(F.pad(matrix, padding, value=fill))
        setattr(data_batch, name + "_pad", torch.stack(padded))

    # Input priors and features used in both modes.
    pad_field("RNA_pred_site", site_lengths=rna_lengths)
    pad_field("prot_pred_site", site_lengths=prot_lengths)
    pad_field("inter_residue_feat", pair=True, fill=0)

    # Ground-truth labels used only for evaluation/training.
    if mode == "eval":
        pad_field("conmap_truth_lbl")
        pad_field("conmap_truth_strict")
        pad_field("RNA_site_truth", site_lengths=rna_lengths)
        pad_field("prot_site_truth", site_lengths=prot_lengths)
    return data_batch



def sample_rigid_transform(device, dtype=torch.float32, translate_std=5.0, rotate_std=2*torch.pi):
    'Sample a rigid transform R [3, 3] and t [1, 3].\n\nrotate_std controls rotation-vector spread in radians; translate_std\ncontrols the translation spread.'
    # 1) Sample a rotation vector; rotate_std controls the perturbation in radians.
    rot_vec = torch.randn(3, device=device, dtype=dtype) * rotate_std
    theta = torch.norm(rot_vec)

    if theta < 1e-8:
        R = torch.eye(3, device=device, dtype=dtype)
    else:
        k = rot_vec / theta
        kx, ky, kz = k[0], k[1], k[2]

        K = torch.tensor([
            [0,   -kz,  ky],
            [kz,   0,  -kx],
            [-ky, kx,   0]
        ], device=device, dtype=dtype)

        I = torch.eye(3, device=device, dtype=dtype)
        R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

    # 2) Sample a translation.
    t = torch.randn(1, 3, device=device, dtype=dtype) * translate_std

    return R, t


def apply_rigid_transform(coords, R, t, center=None):
    'Apply the same rigid transform to coords [N, 3] around center [1, 3].\n\nR has shape [3, 3] and t has shape [1, 3]. If center is None, use the\ncoordinate mean. Return centered, rotated coordinates plus t.'
    if center is None:
        center = coords.mean(dim=0, keepdim=True)

    coords_centered = coords - center
    coords_transformed = coords_centered @ R.T + t
    return coords_transformed


def build_rna_local_frame(P, C4, N, eps=1e-8):
    """
    P, C4, N: [L, 3]
    return:
        R: [L, 3, 3]   # columns = x, y, z
        t: [L, 3]      # origin = C4'
    """
    # x-axis: C4' -> N
    # print(N, C4)
    ex = N - C4
    ex = ex / (torch.linalg.norm(ex, dim=-1, keepdim=True) + eps)

    # helper vector: C4' -> P
    v = P - C4

    # y-axis: orthogonalize v against ex
    v_proj = (v * ex).sum(dim=-1, keepdim=True) * ex
    ey = v - v_proj
    ey = ey / (torch.linalg.norm(ey, dim=-1, keepdim=True) + eps)

    # z-axis: right-handed frame
    ez = torch.cross(ex, ey, dim=-1)
    ez = ez / (torch.linalg.norm(ez, dim=-1, keepdim=True) + eps)

    # re-orthogonalize ey
    ey = torch.cross(ez, ex, dim=-1)
    ey = ey / (torch.linalg.norm(ey, dim=-1, keepdim=True) + eps)

    # rotation matrix, columns are basis vectors
    R = torch.stack([ex, ey, ez], dim=-1)   # [L, 3, 3]
    t = C4                                  # [L, 3]

    return R, t

def rigid_transform_perturb_batch(data_batch, translate_std=5.0, rotate_std=2*torch.pi, seed=None, oprate_trans=True):
    'Apply one rigid perturbation to all RNA atom sets in the batch.\n\nPreserve relative geometry among RNA_xyz_coord_P, RNA_xyz_coord_C4\nand RNA_xyz_coord_N, then rebuild RNA and protein local frames.'
    device = data_batch.RNA_xyz_coord.device
    dtype = data_batch.RNA_xyz_coord.dtype

    # Use a common center for every atom set.
    # Rotate all atom sets about the same center to preserve their relative geometry.

    if oprate_trans:
        if seed is None:
            R, t = sample_rigid_transform(device, dtype, translate_std, rotate_std)
        else:
            # Seed this augmentation without resetting the caller's random stream.
            devices = [device] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                R, t = sample_rigid_transform(device, dtype, translate_std, rotate_std)
        center = data_batch.RNA_xyz_coord.mean(dim=0, keepdim=True)
        data_batch.RNA_xyz_coord = apply_rigid_transform(data_batch.RNA_xyz_coord, R, t, center)
        data_batch.RNA_xyz_coord_P = apply_rigid_transform(data_batch.RNA_xyz_coord_P, R, t, center)
        data_batch.RNA_xyz_coord_C4 = apply_rigid_transform(data_batch.RNA_xyz_coord_C4, R, t, center)
        data_batch.RNA_xyz_coord_N = apply_rigid_transform(data_batch.RNA_xyz_coord_N, R, t, center)


    #=========== build local frame system =============

    RNA_local_R, RNA_local_t = build_rna_local_frame(data_batch.RNA_xyz_coord_P, data_batch.RNA_xyz_coord_C4, data_batch.RNA_xyz_coord_N)
    data_batch.RNA_local_R = RNA_local_R
    data_batch.RNA_local_t = RNA_local_t

    prot_local_R, prot_local_t = build_rna_local_frame(data_batch.prot_xyz_coord_CA, data_batch.prot_xyz_coord_C, data_batch.prot_xyz_coord_N)
    data_batch.prot_local_R = prot_local_R
    data_batch.prot_local_t = prot_local_t

    return data_batch
