import torch.utils.data
from torch_geometric.data import Dataset
import torch
import pandas as pd
from torch_geometric.data import Data
import torch.utils.data
import numpy as np
from torch_geometric.data import Batch

from torch.nn.functional import pad


class RNAProteinMoleculeDataset(Dataset):
    def __init__(self, sequence_data, RNA_obj, prot_obj, RP_conmap_truth_obj=None, prior_dict=None, inter_residue_feat_dict=None, device='cpu', cache_transform=True):
        super(RNAProteinMoleculeDataset, self).__init__()

        if isinstance(sequence_data, pd.core.frame.DataFrame):
            self.pairs = sequence_data.reset_index(drop=True)
        elif isinstance(sequence_data,str):
            self.pairs = pd.read_csv(sequence_data)
        else:
            raise Exception("provide dataframe object or csv path")

        ## RNAS
        if isinstance(RNA_obj, dict):
            self.RNAs = {key: dict(value) for key, value in RNA_obj.items()}
        elif isinstance(RNA_obj, str):
            self.RNAs = torch.load(RNA_obj, map_location='cpu')
        else:
            raise Exception("provide dict RNA object or pickle path")

        ## PROTEINS
        if isinstance(prot_obj, dict):
            self.prots = {key: dict(value) for key, value in prot_obj.items()}
        elif isinstance(prot_obj, str):
            self.prots = torch.load(prot_obj, map_location='cpu')
        else:
            raise Exception("provide dict protein object or pickle path")

        ## RP_CONMAP
        if isinstance(RP_conmap_truth_obj, dict):
            self.conmap_truths = RP_conmap_truth_obj
        elif isinstance(RP_conmap_truth_obj, str):
            self.conmap_truths = torch.load(RP_conmap_truth_obj, map_location='cpu')
        else:
            self.conmap_truths = None


        ## prior_dict
        if isinstance(prior_dict, dict):
            self.prior_dicts = prior_dict
        elif isinstance(prior_dict, str):
            self.prior_dicts = torch.load(prior_dict, map_location='cpu')
        else:
            self.prior_dicts = None

        ## inter_residue_feat_dict
        if isinstance(inter_residue_feat_dict, dict):
            self.inter_residue_feats = inter_residue_feat_dict
        elif isinstance(inter_residue_feat_dict, str):
            self.inter_residue_feats = torch.load(inter_residue_feat_dict, map_location='cpu')
        else:
            self.inter_residue_feats = None


        self.device = device
        self.cache_transform = cache_transform

        if self.cache_transform:
            for _, v in self.prots.items():
                v['seq_feat'] = v['seq_feat'].float()
                v['token_representation'] = v['token_representation'].float()
                v['num_nodes'] = len(v['seq'])
                v['node_pos'] = torch.arange(len(v['seq'])).reshape(-1,1)
                v['edge_weight'] = v['edge_weight'].float()

            for _, v in self.RNAs.items():
                v['seq_feat'] = v['seq_feat'].float()
                v['token_representation'] = v['token_representation'].float()
                v['num_nodes'] = len(v['seq'])
                v['node_pos'] = torch.arange(len(v['seq'])).reshape(-1,1)
                v['edge_weight'] = v['edge_weight'].float()

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return self.__len__()
    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx):
        # Extract data
        # print(idx)
        RNA_key = self.pairs.loc[idx, 'RNA_seq']
        prot_key = self.pairs.loc[idx, 'prot_seq']

        NR = len(RNA_key)
        NP = len(prot_key)
        # print(self.pairs)
        pdbcode = self.pairs.loc[idx, 'PDB_code']
        # print(pdbcode)
        RNA = self.RNAs[pdbcode]
        prot = self.prots[pdbcode]
        if self.inter_residue_feats is None:
            inter_residue_feat_dict = None
        else:
            inter_residue_feat_dict = self.inter_residue_feats[pdbcode]


        conmap_truth = conmap_truth_strict = conmap_truth_origin = None
        conmap_truth_resize = conmap_truth_lbl = None
        RNA_site_truth = prot_site_truth = None
        RNA_pred_site = RNA.get('RNA_pred_site', RNA.get('pred_site'))
        prot_pred_site = prot.get('prot_pred_site', prot.get('pred_site'))

        elec_edge_feat = hbond_edge_feat = pi_pi_edge_feat = sasa_edge_feat = None
        if self.prior_dicts is not None:
            prior_dict = self.prior_dicts[pdbcode]
  
            RNA_pred_site = prior_dict.get('RNA_pred_site', RNA_pred_site)
            prot_pred_site = prior_dict.get('prot_pred_site', prot_pred_site)

            protein_domain_mask = prior_dict['protein_domain_mask']

            elec_edge_feat = prior_dict['elec_edge_feat']
            hbond_edge_feat = prior_dict['hbond_edge_feat']
            pi_pi_edge_feat = prior_dict['pi_pi_edge_feat']
            sasa_edge_feat = prior_dict['sasa_edge_feat']
            inter_residue_feat = np.concatenate([elec_edge_feat, hbond_edge_feat, pi_pi_edge_feat, sasa_edge_feat], axis=2)
        else:
            prior_dict = None
            
        if self.conmap_truths is not None:
            truth = self.conmap_truths[pdbcode]
            conmap_truth = truth.get('conmap_truth')
            conmap_truth_strict = truth.get('conmap_truth_strict')
            conmap_truth_origin = truth.get('conmap_truth_origin')
            conmap_truth_resize = truth.get('conmap_truth_resize')
            if conmap_truth_resize is not None:
                conmap_truth_resize = torch.as_tensor(conmap_truth_resize)
            conmap_truth_lbl = truth.get('conmap_truth_lbl')
            RNA_site_truth = truth.get('RNA_site_truth')
            prot_site_truth = truth.get('prot_site_truth')
            if self.prior_dicts is None:
                RNA_pred_site = truth.get('RNA_pred_site', RNA_pred_site)
                prot_pred_site = truth.get('prot_pred_site', prot_pred_site)

        if self.inter_residue_feats is None:
            if self.prior_dicts is None:
                inter_residue_feat = None   
        else:
            
            elec_edge_feat = inter_residue_feat_dict['elec_edge_feat']
            hbond_edge_feat = inter_residue_feat_dict['hbond_edge_feat']
            pi_pi_edge_feat = inter_residue_feat_dict['pi_pi_edge_feat']
            sasa_edge_feat = inter_residue_feat_dict['sasa_edge_feat']
            inter_residue_feat = np.concatenate([elec_edge_feat, hbond_edge_feat, pi_pi_edge_feat, sasa_edge_feat], axis=2)
            # print(combined_edge_feat.shape)


        if self.cache_transform:
            ## RNA
            RNA_seq = RNA['seq']
            RNA_node_aa = RNA['seq_feat']
            RNA_node_evo = RNA['token_representation']
            RNA_num_nodes = RNA['num_nodes']
            RNA_node_pos = RNA['node_pos']
            RNA_edge_index = RNA['edge_index']
            RNA_edge_weight = RNA['edge_weight']
            # print('RNA_edge_index: ', RNA_edge_index.shape)
            # print('RNA_edge_weight: ', RNA_edge_weight.shape)
            # if RNA_edge_index.shape[1] != RNA_edge_weight.shape[0]:
            #     print(RNA_seq)
            # RNA_domain_mask = RNA['RNA_domain_mask']
            # RNA_domain_mask = RNA_domain_mask['node_mask'] - 1

            ## Prot
            prot_seq = prot['seq']
            prot_node_aa = prot['seq_feat']
            prot_node_evo = prot['token_representation']
            prot_num_nodes = prot['num_nodes']
            prot_node_pos = prot['node_pos']
            prot_edge_index = prot['edge_index']
            prot_edge_weight = prot['edge_weight']
            if prior_dict is None:
                protein_domain_mask = prot['protein_domain_mask']


        else:
            ## RNA
            RNA_seq = RNA['seq']
            RNA_node_aa = RNA['seq_feat'].float()
            RNA_node_evo = RNA['token_representation'].float()
            RNA_num_nodes = len(RNA['seq'])
            RNA_node_pos = torch.arange(len(RNA['seq'])).reshape(-1,1)
            RNA_edge_index = RNA['edge_index']
            RNA_edge_weight = RNA['edge_weight'].float()
            # RNA_domain_mask = RNA['RNA_domain_mask']

            ## prot
            prot_seq = prot['seq']
            prot_node_aa = prot['seq_feat'].float()
            prot_node_evo = prot['token_representation'].float()
            prot_num_nodes = len(prot['seq'])
            prot_node_pos = torch.arange(len(prot['seq'])).reshape(-1,1)
            prot_edge_index = prot['edge_index']
            prot_edge_weight = prot['edge_weight'].float()
            # prot_domain_mask = prot['prot_domain_mask']


        # Coordinates and domain assignments are needed with either cache mode.
        RNA_xyz_coord = RNA['xyz_coord']
        RNA_xyz_coord_P = RNA['xyz_coord_P']
        RNA_xyz_coord_C4 = RNA['xyz_coord_C4']
        RNA_xyz_coord_N = RNA['xyz_coord_N']
        prot_xyz_coord = prot['xyz_coord']
        prot_xyz_coord_CA = prot['xyz_coord_CA']
        prot_xyz_coord_C = prot['xyz_coord_C']
        prot_xyz_coord_N = prot['xyz_coord_N']


        # Preserve variable-size targets as per-graph arrays in PyG batches.
        def as_array(value):
            return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value

        conmap_truth = as_array(conmap_truth)
        conmap_truth_strict = as_array(conmap_truth_strict)
        conmap_truth_origin = as_array(conmap_truth_origin)
        conmap_truth_lbl = as_array(conmap_truth_lbl)
        RNA_site_truth = as_array(RNA_site_truth)
        prot_site_truth = as_array(prot_site_truth)
        RNA_pred_site = as_array(RNA_pred_site)
        prot_pred_site = as_array(prot_pred_site)

        out = MultiGraphData(
                ## RNA
                RNA_node_aa=RNA_node_aa, RNA_node_evo=RNA_node_evo,
                RNA_node_pos=RNA_node_pos, RNA_seq=RNA_seq,
                RNA_edge_index=RNA_edge_index, RNA_edge_weight=RNA_edge_weight,
                RNA_num_nodes=RNA_num_nodes,
                RNA_xyz_coord=RNA_xyz_coord,
                RNA_xyz_coord_P=RNA_xyz_coord_P,
                RNA_xyz_coord_C4=RNA_xyz_coord_C4,
                RNA_xyz_coord_N=RNA_xyz_coord_N,
                # RNA_domain_mask=RNA_domain_mask,

                ## PROTEIN
                prot_node_aa=prot_node_aa, prot_node_evo=prot_node_evo,
                prot_node_pos=prot_node_pos, prot_seq=prot_seq,
                prot_edge_index=prot_edge_index, prot_edge_weight=prot_edge_weight,
                prot_num_nodes=prot_num_nodes,
                prot_xyz_coord=prot_xyz_coord,
                prot_xyz_coord_CA=prot_xyz_coord_CA,
                prot_xyz_coord_C=prot_xyz_coord_C,
                prot_xyz_coord_N=prot_xyz_coord_N,
                # prot_domain_mask=prot_domain_mask,
                protein_domain_mask = protein_domain_mask,
                # protein_domain_lbl = protein_domain_lbl,

                ## Y output
                conmap_truth=conmap_truth,
                conmap_truth_strict=conmap_truth_strict,
                conmap_truth_origin=conmap_truth_origin,
                conmap_truth_resize=conmap_truth_resize,
                conmap_truth_lbl=conmap_truth_lbl,
                RNA_site_truth=RNA_site_truth,
                prot_site_truth=prot_site_truth,
                # Domain_contact=Domain_contact, 
                RNA_pred_site=RNA_pred_site,
                prot_pred_site=prot_pred_site,

                ## inter-residue features
                elec_edge_feat=elec_edge_feat,
                hbond_edge_feat=hbond_edge_feat,
                pi_pi_edge_feat=pi_pi_edge_feat,
                sasa_edge_feat=sasa_edge_feat,
                inter_residue_feat=inter_residue_feat,

                ## keys
                RNA_key = RNA_key, prot_key = prot_key, pdbname = pdbcode,
        )

        return out

class MultiGraphData(Data):
    def __inc__(self, key, item, *args):
        if key == 'RNA_edge_index':
            return self.RNA_node_aa.size(0)
        elif key == 'RNA_struc_edge_index':
            return self.RNA_node_aa.size(0)
        elif key == 'prot_edge_index':
            return self.prot_node_aa.size(0)
        elif key == 'prot_struc_edge_index':
            return self.prot_node_aa.size(0)
        elif key == 'm2p_edge_index':
             return torch.tensor([[self.RNA_node_aa.size(0)], [self.prot_node_aa.size(0)]])
        # elif key == 'edge_index_p2m':
        #     return torch.tensor([[self.prot_node_s.size(0)],[self.mol_x.size(0)]])
        else:
            return super(MultiGraphData, self).__inc__(key, item, *args)
        


def pad_collate(batch):
    if not batch:
        raise ValueError('Cannot collate an empty batch')
    data_list = [item for item in batch]
    label_list = [torch.as_tensor(item.conmap_truth_lbl) for item in batch]  # Each label has shape [M, N].

    # Merge graph structures using PyG Batch.
    batch1 = Batch.from_data_list(data_list, exclude_keys=['conmap_truth_lbl'])
    batch1.conmap_truth_lbl = [item.conmap_truth_lbl for item in batch]

    # Find the maximum dimensions.
    M_max = max(l.shape[0] for l in label_list)
    N_max = max(l.shape[1] for l in label_list)

    padded_labels = []
    for label in label_list:
        M, N = label.shape
        padded = pad(label, (0, N_max - N, 0, M_max - M), value=-100)
        padded_labels.append(padded)

    labels = torch.stack(padded_labels)  # [B, M_max, N_max]
    return batch1, labels
