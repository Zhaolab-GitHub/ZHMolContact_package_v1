import torch.utils.data
from torch_geometric.data import Dataset
# from torch.utils.data import Dataset
import torch
import pandas as pd
from torch_geometric.data import Data
import pickle
import torch.utils.data
from copy import deepcopy
import numpy as np

class RNAProteinMoleculeDataset(Dataset):
    def __init__(self, sequence_data, RNA_obj, prot_obj, RP_conmap_truth_obj=None, device='cpu', cache_transform=True):
        super(RNAProteinMoleculeDataset, self).__init__()

        if isinstance(sequence_data, pd.core.frame.DataFrame):
            self.pairs = sequence_data
            # print(self.pairs)
        elif isinstance(sequence_data,str):
            self.pairs = pd.read_csv(sequence_data)
        else:
            raise Exception("provide dataframe object or csv path")

        ## RNAS
        if isinstance(RNA_obj, dict):
            self.RNAs = RNA_obj
        elif isinstance(RNA_obj, str):
            self.RNAs = torch.load(RNA_obj)
        else:
            raise Exception("provide dict RNA object or pickle path")

        ## PROTEINS
        if isinstance(prot_obj, dict):
            self.prots = prot_obj
        elif isinstance(prot_obj, str):
            self.prots = torch.load(prot_obj)
        else:
            raise Exception("provide dict protein object or pickle path")

        ## RP_CONMAP
        if isinstance(RP_conmap_truth_obj, dict):
            self.conmap_truths = RP_conmap_truth_obj
        elif isinstance(RP_conmap_truth_obj, str):
            self.conmap_truths = torch.load(RP_conmap_truth_obj)
        else:
            self.conmap_truths = None


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
        RNA_key = self.pairs.loc[idx, 'RNA_seq']
        prot_key = self.pairs.loc[idx, 'prot_seq']

        NR = len(RNA_key)
        NP = len(prot_key)

        pdbcode = self.pairs.loc[idx, 'PDB_code']
        pdbname = pdbcode
        RNA = self.RNAs[pdbcode]
        prot = self.prots[pdbcode]
        
        if self.conmap_truths is None:
            conmap_truth = None
            conmap_truth_origin = None
            RNA_site_truth = None
            prot_site_truth = None

        else:
            try:
                conmap_truth = self.conmap_truths[pdbcode]['conmap_truth']
                conmap_truth_origin = self.conmap_truths[pdbcode]['conmap_truth_origin']
                RNA_site_truth = torch.from_numpy(self.conmap_truths[pdbcode]['RNA_site_truth'])
                prot_site_truth = torch.from_numpy(self.conmap_truths[pdbcode]['prot_site_truth'])
                pdbname = self.conmap_truths[pdbcode]['pdbname']

            except KeyError:
                conmap_truth = None



        if self.cache_transform:
            ## RNA
            RNA_seq = RNA['seq']
            RNA_node_aa = RNA['seq_feat']
            RNA_node_evo = RNA['token_representation']
            RNA_num_nodes = RNA['num_nodes']
            RNA_node_pos = RNA['node_pos']
            RNA_edge_index = RNA['edge_index']
            RNA_edge_weight = RNA['edge_weight']
            # RNA_xyz_coord = RNA['xyz_coord']

            ## Prot
            prot_seq = prot['seq']
            prot_node_aa = prot['seq_feat']
            prot_node_evo = prot['token_representation']
            prot_num_nodes = prot['num_nodes']
            prot_node_pos = prot['node_pos']
            prot_edge_index = prot['edge_index']
            prot_edge_weight = prot['edge_weight']
            # prot_xyz_coord = prot['xyz_coord']
        else:
            ## RNA
            RNA_seq = RNA['seq']
            RNA_node_aa = RNA['seq_feat'].float()
            RNA_node_evo = RNA['token_representation'].float()
            RNA_num_nodes = len(RNA['seq'])
            RNA_node_pos = torch.arange(len(RNA['seq'])).reshape(-1,1)
            RNA_edge_index = RNA['edge_index']
            RNA_edge_weight = RNA['edge_weight'].float()

            ## prot
            prot_seq = prot['seq']
            prot_node_aa = prot['seq_feat'].float()
            prot_node_evo = prot['token_representation'].float()
            prot_num_nodes = len(prot['seq'])
            prot_node_pos = torch.arange(len(prot['seq'])).reshape(-1,1)
            prot_edge_index = prot['edge_index']
            prot_edge_weight = prot['edge_weight'].float()

        out = MultiGraphData(
                ## RNA
                RNA_node_aa=RNA_node_aa, RNA_node_evo=RNA_node_evo,
                RNA_node_pos=RNA_node_pos, RNA_seq=RNA_seq,
                RNA_edge_index=RNA_edge_index, RNA_edge_weight=RNA_edge_weight,
                RNA_num_nodes=RNA_num_nodes,
                # RNA_xyz_coord=RNA_xyz_coord,
                ## PROTEIN
                prot_node_aa=prot_node_aa, prot_node_evo=prot_node_evo,
                prot_node_pos=prot_node_pos, prot_seq=prot_seq,
                prot_edge_index=prot_edge_index, prot_edge_weight=prot_edge_weight,
                prot_num_nodes=prot_num_nodes,
                # prot_xyz_coord=prot_xyz_coord,
                ## Y output
                conmap_truth=conmap_truth,
                conmap_truth_origin=conmap_truth_origin,
                RNA_site_truth=RNA_site_truth,
                prot_site_truth=prot_site_truth,
                
                ## keys
                RNA_key = RNA_key, prot_key = prot_key, pdbname = pdbname,
        )

        return out

def maybe_num_nodes(index, num_nodes=None):
    # NOTE(WMF): I find out a problem here,
    # index.max().item() -> int
    # num_nodes -> tensor
    # need type conversion.
    # return index.max().item() + 1 if num_nodes is None else num_nodes
    return index.max().item() + 1 if num_nodes is None else int(num_nodes)

def get_self_loop_attr(edge_index, edge_attr, num_nodes):
    r"""Returns the edge features or weights of self-loops
    :math:`(i, i)` of every node :math:`i \in \mathcal{V}` in the
    graph given by :attr:`edge_index`. Edge features of missing self-loops not
    present in :attr:`edge_index` will be filled with zeros. If
    :attr:`edge_attr` is not given, it will be the vector of ones.

    .. note::
        This operation is analogous to getting the diagonal elements of the
        dense adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_weight = torch.tensor([0.2, 0.3, 0.5])
        >>> get_self_loop_attr(edge_index, edge_weight)
        tensor([0.5000, 0.0000])

        >>> get_self_loop_attr(edge_index, edge_weight, num_nodes=4)
        tensor([0.5000, 0.0000, 0.0000, 0.0000])
    """
    loop_mask = edge_index[0] == edge_index[1]
    loop_index = edge_index[0][loop_mask]

    if edge_attr is not None:
        loop_attr = edge_attr[loop_mask]
    else:  # A vector of ones:
        loop_attr = torch.ones_like(loop_index, dtype=torch.float)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    full_loop_attr = loop_attr.new_zeros((num_nodes, ) + loop_attr.size()[1:])
    full_loop_attr[loop_index] = loop_attr

    return full_loop_attr

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
