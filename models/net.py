### import package
import torch
from models.layers import MLP, RNA_PNAConv, Protein_PNAConv, RNAProteinConv, GCNCluster, PosLinear, dropout_edge
from torch_geometric.nn import global_add_pool
from torch.nn import Linear
from torch_geometric.utils import degree
import torch.nn.functional as F

import numpy as np
import scipy.sparse as sp
from copy import deepcopy
## for cluster
from torch_geometric.utils import to_dense_adj, to_dense_batch, degree, subgraph
from models.protein_pool import dense_mincut_pool
## for cluster
from torch_geometric.nn.norm import GraphNorm
import torch_geometric

#########################################################################

class net(torch.nn.Module):
    def __init__(self, RNA_deg, prot_deg,
                 # MOLECULE
                 RNA_in_channels=21, RNA_evo_channels=640, prot_in_channels=33, prot_evo_channels=1280,
                 hidden_channels=100,
                 pre_layers=2, post_layers=1,
                 aggregators=['mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'linear'],
                 # interaction
                 total_layer=3,
                 K = [5,10,20],
                 t = 1,
                 # training
                 heads=5,
                 dropout=0.1,
                 gaussian_noise=0,
                 num_class=15,
                 device='cuda:0'):
        super(net, self).__init__()

        self.dropout = dropout
        self.drop_nuc = dropout
        self.drop_residue = dropout
        self.gaussian_noise = gaussian_noise
        self.dropout_attn_score = dropout
        self.dropout_cluster_edge = dropout
        self.device = device
        
        self.total_layer = total_layer

        # RNA IN FEAT
        self.RNA_evo = MLP([RNA_evo_channels, hidden_channels * 2, hidden_channels], out_norm=True)
        self.RNA_aa = MLP([RNA_in_channels, hidden_channels * 2, hidden_channels], out_norm=True)

        # PROTEIN IN FEAT
        self.prot_evo = MLP([prot_evo_channels, hidden_channels * 2, hidden_channels], out_norm=True)
        self.prot_aa = MLP([prot_in_channels, hidden_channels * 2, hidden_channels], out_norm=True)

        ### RNA and PROTEIN
        self.RNA_convs = torch.nn.ModuleList()
        self.prot_convs = torch.nn.ModuleList()

        self.RNA_gn2 = torch.nn.ModuleList()
        self.prot_gn2 = torch.nn.ModuleList()

        self.inter_convs = torch.nn.ModuleList()

        # self.atten_linear = torch.nn.ModuleList()

        self.heads =heads

        self.num_cluster = K
        self.t = t
        self.RNA_cluster = torch.nn.ModuleList()
        self.prot_cluster = torch.nn.ModuleList()

        self.RNA_norms = torch.nn.ModuleList()
        self.prot_norms = torch.nn.ModuleList()

        self.nuc_lins = torch.nn.ModuleList()
        self.residue_lins = torch.nn.ModuleList()

        self.c2n_mlps = torch.nn.ModuleList()
        self.c2r_mlps = torch.nn.ModuleList()

        self.RNA_score_lins = torch.nn.ModuleList()
        self.prot_score_lins = torch.nn.ModuleList()


        # self.conLinear = torch.nn.ModuleList()

        self.total_layer = total_layer
        self.RNA_edge_dim = hidden_channels
        self.prot_edge_dim = hidden_channels

        self.num_class = num_class
        for idx in range(total_layer):
            self.RNA_convs.append(RNA_PNAConv(
                RNA_deg, hidden_channels,edge_channels=hidden_channels,
                pre_layers=pre_layers, post_layers=post_layers,
                aggregators=aggregators,
                scalers=scalers,
                num_towers=heads,
                dropout=self.dropout
            ))

            self.prot_convs.append(Protein_PNAConv(
                prot_deg, hidden_channels, edge_channels=hidden_channels, # None,
                pre_layers=pre_layers, post_layers=post_layers,
                aggregators=aggregators,
                scalers=scalers,
                num_towers=heads,
                dropout=self.dropout
            ))


            self.RNA_cluster.append(GCNCluster([hidden_channels, hidden_channels*2, self.num_cluster[idx]], in_norm=True))
            self.prot_cluster.append(GCNCluster([hidden_channels, hidden_channels*2, self.num_cluster[idx]], in_norm=True))


            self.inter_convs.append(RNAProteinConv(
                nuc_channels=hidden_channels,
                residue_channels=hidden_channels,
                heads=heads,
                t=t,
                dropout_attn_score=self.dropout_attn_score
            ))

            self.RNA_score_lins.append(Linear(self.num_cluster[idx], 1, bias=False))
            self.prot_score_lins.append(Linear(self.num_cluster[idx], 1, bias=False))

            self.RNA_norms.append(torch.nn.LayerNorm(hidden_channels))
            self.prot_norms.append(torch.nn.LayerNorm(hidden_channels))

            self.nuc_lins.append(Linear(hidden_channels, hidden_channels, bias=False))
            self.residue_lins.append(Linear(hidden_channels, hidden_channels, bias=False))

            self.c2n_mlps.append(MLP([hidden_channels, hidden_channels * 2, hidden_channels], bias=False))
            self.c2r_mlps.append(MLP([hidden_channels, hidden_channels * 2, hidden_channels], bias=False))

            self.RNA_gn2.append(GraphNorm(hidden_channels))
            self.prot_gn2.append(GraphNorm(hidden_channels))


        self.attn_lin = PosLinear(heads * total_layer, 1, bias=False,
                                       init_value=1 / heads)  # (heads * total_layer))
        
        self.RNA_scores_attn_lin = PosLinear(heads * total_layer, 1, bias=False,
                                       init_value=1 / heads)  # (heads * total_layer))
        

        self.prot_scores_attn_lin = PosLinear(heads * total_layer, 1, bias=False,
                                       init_value=1 / heads)  # (heads * total_layer))

        # self.RNA_out = MLP([hidden_channels, hidden_channels * 2, hidden_channels], out_norm=True)
        # self.prot_out = MLP([hidden_channels, hidden_channels * 2, hidden_channels], out_norm=True)



    def reset_parameters(self):

        self.RNA_evo.reset_parameters()
        self.RNA_aa.reset_parameters()
        self.prot_evo.reset_parameters()
        self.prot_aa.reset_parameters()

        for idx in range(self.total_layer):
            self.RNA_convs[idx].reset_parameters()
            self.prot_convs[idx].reset_parameters()

            self.RNA_gn2[idx].reset_parameters()
            self.prot_gn2[idx].reset_parameters()

            self.RNA_cluster[idx].reset_parameters()
            self.prot_cluster[idx].reset_parameters()

            self.RNA_score_lins[idx].reset_parameters()
            self.prot_score_lins[idx].reset_parameters()
            
            self.RNA_norms[idx].reset_parameters()
            self.prot_norms[idx].reset_parameters()

            self.inter_convs[idx].reset_parameters()

            self.nuc_lins[idx].reset_parameters()
            self.residue_lins[idx].reset_parameters()

            self.c2n_mlps[idx].reset_parameters()
            self.c2r_mlps[idx].reset_parameters()

        self.attn_lin.reset_parameters()
        # self.RNA_out.reset_parameters()
        # self.prot_out.reset_parameters()
        self.RNA_scores_attn_lin.reset_parameters()
        self.prot_scores_attn_lin.reset_parameters() 



    def forward(self,
                # RNA
                nuc_x, nuc_evo_x, nuc_edge_index, nuc_edge_weight,
                # Protein
                residue_x, residue_evo_x, residue_edge_index, residue_edge_weight,
                # RNA-Protein Interaction batch
                RNA_batch=None, prot_batch=None,
                ## only if you're interested in clustering algorithm
                save_cluster = True):
        # print(prot_batch.shape)
        # print(prot_batch)

        # Init variables
        conmap_preds = None

        nuc_edge_attr = _rbf(nuc_edge_weight, D_max=1.0, D_count=self.RNA_edge_dim, device=self.device)
        residue_edge_attr = _rbf(residue_edge_weight, D_max=1.0, D_count=self.prot_edge_dim, device=self.device)

        # PROTEIN Featurize
        residue_x = self.prot_aa(residue_x) + self.prot_evo(residue_evo_x)

        # MOLECULE Featurize
        nuc_x = self.RNA_aa(nuc_x) + self.RNA_evo(nuc_evo_x)

        # cluster loss
        spectral_loss = torch.tensor(0.).to(self.device)
        prot_ortho_loss = torch.tensor(0.).to(self.device)
        prot_cluster_loss = torch.tensor(0.).to(self.device)

        RNA_ortho_loss = torch.tensor(0.).to(self.device)
        RNA_cluster_loss = torch.tensor(0.).to(self.device)

        nuc_scores = []
        residue_scores = []
        RNA_scores = []
        prot_scores = []
        layer_RNA_s = {}
        layer_prot_s = {}

        conmap_scores_flatten = []

        # RNA-PROTEIN Layers
        for idx in range(self.total_layer):

            ############################################################
            #####      Modeling Intramolecular Forces LAYER        #####
            ############################################################

            nuc_x = self.RNA_convs[idx](nuc_x, nuc_edge_index, nuc_edge_attr)
            residue_x = self.prot_convs[idx](residue_x, residue_edge_index, residue_edge_attr)

            # print(nuc_x.shape)
            # print(nuc_x)
            # print(residue_x.shape)
            # print(residue_x)

            ############################################################
            ###  Cluster based on physiochemical constraints LAYER   ###
            ############################################################
           
            ## Cluster RNA nucleotides
            dropped_nuc_edge_index, _ = dropout_edge(nuc_edge_index, p=self.dropout_cluster_edge,
                                                         force_undirected=True, training=self.training)
            RNA_s = self.RNA_cluster[idx](nuc_x, dropped_nuc_edge_index)

            # print(RNA_s.shape)
            # print(RNA_s)
            # RNA_max_indices = RNA_s.argmax(dim=1, keepdim=True)
            # print(f"RNA cluster")
            # print(RNA_max_indices)

            nuc_hx, nuc_mask = to_dense_batch(nuc_x, RNA_batch)

            # print(nuc_hx.shape)
            # print(nuc_hx)
            #
            # print(nuc_mask.shape)
            # print(nuc_mask)

            if save_cluster:
                layer_RNA_s[idx] = RNA_s

                # cluster features
            RNA_s, _ = to_dense_batch(RNA_s, RNA_batch)

            nuc_adj = to_dense_adj(nuc_edge_index, RNA_batch)
            RNA_cluster_mask = nuc_mask

            RNA_cluster_drop_mask = None
            if self.drop_nuc != 0 and self.training:
                _, _, nuc_drop_mask = dropout_node(nuc_edge_index, self.drop_nuc, nuc_x.size(0),
                                                       RNA_batch,
                                                       self.training)  # drop nuc for regularization
                nuc_drop_mask, _ = to_dense_batch(nuc_drop_mask.reshape(-1, 1),
                                                      RNA_batch)  # drop nuc for regularization
                nuc_drop_mask = nuc_drop_mask.squeeze()
                RNA_cluster_drop_mask = nuc_mask * nuc_drop_mask.squeeze()

            RNA_s, RNA_cluster_x, nuc_adj, cl_loss, o_loss = dense_mincut_pool(nuc_hx, nuc_adj, RNA_s, RNA_cluster_mask,
                                                                           RNA_cluster_drop_mask)

            # spectral_loss += sp_loss
            RNA_ortho_loss += o_loss
            RNA_cluster_loss += cl_loss
            RNA_cluster_x = self.RNA_norms[idx](RNA_cluster_x)



            ## Cluster protein residues
            dropped_residue_edge_index, _ = dropout_edge(residue_edge_index, p=self.dropout_cluster_edge,
                                                         force_undirected=True, training=self.training)
            prot_s = self.prot_cluster[idx](residue_x, dropped_residue_edge_index)
            residue_hx, residue_mask = to_dense_batch(residue_x, prot_batch)

            if save_cluster:
                layer_prot_s[idx] = prot_s

                # cluster features
            prot_s, _ = to_dense_batch(prot_s, prot_batch)
            residue_adj = to_dense_adj(residue_edge_index, prot_batch)
            # print("residue adj")
            # print(residue_adj.shape)
            # print(residue_adj)
            prot_cluster_mask = residue_mask

            prot_cluster_drop_mask = None
            if self.drop_residue != 0 and self.training:
                _, _, residue_drop_mask = dropout_node(residue_edge_index, self.drop_residue, residue_x.size(0),
                                                       prot_batch,
                                                       self.training)  # drop residue for regularization
                residue_drop_mask, _ = to_dense_batch(residue_drop_mask.reshape(-1, 1),
                                                      prot_batch)  # drop residue for regularization
                residue_drop_mask = residue_drop_mask.squeeze()
                prot_cluster_drop_mask = residue_mask * residue_drop_mask.squeeze()

            prot_s, prot_cluster_x, residue_adj, cl_loss, o_loss = dense_mincut_pool(residue_hx, residue_adj, prot_s, prot_cluster_mask,
                                                                           prot_cluster_drop_mask)
            # print(prot_s.shape)
            # print(prot_s)

            # print(prot_cluster_x.shape)
            # print(prot_cluster_x)

            # spectral_loss += sp_loss
            prot_ortho_loss += o_loss
            prot_cluster_loss += cl_loss
            prot_cluster_x = self.prot_norms[idx](prot_cluster_x)


            # connect RNA and protein cluster

            prot_batch_size = prot_s.size(0)
            cluster_residue_batch = torch.arange(prot_batch_size).repeat_interleave(self.num_cluster[idx]).to(self.device)
            prot_cluster_x = prot_cluster_x.reshape(prot_batch_size*self.num_cluster[idx], -1)

            RNA_batch_size = RNA_s.size(0)
            cluster_nuc_batch = torch.arange(RNA_batch_size).repeat_interleave(self.num_cluster[idx]).to(self.device)
            RNA_cluster_x = RNA_cluster_x.reshape(RNA_batch_size*self.num_cluster[idx], -1)

            # p2r_edge_index = torch.stack([torch.arange(RNA_batch_size*self.num_cluster[idx]),
            #                                 torch.arange(prot_batch_size*self.num_cluster[idx])]
            #                             ).to(self.device)
            p2r_edge_index = torch.empty(2, 0, dtype=torch.long).to(self.device)
            for Ridx in range(RNA_batch_size):
                RP_edge_index = torch.stack([torch.arange(Ridx*self.num_cluster[idx], (Ridx+1)*self.num_cluster[idx]).repeat(self.num_cluster[idx]),
                                             torch.arange(Ridx*self.num_cluster[idx], (Ridx+1)*self.num_cluster[idx]).repeat_interleave(self.num_cluster[idx])]
                                            ).to(self.device)

                p2r_edge_index = torch.cat((p2r_edge_index, RP_edge_index), dim=1)

            p2r_edge_index = p2r_edge_index.flip(0).to(self.device)
            # print(p2r_edge_index.flip(0))

            ## model interative relationship
            RNA_cluster_x, prot_cluster_x, inter_attn = self.inter_convs[idx](RNA_cluster_x, prot_cluster_x,
                                                                    p2r_edge_index)
            
            RNA_inter_attn = inter_attn[1] # [Btz*K*K, heads]
            prot_inter_attn = inter_attn[2] # [Btz*K*K, heads]
            RNA_inter_attn = RNA_inter_attn.view(RNA_batch_size, self.num_cluster[idx], self.num_cluster[idx], self.heads) # [Btz, K, K, heads]
            prot_inter_attn = prot_inter_attn.view(prot_batch_size, self.num_cluster[idx], self.num_cluster[idx], self.heads) # [Btz, K, K, heads]

            RNA_inter_attn = RNA_inter_attn.permute(0, 3, 1, 2) # [Btz, heads, K, K]
            prot_inter_attn = prot_inter_attn.permute(0, 3, 1, 2) # [Btz, heads, K, K]

            # print(f'RNA_inter_attn.shape: {RNA_inter_attn.shape}')
            # print(f'prot_inter_attn.shape:{prot_inter_attn.shape}')
            # print(f'RNA_inter_attn: {RNA_inter_attn}')
            # print(f'prot_inter_attn: {prot_inter_attn}')

            RNA_s_heads = RNA_s.unsqueeze(1) # [Btz, 1, max_nuc_number, K]
            # print(RNA_s_heads.shape)
            RNA_s_heads = RNA_s_heads.repeat(1, self.heads, 1, 1) # [Btz, heads, max_nuc_number, K]


            prot_s_heads = prot_s.unsqueeze(1) # [Btz, 1, max_residue_number, K]
            prot_s_heads = prot_s_heads.repeat(1, self.heads, 1, 1)    # [Btz, heads, max_residue_number, K]


            RNA_score = torch.einsum('bhij,bhjk->bhik', RNA_s_heads, RNA_inter_attn) # [Btz, heads, max_nuc_number, K]
            RNA_score_shape = RNA_score.shape
            # print(f'RNA_score.shape: {RNA_score.shape}')
            RNA_score = RNA_score.view(-1, self.num_cluster[idx])
            RNA_score = self.RNA_score_lins[idx](RNA_score)
            RNA_score = RNA_score.view(RNA_score_shape[0], RNA_score_shape[1], RNA_score_shape[2], 1) # [Btz, heads, max_nuc_number, 1]         
            # print(f'RNA_score.shape: {RNA_score.shape}')   
            # print(RNA_score.shape)
            RNA_scores.append(RNA_score)

            prot_score = torch.einsum('bhjk,bhkl->bhjl', prot_inter_attn, prot_s_heads.permute(0, 1, 3, 2))  # [Btz, heads, max_residue_number, K]
            prot_score = prot_score.permute(0, 1, 3, 2)
            prot_score_shape = prot_score.shape
            prot_score = prot_score.reshape(-1, self.num_cluster[idx])
            prot_score = self.prot_score_lins[idx](prot_score)
            prot_score = prot_score.view(prot_score_shape[0], prot_score_shape[1], prot_score_shape[2], 1) # [Btz, heads, max_residue_number, 1]     
            # print(f'prot_score.shape: {prot_score.shape}')
            # print(prot_score.shape)
            prot_scores.append(prot_score)



 
        #     # Residual
            nuc_hx, _ = to_dense_batch(RNA_cluster_x, cluster_nuc_batch)
        #     # print(nuc_hx)
        #     # print(nuc_hx.shape)
        #     # RNA_inter_attn, _ = to_dense_batch(RNA_inter_attn, cluster_nuc_batch)
            nuc_x = nuc_x + F.relu(self.nuc_lins[idx]((RNA_s @ nuc_hx)[nuc_mask]))  # cluster -> nucleotide
            nuc_x = nuc_x + self.c2n_mlps[idx](nuc_x)

            nuc_x = F.dropout(nuc_x, self.dropout, training=self.training)
        #     # RNA_inter_attn = (RNA_s @ RNA_inter_attn)[nuc_mask]
        #     # nuc_scores.append(RNA_inter_attn)

            residue_hx, _ = to_dense_batch(prot_cluster_x, cluster_residue_batch)
        #     # prot_inter_attn, _ = to_dense_batch(prot_inter_attn, cluster_residue_batch)

            residue_x = residue_x + F.relu(self.residue_lins[idx]((prot_s @ residue_hx)[residue_mask]))  # cluster -> residue
            residue_x = residue_x + self.c2r_mlps[idx](residue_x)

            residue_x = F.dropout(residue_x, self.dropout, training=self.training)
        #     # print(RNA_s.shape)
        #     # print(RNA_inter_attn.shape)
        #     # print(prot_s.shape)
        #     # print(prot_inter_attn.shape)
        #     # prot_inter_attn = (prot_s @ prot_inter_attn)[residue_mask]
        #     # residue_scores.append(prot_inter_attn)

            ## Graph Normalization
            nuc_x = self.RNA_gn2[idx](nuc_x, RNA_batch)
            residue_x = self.prot_gn2[idx](residue_x, prot_batch)



        RNA_scores = torch.cat(RNA_scores, dim=1)
        # print(RNA_scores.shape)
        RNA_scores = RNA_scores.squeeze(-1)
        RNA_scores = RNA_scores.permute(0, 2, 1)
        RNA_scores = self.RNA_scores_attn_lin(RNA_scores)
        RNA_final_scores = RNA_scores.squeeze(-1)
        RNA_final_scores = RNA_final_scores[nuc_mask] # shape [Nr]
        # print(f'RNA_final_scores.shape: {RNA_final_scores.shape}')

        prot_scores = torch.cat(prot_scores, dim=1)
        # print(prot_scores.shape)
        prot_scores = prot_scores.squeeze(-1)
        prot_scores = prot_scores.permute(0, 2, 1)
        prot_scores = self.prot_scores_attn_lin(prot_scores)
        prot_final_scores = prot_scores.squeeze(-1)
        prot_final_scores = prot_final_scores[residue_mask] # shape [Np]
        # print(f'prot_final_scores.shape: {prot_final_scores.shape}')


        return RNA_final_scores, prot_final_scores, RNA_ortho_loss, RNA_cluster_loss, prot_ortho_loss, prot_cluster_loss, layer_RNA_s, layer_prot_s  # , conmap_preds

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, amsgrad):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, GraphNorm, PosLinear)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias') or pn.endswith('mean_scale'):  # or pn.endswith('logit_scale'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                    # if mn.startswith('cluster'):
                    #     print(mn, 'not decayed!')
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad)

        return optimizer

def _rbf(D, D_min=0., D_max=1., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D = torch.where(D < D_max, D, torch.tensor(D_max).float().to(device) )
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def build_attention_matrix(inter_attn, edge_index, RNA_num_nodes, prot_num_nodes, head_number):
    # build attention matrix for each head: [head_number, RNA_num_nodes, prot_num_nodes]
    attention_matrix = torch.zeros((head_number, RNA_num_nodes, prot_num_nodes), dtype=torch.float)

    # iterate over all edges
    for edge_idx in range(edge_index.shape[1]):
        src_node = edge_index[0, edge_idx]  # source node
        tgt_node = edge_index[1, edge_idx]  # target node

        # fill attention scores for each head
        for head in range(head_number):
            attention_matrix[head, src_node, tgt_node] = inter_attn[edge_idx, head]

    return attention_matrix  # return attention matrix for each head


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

def dropout_node(edge_index, p, num_nodes, batch, training):
    r"""Randomly drops nodes from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    indicating which edges were retained. (3) the node mask indicating
    which nodes were retained.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor`, :class:`BoolTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask, node_mask = dropout_node(edge_index)
        >>> edge_index
        tensor([[0, 1],
                [1, 0]])
        >>> edge_mask
        tensor([ True,  True, False, False, False, False])
        >>> node_mask
        tensor([ True,  True, False, False])
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask

    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p

    ## ensure no graph is totally dropped out
    batch_tf = global_add_pool(node_mask.view(-1, 1), batch).flatten()
    unbatched_node_mask = unbatch(node_mask, batch)
    node_mask_list = []

    for true_false, sub_node_mask in zip(batch_tf, unbatched_node_mask):
        if true_false.item():
            node_mask_list.append(sub_node_mask)
        else:
            perm = torch.randperm(sub_node_mask.size(0))
            idx = perm[:1]
            sub_node_mask[idx] = True
            node_mask_list.append(sub_node_mask)

    node_mask = torch.cat(node_mask_list)

    edge_index, _, edge_mask = subgraph(node_mask, edge_index,
                                        num_nodes=num_nodes,
                                        return_edge_mask=True)
    return edge_index, edge_mask, node_mask