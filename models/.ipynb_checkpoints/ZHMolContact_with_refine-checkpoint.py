import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


##############################################
############### Main Model ###################
##############################################
class ZHMolContact(nn.Module):
    def __init__(
        self,
        heads=4,
        nuc_x_dim=21,
        nuc_evo_dim=640,
        prot_x_dim=33,
        prot_evo_dim=1280,
        evo_proj_dim=32,
        hidden_channels=48,
        edge_hidden_dim=48,
        num_classes=37,
        dropout=0,
        num_layers=2,
        num_triangle_layers=2,
        num_rbf=16,
        inter_feat_dim=25,
        node_feat_dropout=0.1,
        evo_feat_dropout=0.2,
        domain_dim=16,
        site_dim=8,
        rna_site_input_dim=1,
        prot_site_input_dim=1,
        use_site_logit_bias=True,
        init_site_bias_scale=0.1,
        device='cuda:0',
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.pair_dim = hidden_channels
        self.node_feat_dropout = node_feat_dropout
        self.evo_feat_dropout = evo_feat_dropout
        self.domain_dim = domain_dim
        self.site_dim = site_dim
        self.num_classes = num_classes
        self.device = device
        self.evo_proj_dim = evo_proj_dim
        self.inter_feat_dim = inter_feat_dim
        self.use_site_logit_bias = use_site_logit_bias

        # =========================================================
        # 1. Intra-molecular Interaction Learning
        # =========================================================
        self.rna_site_proj = nn.Sequential(
            nn.Linear(rna_site_input_dim, self.site_dim),
            nn.LayerNorm(self.site_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.prot_site_proj = nn.Sequential(
            nn.Linear(prot_site_input_dim, self.site_dim),
            nn.LayerNorm(self.site_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.rna_evo_proj = nn.Sequential(
            nn.Linear(nuc_evo_dim, evo_proj_dim),
            nn.LayerNorm(evo_proj_dim),
            nn.GELU(),
            nn.Dropout(evo_feat_dropout)
        )

        self.prot_evo_proj = nn.Sequential(
            nn.Linear(prot_evo_dim, evo_proj_dim),
            nn.LayerNorm(evo_proj_dim),
            nn.GELU(),
            nn.Dropout(evo_feat_dropout)
        )

        self.rna_encoder = GeoGraphEncoder(
            node_in_dim=nuc_x_dim + evo_proj_dim + self.site_dim,
            node_dim=hidden_channels,
            edge_hidden_dim=edge_hidden_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            dropout=dropout
        )

        self.prot_encoder = GeoGraphEncoder(
            node_in_dim=prot_x_dim + evo_proj_dim + self.site_dim,
            node_dim=hidden_channels,
            edge_hidden_dim=edge_hidden_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            dropout=dropout
        )

        # =========================================================
        # 2. Cross-molecular Pair Modeling
        # =========================================================
        self.domain_scorer = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )

        self.domain_prior_proj = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )

        self.domain_residue_gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Sigmoid()
        )
        self.domain_residue_norm = nn.LayerNorm(hidden_channels)

        self.inter_pair_proj = nn.Sequential(
            nn.Linear(inter_feat_dim, self.pair_dim),
            nn.LayerNorm(self.pair_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.pair_dim, self.pair_dim)
        )

        self.pair_fuse = nn.Sequential(
            nn.Linear(self.pair_dim * 2, self.pair_dim),
            nn.LayerNorm(self.pair_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.pair_dim, self.pair_dim)
        )

        self.bilinear = nn.Bilinear(hidden_channels, hidden_channels, self.pair_dim)

        self.pair_net = PairResNet(
            pair_dim=self.pair_dim,
            hidden_dim=hidden_channels,
            num_blocks=num_layers,
            dropout=dropout
        )

        # =========================================================
        # 3. Geometry-aware Interaction Learning
        # =========================================================
        self.rna_pair_encoder = IntraPairEncoder(
            node_dim=hidden_channels,
            pair_dim=self.pair_dim,
            dropout=dropout
        )
        self.prot_pair_encoder = IntraPairEncoder(
            node_dim=hidden_channels,
            pair_dim=self.pair_dim,
            dropout=dropout
        )

        self.num_triangle_layers = num_triangle_layers
        self.triangle_layers = nn.ModuleList([
            HeteroTriangleUpdate(
                pair_dim=self.pair_dim,
                hidden_dim=self.pair_dim,
                dropout=dropout
            )
            for _ in range(self.num_triangle_layers)
        ])

        self.site_pair_input_dim = 3 * self.site_dim

        self.site_pair_proj = nn.Sequential(
            nn.Linear(self.site_pair_input_dim, self.pair_dim),
            nn.LayerNorm(self.pair_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.pair_dim, self.pair_dim)
        )

        self.site_pair_gate = nn.Sequential(
            nn.Linear(self.pair_dim * 2, self.pair_dim),
            nn.LayerNorm(self.pair_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.pair_dim, self.pair_dim),
            nn.Sigmoid()
        )
        self.site_pair_norm = nn.LayerNorm(self.pair_dim)

        self.site_logit_bias = nn.Sequential(
            nn.Linear(self.site_pair_input_dim, self.pair_dim),
            nn.LayerNorm(self.pair_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.pair_dim, num_classes)
        )
        self.site_bias_scale = nn.Parameter(torch.tensor(float(init_site_bias_scale)))

        self.axial_stack = AxialStack(
            dim=self.pair_dim,
            heads=heads,
            num_layers=1,
            dropout=dropout,
            num_buckets=32,
            max_row_distance=128,
            max_col_distance=128
        )

        self.post_axial_refine = PairResNet(
            pair_dim=self.pair_dim,
            hidden_dim=self.pair_dim,
            num_blocks=num_layers,
            dropout=dropout
        )

        self.classifier = nn.Linear(self.pair_dim, num_classes)
        self.reset_prior_parameters()

    def reset_prior_parameters(self):
        zero_init_last_linear(self.domain_prior_proj, scale=1e-3)
        zero_init_last_linear(self.site_pair_proj, scale=1e-3)
        zero_init_last_linear(self.site_logit_bias, scale=1e-3)

    def forward(
        self,
        nuc_x, nuc_evo_x, nuc_coord_x, nuc_edge_index, nuc_edge_weight, RNA_local_R, RNA_local_t, RNA_pred_site,
        residue_x, residue_evo_x, residue_coord_x, residue_edge_index, residue_edge_weight, prot_local_R, prot_local_t, prot_pred_site, protein_domain_mask,
        RNA_batch, prot_batch,
        inter_residue_feat_pad
    ):
        # =========================================================
        # 1. Intra-molecular Interaction Learning
        # =========================================================
        rna_x = F.dropout(nuc_x, p=self.node_feat_dropout, training=self.training)
        nuc_evo_x = F.dropout(nuc_evo_x, p=self.evo_feat_dropout, training=self.training)
        nuc_evo_feat = self.rna_evo_proj(nuc_evo_x)

        RNA_pred_site = prepare_site_tensor(
            RNA_pred_site,
            RNA_batch,
            self.rna_site_proj[0].in_features
        )

        rna_site_feat = self.rna_site_proj(RNA_pred_site)
        rna_site_feat = F.dropout(rna_site_feat, p=self.node_feat_dropout, training=self.training)
        rna_node_feat = torch.cat([rna_x, nuc_evo_feat, rna_site_feat], dim=-1)

        h_rna = self.rna_encoder(
            node_feat=rna_node_feat,
            edge_index=nuc_edge_index,
            edge_weight=nuc_edge_weight,
            local_R=RNA_local_R,
            local_t=RNA_local_t
        )

        prot_x = F.dropout(residue_x, p=self.node_feat_dropout, training=self.training)
        residue_evo_x = F.dropout(residue_evo_x, p=self.evo_feat_dropout, training=self.training)
        prot_evo_feat = self.prot_evo_proj(residue_evo_x)

        prot_pred_site = prepare_site_tensor(
            prot_pred_site,
            prot_batch,
            self.prot_site_proj[0].in_features
        )

        prot_site_feat = self.prot_site_proj(prot_pred_site)
        prot_site_feat = F.dropout(prot_site_feat, p=self.node_feat_dropout, training=self.training)
        prot_node_feat = torch.cat([prot_x, prot_evo_feat, prot_site_feat], dim=-1)

        h_prot = self.prot_encoder(
            node_feat=prot_node_feat,
            edge_index=residue_edge_index,
            edge_weight=residue_edge_weight,
            local_R=prot_local_R,
            local_t=prot_local_t
        )

        # =========================================================
        # 2. Cross-molecular Pair Modeling
        # =========================================================
        h_rna_dense, rna_mask = to_dense_batch(h_rna, RNA_batch)
        h_prot_dense, prot_mask = to_dense_batch(h_prot, prot_batch)

        rna_site_dense, _ = to_dense_batch(rna_site_feat, RNA_batch)
        prot_site_dense, _ = to_dense_batch(prot_site_feat, prot_batch)

        B, H, d = h_rna_dense.shape
        _, W, _ = h_prot_dense.shape

        domain_logits, residue_domain_prior, residue_domain_score, domain_mask_norm = build_domain_prior(
            h_rna_dense=h_rna_dense,
            h_prot_dense=h_prot_dense,
            rna_mask=rna_mask,
            prot_mask=prot_mask,
            protein_domain_mask=protein_domain_mask,
            prot_batch=prot_batch,
            domain_scorer=self.domain_scorer,
            domain_prior_proj=self.domain_prior_proj
        )

        domain_gate = self.domain_residue_gate(torch.cat([h_prot_dense, residue_domain_prior], dim=-1))
        h_prot_dense = self.domain_residue_norm(h_prot_dense + domain_gate * residue_domain_prior)
        h_prot_dense = h_prot_dense * prot_mask.unsqueeze(-1).float()

        rna_expand = h_rna_dense.unsqueeze(2).expand(B, H, W, d)
        prot_expand = h_prot_dense.unsqueeze(1).expand(B, H, W, d)

        pair_feat = self.bilinear(rna_expand, prot_expand)
        pair_mask = rna_mask.unsqueeze(2) & prot_mask.unsqueeze(1)
        pair_feat = pair_feat * pair_mask.unsqueeze(-1).float()

        inter_pair_feat = prepare_inter_residue_tensor(
            inter_feat=inter_residue_feat_pad,
            B=B,
            H=H,
            W=W,
            expected_dim=self.inter_feat_dim,
            pair_mask=pair_mask
        )

        inter_pair_feat = self.inter_pair_proj(inter_pair_feat)
        inter_pair_feat = inter_pair_feat * pair_mask.unsqueeze(-1).float()
        pair_feat = pair_feat + self.pair_fuse(torch.cat([pair_feat, inter_pair_feat], dim=-1))
        pair_feat = pair_feat * pair_mask.unsqueeze(-1).float()
        pair_feat = self.pair_net(pair_feat, pair_mask)

        # =========================================================
        # 3. Geometry-aware Interaction Learning
        # =========================================================
        rna_pair_feat = self.rna_pair_encoder(h_rna_dense)
        prot_pair_feat = self.prot_pair_encoder(h_prot_dense)

        rna_pair_mask = rna_mask.unsqueeze(2) & rna_mask.unsqueeze(1)
        prot_pair_mask = prot_mask.unsqueeze(2) & prot_mask.unsqueeze(1)

        rna_pair_feat = rna_pair_feat * rna_pair_mask.unsqueeze(-1).float()
        prot_pair_feat = prot_pair_feat * prot_pair_mask.unsqueeze(-1).float()

        for tri_layer in self.triangle_layers:
            pair_feat = tri_layer(
                pair_feat=pair_feat,
                rna_pair_feat=rna_pair_feat,
                prot_pair_feat=prot_pair_feat,
                pair_mask=pair_mask,
                rna_pair_mask=rna_pair_mask,
                prot_pair_mask=prot_pair_mask
            )

        site_pair_input = build_site_pair_input(
            rna_site_dense=rna_site_dense,
            prot_site_dense=prot_site_dense
        )

        site_pair_prior = self.site_pair_proj(site_pair_input)
        site_pair_prior = site_pair_prior * pair_mask.unsqueeze(-1).float()
        site_gate = self.site_pair_gate(torch.cat([pair_feat, site_pair_prior], dim=-1))
        pair_feat = self.site_pair_norm(pair_feat + site_gate * site_pair_prior)
        pair_feat = pair_feat * pair_mask.unsqueeze(-1).float()

        pair_feat = self.axial_stack(pair_feat, pair_mask)
        pair_feat = self.post_axial_refine(pair_feat, pair_mask)

        logits = self.classifier(pair_feat)

        if self.use_site_logit_bias:
            site_bias = self.site_logit_bias(site_pair_input)
            site_bias = site_bias * pair_mask.unsqueeze(-1).float()
            logits = logits + self.site_bias_scale * site_bias

        logits = logits * pair_mask.unsqueeze(-1).float()
        logits = logits.permute(0, 3, 1, 2).contiguous()

        return logits


##############################################
############### Helper Functions #############
##############################################
def zero_init_last_linear(module, scale=1e-3):
    last_linear = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        nn.init.normal_(last_linear.weight, mean=0.0, std=scale)
        if last_linear.bias is not None:
            nn.init.constant_(last_linear.bias, 0.0)


def prepare_inter_residue_tensor(inter_feat, B, H, W, expected_dim, pair_mask=None):
    if inter_feat is None:
        raise ValueError('inter_residue_feat_pad is None')
    if inter_feat.dim() != 4:
        raise ValueError(f'inter_residue_feat_pad must be [B,H,W,C], got shape={inter_feat.shape}')
    if inter_feat.size(0) != B or inter_feat.size(1) != H or inter_feat.size(2) != W:
        raise ValueError(
            f'inter_residue_feat_pad shape mismatch: expected [{B},{H},{W},C], got {list(inter_feat.shape)}'
        )
    if inter_feat.size(-1) != expected_dim:
        raise ValueError(
            f'inter_residue_feat_pad feature dim mismatch: expected {expected_dim}, got {inter_feat.size(-1)}'
        )

    inter_feat = inter_feat.float()
    if pair_mask is not None:
        inter_feat = inter_feat * pair_mask.unsqueeze(-1).float()
    return inter_feat


def masked_mean(x, mask, dim):
    mask_f = mask.float()
    while mask_f.dim() < x.dim():
        mask_f = mask_f.unsqueeze(-1)
    num = (x * mask_f).sum(dim=dim)
    den = mask_f.sum(dim=dim).clamp(min=1.0)
    return num / den


def prepare_protein_domain_mask(protein_domain_mask, prot_batch, prot_mask=None):
    if protein_domain_mask is None:
        raise ValueError('protein_domain_mask is None')

    x = protein_domain_mask

    if x.dim() == 1:
        domain_ids = x.long()
        if domain_ids.numel() != prot_batch.numel():
            raise ValueError(
                f'protein_domain_mask [N] length mismatch: got N={domain_ids.numel()}, but prot_batch has N={prot_batch.numel()}'
            )

        valid = domain_ids >= 0
        num_domains = int(domain_ids[valid].max().item()) + 1 if valid.any() else 1
        dense_ids, dense_valid = to_dense_batch(domain_ids, prot_batch, fill_value=-1)
        B, W = dense_ids.shape

        domain_mask_dense = torch.zeros(B, W, num_domains, device=dense_ids.device, dtype=torch.float)
        valid_pos = dense_ids >= 0
        if valid_pos.any():
            domain_mask_dense[valid_pos, dense_ids[valid_pos]] = 1.0

        if prot_mask is not None:
            domain_mask_dense = domain_mask_dense * prot_mask.unsqueeze(-1).float()
        return domain_mask_dense

    if x.dim() == 2 and x.size(-1) == 1:
        return prepare_protein_domain_mask(x.squeeze(-1), prot_batch=prot_batch, prot_mask=prot_mask)

    if x.dim() == 3:
        domain_mask_dense = x.float()
        if prot_mask is not None:
            B, W = prot_mask.shape
            if domain_mask_dense.size(0) != B or domain_mask_dense.size(1) != W:
                raise ValueError(
                    f'protein_domain_mask [B,W,D] shape mismatch: expected first two dims [{B},{W}], got {list(domain_mask_dense.shape)}'
                )
            domain_mask_dense = domain_mask_dense * prot_mask.unsqueeze(-1).float()
        return domain_mask_dense

    raise ValueError(
        f'Unsupported protein_domain_mask shape={x.shape}. Expected [N], [N,1], or [B,W,D].'
    )



def build_domain_prior(
    h_rna_dense,
    h_prot_dense,
    rna_mask,
    prot_mask,
    protein_domain_mask,
    prot_batch,
    domain_scorer,
    domain_prior_proj,
):
    B, H, C = h_rna_dense.shape
    _, W, Cp = h_prot_dense.shape
    if C != Cp:
        raise ValueError(f'h_rna_dense dim {C} != h_prot_dense dim {Cp}')

    domain_mask = prepare_protein_domain_mask(
        protein_domain_mask=protein_domain_mask,
        prot_batch=prot_batch,
        prot_mask=prot_mask,
    )

    if domain_mask.size(0) != B or domain_mask.size(1) != W:
        raise ValueError(
            f'domain mask mismatch after prepare: expected [B={B},W={W},D], got {domain_mask.shape}'
        )

    domain_denom = domain_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    domain_mask_norm = domain_mask / domain_denom
    domain_feat = torch.einsum('bwc,bwd->bdc', h_prot_dense, domain_mask_norm)

    rna_valid = rna_mask.unsqueeze(-1).float()
    rna_global = (h_rna_dense * rna_valid).sum(dim=1) / rna_valid.sum(dim=1).clamp(min=1.0)
    rna_expand = rna_global.unsqueeze(1).expand_as(domain_feat)

    score_input = torch.cat([
        domain_feat,
        rna_expand,
        domain_feat * rna_expand,
        torch.abs(domain_feat - rna_expand),
    ], dim=-1)

    domain_logits = domain_scorer(score_input).squeeze(-1)
    domain_prob = torch.sigmoid(domain_logits)

    residue_domain_score = torch.einsum('bd,bwd->bw', domain_prob, domain_mask)
    residue_domain_score = residue_domain_score * prot_mask.float()

    residue_domain_prior = domain_prior_proj(residue_domain_score.unsqueeze(-1))
    residue_domain_prior = residue_domain_prior * prot_mask.unsqueeze(-1).float()

    return domain_logits, residue_domain_prior, residue_domain_score, domain_mask_norm



def prepare_site_tensor(site_tensor, batch_ids, expected_dim):
    if site_tensor is None:
        raise ValueError('site_tensor is None, but site feature is enabled.')

    site_tensor = site_tensor.float()

    if site_tensor.dim() == 1:
        site_tensor = site_tensor.unsqueeze(-1)
    elif site_tensor.dim() == 2:
        pass
    elif site_tensor.dim() == 3:
        B, L, D = site_tensor.shape
        batch_ids = batch_ids.long()

        num_graphs = int(batch_ids.max().item()) + 1
        if num_graphs != B:
            raise ValueError(
                f'site tensor batch dim mismatch: site B={B}, but batch_ids imply B={num_graphs}'
            )

        counts = torch.bincount(
            batch_ids.detach().cpu(),
            minlength=B
        ).to(batch_ids.device)
        
        if (counts > L).any():
            raise ValueError('Padded site tensor is shorter than a graph in batch_ids')
        valid = torch.arange(L, device=site_tensor.device)[None, :] < counts[:, None]
        site_tensor = site_tensor[valid]
    else:
        raise ValueError(f'site_tensor must be 1D / 2D / 3D, got shape={site_tensor.shape}')

    if site_tensor.dim() != 2:
        raise ValueError(f'site_tensor reshape failed, got shape={site_tensor.shape}')
    if site_tensor.size(-1) != expected_dim:
        raise ValueError(f'site feature dim mismatch: expected {expected_dim}, got {site_tensor.size(-1)}')
    if site_tensor.size(0) != batch_ids.numel():
        raise ValueError('Site feature count must match the number of graph nodes')
    return site_tensor



def build_site_pair_input(rna_site_dense, prot_site_dense):
    B, H, Ds = rna_site_dense.shape
    _, W, _ = prot_site_dense.shape

    rna_site_expand = rna_site_dense.unsqueeze(2).expand(B, H, W, Ds)
    prot_site_expand = prot_site_dense.unsqueeze(1).expand(B, H, W, Ds)
    site_cross = rna_site_expand * prot_site_expand

    site_pair_input = torch.cat([rna_site_expand, prot_site_expand, site_cross], dim=-1)
    return site_pair_input



def rbf_expansion(dist, num_centers=16, cutoff=20.0):
    if dist.dim() == 1:
        dist = dist.unsqueeze(-1)

    centers = torch.linspace(0, cutoff, num_centers, device=dist.device)
    centers = centers.view(1, -1)
    gamma = 1.0 / (cutoff / num_centers) ** 2
    return torch.exp(-gamma * (dist - centers) ** 2)



def build_geometric_edge_feature(edge_index, local_R, local_t, edge_weight=None, num_rbf=16):
    src, dst = edge_index

    Ri = local_R[src]
    Rj = local_R[dst]
    ti = local_t[src]
    tj = local_t[dst]

    delta = tj - ti
    dist = torch.norm(delta, dim=-1, keepdim=True)
    dist_rbf = rbf_expansion(dist, num_centers=num_rbf)

    Ri_T = Ri.transpose(-1, -2)
    Rj_T = Rj.transpose(-1, -2)

    delta_local_i = torch.bmm(Ri_T, delta.unsqueeze(-1)).squeeze(-1)
    delta_local_j = torch.bmm(Rj_T, (-delta).unsqueeze(-1)).squeeze(-1)

    relR = torch.bmm(Ri_T, Rj).reshape(delta.shape[0], -1)

    if edge_weight is None:
        edge_weight = torch.zeros(delta.shape[0], 1, device=delta.device, dtype=delta.dtype)
    else:
        if edge_weight.dim() == 1:
            edge_weight = edge_weight.unsqueeze(-1)

    edge_feat = torch.cat([delta_local_i, delta_local_j, dist_rbf, relR, edge_weight], dim=-1)
    return edge_feat


class GeoMessagePassingLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )

        self.upd_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )

        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_feat):
        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]

        msg_input = torch.cat([h_src, h_dst, edge_feat], dim=-1)
        msg = self.msg_mlp(msg_input)

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)

        upd_input = torch.cat([h, agg], dim=-1)
        dh = self.upd_mlp(upd_input)

        h = self.norm(h + self.dropout(dh))
        return h


class NodeInputEmbedding(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class EdgeEmbedding(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x + self.ffn(x)
        x = self.out_norm(x)
        return x


class GeoGraphEncoder(nn.Module):
    def __init__(self, node_in_dim, node_dim=128, edge_hidden_dim=128, num_layers=3, num_rbf=16, dropout=0.1):
        super().__init__()
        self.num_rbf = num_rbf
        self.raw_edge_dim = 3 + 3 + num_rbf + 9 + 1

        self.input_proj = NodeInputEmbedding(node_in_dim, node_dim, node_dim, dropout=dropout)
        self.edge_proj = EdgeEmbedding(self.raw_edge_dim, edge_hidden_dim, dropout)

        self.layers = nn.ModuleList([
            GeoMessagePassingLayer(
                node_dim=node_dim,
                edge_dim=edge_hidden_dim,
                hidden_dim=edge_hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(node_dim)

    def forward(self, node_feat, edge_index, edge_weight, local_R, local_t):
        h = self.input_proj(node_feat)

        raw_edge_feat = build_geometric_edge_feature(
            edge_index=edge_index,
            local_R=local_R,
            local_t=local_t,
            edge_weight=edge_weight,
            num_rbf=self.num_rbf,
        )

        edge_feat = self.edge_proj(raw_edge_feat)

        for layer in self.layers:
            h = layer(h, edge_index, edge_feat)

        h = self.out_norm(h)
        return h


class TriangularCrossAttnBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1, triangle='lower'):
        super().__init__()
        assert triangle in ['lower', 'upper']
        self.triangle = triangle

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def build_triangle_mask(self, Lq, Lk, device):
        if Lq == 0 or Lk == 0:
            return None

        q_pos = torch.arange(Lq, device=device).float()
        k_pos = torch.arange(Lk, device=device).float()

        q_pos = q_pos / (Lq - 1) if Lq > 1 else torch.zeros_like(q_pos)
        k_pos = k_pos / (Lk - 1) if Lk > 1 else torch.zeros_like(k_pos)

        q_grid = q_pos.unsqueeze(1)
        k_grid = k_pos.unsqueeze(0)

        if self.triangle == 'lower':
            attn_mask = k_grid > q_grid
        else:
            attn_mask = k_grid < q_grid
        return attn_mask

    def forward(self, q, kv, kv_mask=None):
        q_in = self.norm_q(q)
        kv_in = self.norm_kv(kv)

        Lq = q_in.size(1)
        Lk = kv_in.size(1)
        tri_mask = self.build_triangle_mask(Lq, Lk, q.device)

        attn_out, _ = self.attn(
            query=q_in,
            key=kv_in,
            value=kv_in,
            key_padding_mask=kv_mask,
            attn_mask=tri_mask,
        )

        x = q + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


def _get_gn_groups(channels, candidates=(8, 4, 2, 1)):
    for g in candidates:
        if channels % g == 0:
            return g
    return 1


class ResBlock2D(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        gn_groups = _get_gn_groups(channels)

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=gn_groups, num_channels=channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=gn_groups, num_channels=channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class PairResNet(nn.Module):
    def __init__(self, pair_dim=256, hidden_dim=256, num_blocks=4, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Conv2d(pair_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(*[ResBlock2D(hidden_dim, dropout=dropout) for _ in range(num_blocks)])
        self.out_head = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, pair_feat, pair_mask=None):
        x = pair_feat.permute(0, 3, 1, 2).contiguous()
        x = self.in_proj(x)
        x = self.blocks(x)
        pair_feat = self.out_head(x)
        if pair_mask is not None:
            pair_feat = pair_feat * pair_mask.unsqueeze(1).float()
        pair_feat = pair_feat.permute(0, 2, 3, 1).contiguous()
        return pair_feat


###################################################
##############    axial attention   ###############
###################################################
def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
    num_buckets = int(num_buckets)
    max_distance = int(max_distance)

    ret = 0
    n = -relative_position

    half_buckets = num_buckets // 2
    ret += (n < 0).long() * half_buckets
    n = torch.abs(n)

    max_exact = half_buckets // 2
    is_small = n < max_exact

    n_float = torch.clamp(n.float(), min=1.0)
    val_if_large = max_exact + (
        torch.log(n_float / max_exact) / math.log(max_distance / max_exact)
    ) * (half_buckets - max_exact)

    val_if_large = val_if_large.long()
    val_if_large = torch.clamp(val_if_large, max=half_buckets - 1)

    bucket = torch.where(is_small, n, val_if_large)
    ret += bucket
    return ret


class RelPosMultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1, num_buckets=32, max_distance=128):
        super().__init__()
        assert dim % heads == 0, f'dim={dim} must be divisible by heads={heads}'

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.rel_pos_bias = nn.Embedding(num_buckets, heads)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def _compute_rel_pos_bias(self, L, device):
        pos = torch.arange(L, device=device)
        rel_pos = pos[None, :] - pos[:, None]
        buckets = relative_position_bucket(
            rel_pos,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )

        bias = self.rel_pos_bias(buckets)
        bias = bias.permute(2, 0, 1).contiguous()
        return bias

    def forward(self, x, key_padding_mask=None):
        B, L, C = x.shape

        q = self.q_proj(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        rel_bias = self._compute_rel_pos_bias(L, x.device)
        attn_scores = attn_scores + rel_bias.unsqueeze(0)

        if key_padding_mask is not None:
            key_mask = key_padding_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(key_mask, float('-inf'))
            # An entirely padded axis has no keys; avoid softmax(-inf, ..., -inf).
            empty_axes = key_padding_mask.all(dim=-1)[:, None, None, None]
            attn_scores = attn_scores.masked_fill(empty_axes, 0.0)

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, C)
        out = self.out_proj(out)
        out = self.proj_dropout(out)

        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return out


class AxialAttention2D(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1, num_buckets=32, max_row_distance=128, max_col_distance=128):
        super().__init__()
        self.dim = dim
        self.heads = heads

        self.row_norm = nn.LayerNorm(dim)
        self.col_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

        self.row_attn = RelPosMultiHeadSelfAttention(
            dim=dim,
            heads=heads,
            dropout=dropout,
            num_buckets=num_buckets,
            max_distance=max_row_distance,
        )

        self.col_attn = RelPosMultiHeadSelfAttention(
            dim=dim,
            heads=heads,
            dropout=dropout,
            num_buckets=num_buckets,
            max_distance=max_col_distance,
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, pair_mask=None):
        B, H, W, C = x.shape

        x_row = x.permute(0, 2, 1, 3).contiguous().view(B * W, H, C)
        if pair_mask is not None:
            row_mask = pair_mask.permute(0, 2, 1).contiguous().view(B * W, H)
            row_key_padding_mask = ~row_mask
        else:
            row_key_padding_mask = None

        x_row = x_row + self.row_attn(self.row_norm(x_row), key_padding_mask=row_key_padding_mask)
        x = x_row.view(B, W, H, C).permute(0, 2, 1, 3).contiguous()

        x_col = x.view(B * H, W, C)
        if pair_mask is not None:
            col_mask = pair_mask.view(B * H, W)
            col_key_padding_mask = ~col_mask
        else:
            col_key_padding_mask = None

        x_col = x_col + self.col_attn(self.col_norm(x_col), key_padding_mask=col_key_padding_mask)
        x = x_col.view(B, H, W, C)

        x = x + self.ffn(self.ffn_norm(x))
        if pair_mask is not None:
            x = x * pair_mask.unsqueeze(-1).float()
        return x


class AxialStack(nn.Module):
    def __init__(self, dim, heads=4, num_layers=2, dropout=0.1, num_buckets=32, max_row_distance=128, max_col_distance=128):
        super().__init__()
        self.layers = nn.ModuleList([
            AxialAttention2D(
                dim=dim,
                heads=heads,
                dropout=dropout,
                num_buckets=num_buckets,
                max_row_distance=max_row_distance,
                max_col_distance=max_col_distance,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, pair_mask=None):
        for layer in self.layers:
            x = layer(x, pair_mask)
        return x


class IntraPairEncoder(nn.Module):
    def __init__(self, node_dim, pair_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(node_dim * 4, pair_dim),
            nn.LayerNorm(pair_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim, pair_dim),
        )

    def forward(self, x):
        B, L, D = x.shape
        xi = x.unsqueeze(2).expand(B, L, L, D)
        xj = x.unsqueeze(1).expand(B, L, L, D)
        feat = torch.cat([xi, xj, torch.abs(xi - xj), xi * xj], dim=-1)
        return self.proj(feat)


class HeteroTriangleUpdate(nn.Module):
    'Lightweight heteromolecular triangle update.\n\nEdge-level scalar gates emphasize triangle paths while limiting memory.\nMessages are added without a [B,H,W,3C] concatenation; the default\nhidden dimension is max(pair_dim // 2, 16).'

    def __init__(self, pair_dim, hidden_dim=None, dropout=0.1):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = max(pair_dim // 2, 16)

        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim

        # ---------------- RNA-anchor ----------------
        self.rna_left_proj = nn.Linear(pair_dim, hidden_dim, bias=False)
        self.rna_right_proj = nn.Linear(pair_dim, hidden_dim, bias=False)

        # Single-channel gates reduce parameter count and activation memory.
        self.rna_left_gate = nn.Linear(pair_dim, 1, bias=True)
        self.rna_right_gate = nn.Linear(pair_dim, 1, bias=True)

        self.rna_out_proj = nn.Linear(hidden_dim, pair_dim, bias=True)

        # ---------------- Protein-anchor ----------------
        self.prot_left_proj = nn.Linear(pair_dim, hidden_dim, bias=False)
        self.prot_right_proj = nn.Linear(pair_dim, hidden_dim, bias=False)

        self.prot_left_gate = nn.Linear(pair_dim, 1, bias=True)
        self.prot_right_gate = nn.Linear(pair_dim, 1, bias=True)

        self.prot_out_proj = nn.Linear(hidden_dim, pair_dim, bias=True)

        # ---------------- lightweight fusion ----------------
        # Add messages before projection to avoid a [B,H,W,3C] concatenation.
        self.msg_fuse_norm = nn.LayerNorm(pair_dim)
        self.msg_fuse_proj = nn.Sequential(
            nn.Linear(pair_dim, pair_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim, pair_dim),
        )

        # Let pair_feat control the contribution of triangle messages.
        self.absorb_gate = nn.Sequential(
            nn.Linear(pair_dim, pair_dim),
            nn.Sigmoid()
        )

        # Start with a small learned scale to limit early updates.
        self.res_scale = nn.Parameter(torch.tensor(0.25))

        self.out_norm = nn.LayerNorm(pair_dim)
        self.ffn_norm = nn.LayerNorm(pair_dim)
        self.ffn = nn.Sequential(
            nn.Linear(pair_dim, pair_dim * 2),   # Use a 2x FFN expansion instead of 4x.
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim * 2, pair_dim),
            nn.Dropout(dropout),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.rna_out_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.prot_out_proj.weight, gain=0.5)

    def forward(
        self,
        pair_feat,          # [B,H,W,C]
        rna_pair_feat,      # [B,H,H,C]
        prot_pair_feat,     # [B,W,W,C]
        pair_mask=None,     # [B,H,W]
        rna_pair_mask=None, # [B,H,H]
        prot_pair_mask=None # [B,W,W]
    ):
        B, H, W, C = pair_feat.shape

        # =========================================================
        # 1) RNA-anchor triangle: R_ik + P_kj -> P_ij
        # =========================================================
        rna_left = self.rna_left_proj(rna_pair_feat)    # [B,H,H,D]
        rna_right = self.rna_right_proj(pair_feat)      # [B,H,W,D]

        # Apply scalar gates by broadcasting.
        rna_left = rna_left * torch.sigmoid(self.rna_left_gate(rna_pair_feat))
        rna_right = rna_right * torch.sigmoid(self.rna_right_gate(pair_feat))

        if rna_pair_mask is not None:
            rna_left = rna_left * rna_pair_mask.unsqueeze(-1).float()
        if pair_mask is not None:
            rna_right = rna_right * pair_mask.unsqueeze(-1).float()

        rna_msg = torch.einsum('bikd,bkjd->bijd', rna_left, rna_right)
        rna_msg = rna_msg / math.sqrt(max(H, 1))
        rna_msg = self.rna_out_proj(rna_msg)            # [B,H,W,C]

        if pair_mask is not None:
            rna_msg = rna_msg * pair_mask.unsqueeze(-1).float()

        # Release intermediate activation references early.
        del rna_left, rna_right

        # =========================================================
        # 2) Protein-anchor triangle: P_ik + S_kj -> P_ij
        # =========================================================
        prot_left = self.prot_left_proj(pair_feat)      # [B,H,W,D]
        prot_right = self.prot_right_proj(prot_pair_feat)  # [B,W,W,D]

        prot_left = prot_left * torch.sigmoid(self.prot_left_gate(pair_feat))
        prot_right = prot_right * torch.sigmoid(self.prot_right_gate(prot_pair_feat))

        if pair_mask is not None:
            prot_left = prot_left * pair_mask.unsqueeze(-1).float()
        if prot_pair_mask is not None:
            prot_right = prot_right * prot_pair_mask.unsqueeze(-1).float()

        prot_msg = torch.einsum('bikd,bkjd->bijd', prot_left, prot_right)
        prot_msg = prot_msg / math.sqrt(max(W, 1))
        prot_msg = self.prot_out_proj(prot_msg)         # [B,H,W,C]

        if pair_mask is not None:
            prot_msg = prot_msg * pair_mask.unsqueeze(-1).float()

        del prot_left, prot_right

        # =========================================================
        # 3) lightweight fusion
        # =========================================================
        tri_msg = rna_msg + prot_msg
        tri_msg = self.msg_fuse_proj(self.msg_fuse_norm(tri_msg))

        # Use pair_feat to gate the update.
        gate = self.absorb_gate(pair_feat)
        x = pair_feat + self.res_scale * gate * tri_msg
        x = self.out_norm(x)

        # =========================================================
        # 4) lighter FFN
        # =========================================================
        x = x + self.ffn(self.ffn_norm(x))

        if pair_mask is not None:
            x = x * pair_mask.unsqueeze(-1).float()

        return x
