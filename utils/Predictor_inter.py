import torch

class Predictor(object):
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(self, data):
        """
        Predict the output using the model.
        :param data: The input data for prediction.
        :return: The predicted output.
        """
        # Ensure the model is in evaluation mode
        self.model.eval()
        with torch.inference_mode():
            batch = read_batch(data, self.device)
            final_logits = self.forward_logits(batch)

        return final_logits
    
    # ========================================================
    # forward
    # ========================================================
    def forward_logits(self, batch) -> torch.Tensor:
        (
            pdbname,
            nuc_x, nuc_evo_x, nuc_coord_x, nuc_edge_index, nuc_edge_weight, RNA_local_R, RNA_local_t, RNA_pred_site,
            residue_x, residue_evo_x, residue_coord_x, residue_edge_index, residue_edge_weight, prot_local_R, prot_local_t, prot_pred_site, protein_domain_mask, protein_domain_lbl,
            RNA_batch, prot_batch, 
            inter_residue_feat_pad,
            _, _, _, _
        ) = batch
        # print(f"Processing {pdbname} with RNA size {nuc_x.shape[0]} and Protein size {residue_x.shape[0]}")
        final_logits = self.model(
            nuc_x, nuc_evo_x, nuc_coord_x, nuc_edge_index, nuc_edge_weight, RNA_local_R, RNA_local_t, RNA_pred_site,
            residue_x, residue_evo_x, residue_coord_x, residue_edge_index, residue_edge_weight, prot_local_R, prot_local_t, prot_pred_site, protein_domain_mask, 
            RNA_batch, prot_batch,
            inter_residue_feat_pad
        )  # [B, h, H, W]

        return final_logits
    
# ========================================================
# Unpack the batch
# ========================================================
def read_batch(data, device):
    pdbname = data.pdbname  # Read complex identifiers from the batch.
    # RNA
    nuc_x = data.RNA_node_aa.to(device)
    nuc_evo_x = data.RNA_node_evo.to(device)
    nuc_coord_x = data.RNA_xyz_coord.to(device)
    nuc_edge_index = data.RNA_edge_index.to(device)
    nuc_edge_weight = data.RNA_edge_weight.to(device)
    # RNA_domain_mask = data.RNA_domain_mask.to(self.device)
    RNA_local_R = data.RNA_local_R.to(device)
    RNA_local_t = data.RNA_local_t.to(device)
    # print(f'RNA_domain_mask: {RNA_domain_mask}')
    
    # Protein
    residue_x = data.prot_node_aa.to(device)
    residue_evo_x = data.prot_node_evo.to(device)
    residue_coord_x = data.prot_xyz_coord.to(device)
    residue_edge_index = data.prot_edge_index.to(device)
    residue_edge_weight = data.prot_edge_weight.to(device)
    # prot_domain_mask = data.prot_domain_mask.to(self.device)
    prot_local_R = data.prot_local_R.to(device)
    prot_local_t = data.prot_local_t.to(device)
    protein_domain_mask = data.protein_domain_mask.to(device)
    protein_domain_lbl = getattr(data, 'protein_domain_lbl', None)
    if protein_domain_lbl is not None:
        protein_domain_lbl = protein_domain_lbl.to(device)

    # batch index
    RNA_batch = data.RNA_node_aa_batch.to(device)
    prot_batch = data.prot_node_aa_batch.to(device)
    inter_residue_feat_pad = _optional_to_device(data, 'inter_residue_feat_pad', device)

    # label (B, H, W)
    conmap_truth_strict_pad = _optional_to_device(data, 'conmap_truth_strict_pad', device)
    conmap_truth_lbl_pad = _optional_to_device(data, 'conmap_truth_lbl_pad', device)
    # Domain_contact_pad = data.Domain_contact_pad.to(self.device)
    RNA_site_truth = _optional_to_device(data, 'RNA_site_truth_pad', device)
    prot_site_truth = _optional_to_device(data, 'prot_site_truth_pad', device)
    RNA_pred_site = data.RNA_pred_site_pad.to(device)
    prot_pred_site = data.prot_pred_site_pad.to(device)

    # print(f'Domain_contact.shape: {Domain_contact.shape}')
    
    
    return (
        pdbname,
        nuc_x, nuc_evo_x, nuc_coord_x, nuc_edge_index, nuc_edge_weight, RNA_local_R, RNA_local_t, RNA_pred_site,
        residue_x, residue_evo_x, residue_coord_x, residue_edge_index, residue_edge_weight, prot_local_R, prot_local_t, prot_pred_site, protein_domain_mask, protein_domain_lbl,
        RNA_batch, prot_batch,
        inter_residue_feat_pad,
        conmap_truth_strict_pad,
        conmap_truth_lbl_pad,
        RNA_site_truth,
        prot_site_truth
    )


def _optional_to_device(data, name, device):
    value = getattr(data, name, None)
    return value.to(device) if value is not None else None
