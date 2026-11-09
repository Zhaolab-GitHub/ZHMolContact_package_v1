import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

class Predictor(object):
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        """
        Predict the output using the model.
        :param data: The input data for prediction.
        :return: The predicted output.
        """
        # Ensure the model is in evaluation mode
        self.model.eval()

        # Perform prediction
        with torch.no_grad():
            RNA_final_scores, _, _, _, _, _, RNA_s, prot_s = self.model(
            # RNA
            nuc_x=data.RNA_node_aa, nuc_evo_x=data.RNA_node_evo,
            nuc_edge_index=data.RNA_edge_index,
            nuc_edge_weight=data.RNA_edge_weight,
            # Protein
            residue_x=data.prot_node_aa, residue_evo_x=data.prot_node_evo,
            residue_edge_index=data.prot_edge_index,
            residue_edge_weight=data.prot_edge_weight,
            # RNA-Protein Interaction batch
            RNA_batch=data.RNA_node_aa_batch, prot_batch=data.prot_node_aa_batch,
            save_cluster = True
            )
            
            # print(f'RNA_final_scores.shape: {RNA_final_scores.shape}')
            RNA_final_scores = torch.sigmoid(RNA_final_scores)
            # print(RNA_final_scores)

        return RNA_final_scores, RNA_s, prot_s
    
    def predict_labels(self, data, threshold=0.5):
        """
        Predict binary labels based on a threshold.
        :param data: The input data for prediction.
        :param threshold: The threshold for converting scores to binary labels.
        :return: The predicted binary labels.
        """
        # Get the predicted scores
        scores = self.predict(data)
        
        # Convert scores to binary labels based on the threshold
        labels = (scores >= threshold).int()
        
        return labels
    


def calculate_performance(final_scores, site_truth, batch_key, threshold=0.5):
    """
    Calculate performance metrics (Precision, Recall, F1) for each batch and return the average.
    :param final_scores: The scores output by the model
    :param site_truth: The true labels
    :param batch_key: The batch indices
    :param threshold: The threshold for converting scores to binary labels
    :return: Precision, Recall, MCC
    """

    unique_batches = torch.unique(batch_key)

    pres = []
    recs = []
    f1s = []

    for batch in unique_batches:
        # Filter the current batch data
        batch_mask = batch_key == batch
        # print(f'batch_mask.shape: {batch_mask.shape}')
        # print(f'batch_mask: {batch_mask}')
        batch_scores = final_scores[batch_mask].detach().cpu().numpy()
        batch_truth = site_truth[batch_mask].detach().cpu().numpy()
        # print(f'batch_scores.shape: {batch_scores.shape}')
        # print(f'batch_truth.shape: {batch_truth.shape}')
        # Check for missing positive or negative classes
        batch_truth_fix = fix_missing_class_labels(batch_truth)
        

        # Calculate Precision

        batch_score_labels = (batch_scores >= threshold).astype(int)

        pre = precision_score(batch_truth, batch_score_labels, zero_division=0)
        # Calculate Recall
        rec = recall_score(batch_truth, batch_score_labels)

        # Calculate F1 score
        f1 = f1_score(batch_truth, batch_score_labels)


        pres.append(pre)  
        recs.append(rec)
        f1s.append(f1)


    aver_pre = sum(pres)/len(pres)
    aver_rec = sum(recs)/len(recs)
    aver_f1 = sum(f1s)/len(f1s)
    return aver_pre, aver_rec, aver_f1


def fix_missing_class_labels(y_true, pos_label=1, neg_label=0):
    """
    Examine whether the binary classification labels are missing positive or negative classes,
    and if so, modify the first label to the missing class.
    :param y_true: One-dimensional label array
    :param pos_label: Positive class label (default 1)
    :param neg_label: Negative class label (default 0)
    :return: The repaired label array (returns original value if both classes are present)
    """
    # Convert to numpy array (compatible with list input)
    y_true = np.array(y_true)

    # Check if it is a one-dimensional array
    if y_true.ndim != 1:
        raise ValueError("y_true must be a one-dimensional array")

    # Get unique labels
    unique_labels = set(y_true)
    has_pos = pos_label in unique_labels
    has_neg = neg_label in unique_labels

    # Case 1: Missing positive class (only negative class present)
    if has_neg and not has_pos:
        # print(f"Warning: Detected all {neg_label} samples, missing positive class {pos_label}, fixed last label")
        y_true[-1] = pos_label  # change last label to positive class
        return y_true

    # Case 2: Missing negative class (only positive class present)
    if has_pos and not has_neg:
        # print(f"Warning: Detected all {pos_label} samples, missing negative class {neg_label}, fixed last label")
        y_true[-1] = neg_label  # change last label to negative class
        return y_true

    # Case 3: Labels are normal (both positive and negative classes are present)
    if has_pos and has_neg:
        # print("Labels are valid: contain both positive and negative class samples")
        return y_true

    # Case 4: No specified positive or negative class in labels (e.g., all 2s or other values)
    raise ValueError(f"Labels do not contain {pos_label} or {neg_label}, please check label values")