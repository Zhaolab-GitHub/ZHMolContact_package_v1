import pandas as pd
import numpy as np

def build_domain_contact_map(prot_mask_dict, rna_mask_dict, dist_df):
    'Count contacts between protein domains and RNA domains.'
    # 1. Extract the domain-rank sequence from node_mask.
    p_mask = prot_mask_dict['node_mask'].numpy()
    r_mask = rna_mask_dict['node_mask'].numpy()
    
    # 2. Collect all domain-rank IDs.
    p_ranks = sorted(np.unique(p_mask))
    r_ranks = sorted(np.unique(r_mask))
    
    # 3. Initialize counts with protein ranks as rows and RNA ranks as columns.
    contact_matrix = np.zeros((len(p_ranks), len(r_ranks)))
    
    # Map domain-rank IDs to matrix indices.
    p_map = {rank: i for i, rank in enumerate(p_ranks)}
    r_map = {rank: i for i, rank in enumerate(r_ranks)}
    
    # 4. Populate the matrix.
    for _, row in dist_df.iterrows():
        p_res = int(row['residue_number'])
        r_nuc = int(row['nucleotide_number'])
        
        # Check bounds and look up domain ranks.
        if 1 <= p_res <= len(p_mask) and 1 <= r_nuc <= len(r_mask):
            p_rank_id = p_mask[p_res - 1]
            r_rank_id = r_mask[r_nuc - 1]
            
            # Accumulate contact counts in the matrix.
            contact_matrix[p_map[p_rank_id], r_map[r_rank_id]] += 1
            
    # 5. Convert to a DataFrame for display.
    contact_df = pd.DataFrame(
        contact_matrix, 
        index=[f"Prot_domain_{r}" for r in p_ranks],
        columns=[f"RNA_domain_{r}" for r in r_ranks]
    )
    contact_df = contact_df.T
    
    return contact_df