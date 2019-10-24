import numpy as np

def prepare_sponsor_party_percs(vote_df):

    r_perc = vote_df['r_perc']
    meta_rep = np.repeat(r_perc[:, np.newaxis], 1, axis=1)

    d_perc = vote_df['d_perc']
    meta_dem = np.repeat(d_perc[:, np.newaxis], 1, axis=1)

    return meta_rep, meta_dem
