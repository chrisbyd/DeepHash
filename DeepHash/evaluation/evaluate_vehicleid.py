
import numpy as np
from tqdm import tqdm
def eval_vehicleid(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with vehicleid metric
    Key: gallery contains one images for each test vehicles and the other images in test
         use as query
    """
    num_q, num_g = distmat.shape


    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1) # shape [num_q,num_g]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32) #[num_q,num_g]

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in tqdm(range(num_q)):
        # get query pid and camid
        # remove gallery samples that have the same pid and camid with query
        '''
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid) # original remove
        '''
        remove = False  # without camid imformation remove no images in gallery
        keep = np.invert(remove)
        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        # print("The raw_cmc has shape", raw_cmc.shape)
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP