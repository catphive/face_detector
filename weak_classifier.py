
from itertools import (ifilter,
                       ifilterfalse,
                       islice,
                       izip)

def interp(low, high):
    return low + (high - low) / 2.0

def thresh_helper(pairs, f_idx, t_pos, t_neg, s_pos, s_neg):
    """returns (thresh, parity, err)"""
    assert s_pos
    assert len(s_pos) == len(s_neg)
    
    err = None
    thresh = None
    parity = None
    
    for idx, _ in enumerate(s_pos):
        neg_parity_err = s_pos[idx] + (t_neg - s_neg[idx])
        pos_parity_err = s_neg[idx] + (t_pos - s_pos[idx])

        if neg_parity_err < pos_parity_err:
            cur_err = neg_parity_err
            cur_parity = -1
        else:
            cur_err = pos_parity_err
            cur_parity = 1

        if err is None or cur_err < err:
            err = cur_err
            parity = cur_parity
            if idx > 0:
                thresh = interp(pairs[idx - 1][0].features[f_idx],
                                pairs[idx][0].features[f_idx])
            else:
                thresh = float("-inf")
        
    return (thresh, parity, err)

def best_thresh(data, weights, f_idx):
    assert data
    assert len(data) == len(weights)
    pairs = sorted(izip(data, weights), key=lambda pair: pair[0].features[f_idx])
    t_pos = 0
    t_neg = 0
    s_pos = [None] * len(pairs)
    s_neg = [None] * len(pairs)

    s_pos[0] = 0
    s_neg[0] = 0

    for idx, (datum, weight) in enumerate(pairs):
        if datum.label == True:
            t_pos += weight
            if idx + 1 < len(s_pos):
                s_pos[idx + 1] = s_pos[idx] + pairs[idx][1]
                s_neg[idx + 1] = s_neg[idx]
        else:
            assert datum.label == False
            t_neg += weight
            if idx + 1 < len(s_neg):
                s_neg[idx + 1] = s_neg[idx] + pairs[idx][1]
                s_pos[idx + 1] = s_pos[idx]

    return thresh_helper(pairs, f_idx, t_pos, t_neg, s_pos, s_neg)

def best_feature(data, weights):
    """returns (feature_index, threshhold, parity, err)"""

    f_idx, (thresh, parity, err) = min(((f_idx, best_thresh(data, weights, f_idx))
                                        for f_idx in xrange(len(data[0].features))),
                                       key=lambda elem: elem[1][2])

    return (f_idx, thresh, parity, err)
