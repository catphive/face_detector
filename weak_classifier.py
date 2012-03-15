from itertools import izip

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

        if ((err is None or cur_err < err) and
            (idx == 0 or
             pairs[idx - 1][0].features[f_idx] != pairs[idx][0].features[f_idx])):
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
    # Sorting should really be performed ahead of time once for each
    # feature.  Currently this makes the complete boosting algorithm
    # O(M K N log (N)) when it should be O(M K N). Profiling suggests
    # this isn't as big of a deal as you might think as in practice
    # quicksort performs closer linear time.
    pairs = sorted(izip(data, weights), key=lambda pair: pair[0].features[f_idx])
    t_pos = 0
    t_neg = 0
    s_pos = [None] * len(pairs)
    s_neg = [None] * len(pairs)

    s_pos[0] = 0
    s_neg[0] = 0

    for idx, (datum, weight) in enumerate(pairs):
        if datum.label == 1:
            t_pos += weight
            if idx + 1 < len(s_pos):
                s_pos[idx + 1] = s_pos[idx] + pairs[idx][1]
                s_neg[idx + 1] = s_neg[idx]
        else:
            assert datum.label == -1
            t_neg += weight
            if idx + 1 < len(s_neg):
                s_neg[idx + 1] = s_neg[idx] + pairs[idx][1]
                s_pos[idx + 1] = s_pos[idx]

    return thresh_helper(pairs, f_idx, t_pos, t_neg, s_pos, s_neg)

def best_feature(data, weights):
    """returns (feature_index, threshhold, parity, err)"""
    assert data
    assert len(data) == len(weights)
    
    f_idx, (thresh, parity, err) = min(((f_idx, best_thresh(data, weights, f_idx))
                                        for f_idx in xrange(len(data[0].features))),
                                       key=lambda elem: elem[1][2])

    return (f_idx, thresh, parity, err)


def train_classifier(data, weights):
    f_idx, thresh, parity, err = best_feature(data, weights)
    return WeakClassifier(f_idx, thresh, parity, err)

class WeakClassifier(object):

    def __init__(self, f_idx, thresh, parity, expected_err):
        self.f_idx = f_idx
        self.thresh = thresh
        self.parity = parity
        self.expected_err = expected_err
        
    def __repr__(self):
        return ("WeakClassifier(%s, %s, %s, %s)" %
                (self.f_idx, self.thresh, self.parity, self.expected_err))

    def classify(self, data):
        for datum in data:
            if self.parity * datum.features[self.f_idx] < self.parity * self.thresh:
                yield 1
            else:
                yield -1

