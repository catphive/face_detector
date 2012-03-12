import weak_classifier
from math import log, exp

def error(data, weights, guesses):
    return sum((weights[i] for i, d in enumerate(data) if d.label != guesses[i]),
               0.0)

def calc_alpha(data, weights, guesses):
    er = error(data, weights, guesses)
    if er < 2e-300:
        return 100
    return min(log((1 - er) / er) / 2, 100)

def train_classifier(data, max_iterations):
    assert data
    assert max_iterations

    weights = [1/float(len(data))] * len(data)
    norm = [None] * max_iterations
    # Base hypothesis.
    base_h = [None] * max_iterations
    alpha = [None] * max_iterations

    for iter in xrange(max_iterations):
        base_h[iter] = weak_classifier.train_classifier(data, weights)
        guesses = list(base_h[iter].classify(data))

        alpha[iter] = calc_alpha(data, weights, guesses)
        print "alpha = %s" % alpha[iter]

        norm[iter] = sum((weights[i] * exp(- alpha[iter] * d.label * guesses[i])
                          for i, d in enumerate(data)),
                         0.0)

        weights = [(weights[i] * exp(- alpha[iter] * d.label * guesses[i])) / norm[iter]
                   for i, d in enumerate(data)]

    return BoostClassifier(base_h, alpha, max_iterations)

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

class BoostClassifier(object):

    def __init__(self, base_h, alpha, iterations):
        assert len(base_h) == iterations
        assert len(alpha) == iterations
        self.iterations = iterations
        self.base_h = base_h
        self.alpha = alpha

    def __repr__(self):
        return ("BoostClassifier(base_h=%s, alpha=%s, iterations=%s)" %
                (self.base_h, self.alpha, self.iterations))

    def classify(self, data):
        guesses_table = [list(self.base_h[iter].classify(data))
                         for iter in xrange(self.iterations)]
        for i in xrange(len(data)):
            yield sign(sum(self.alpha[iter] * guesses_table[iter][i]
                           for iter in xrange(self.iterations)))
