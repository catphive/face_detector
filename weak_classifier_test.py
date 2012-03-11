import unittest
import numpy
import faces
import weak_classifier

def norm_label(label):
    if label == True:
        return 1
    elif label == False:
        return -1
    else:
        assert label == 1 or label == -1
        return label

class TestBestThresh(unittest.TestCase):

    def validate(self, features, labels, weights, output):
        assert len(features) == len(labels) == len(weights)
        
        data = [faces.Datum(str(idx),
                            norm_label(labels[idx]),
                            faces.f_vec([features[idx]]))
                for idx in xrange(len(features))]

        self.assertEqual(weak_classifier.best_thresh(data, weights, 0),
                         output)
    
    def test_failure(self):
        with self.assertRaises(AssertionError):
            self.validate([], [], [], (-1, 1, 0))

    def test_single(self):
        self.validate([1], [True], [1], (float("-inf"), -1, 0))
        self.validate([0], [True], [1], (float("-inf"), -1, 0))
        self.validate([0], [False], [1], (float("-inf"), 1, 0))

    def test_double(self):
        self.validate([0, 10], [False, False], [1, 1], (float("-inf"), 1, 0))
        self.validate([0, 10], [False, True], [1, 1], (5, -1, 0))
        self.validate([0, 10], [True, False], [1, 1], (5, 1, 0))
        self.validate([0, 10], [True, True], [1, 1], (float("-inf"), -1, 0))
        
        self.validate([0, 10], [False, False], [1, 0], (float("-inf"), 1, 0))
        self.validate([0, 10], [False, False], [0, 1], (float("-inf"), 1, 0))
        self.validate([0, 10], [False, True], [1, 0], (float("-inf"), 1, 0))
        self.validate([0, 10], [False, True], [0, 1], (float("-inf"), -1, 0))
        self.validate([0, 10], [True, False], [1, 0], (float("-inf"), -1, 0))

    def test_triple(self):
        self.validate([0, 10, 20], [False, False, False], [1, 1, 1], (float("-inf"), 1, 0))
        self.validate([0, 10, 20], [True, False, False], [1, 1, 1], (5, 1, 0))
        self.validate([0, 10, 20], [False, True, False], [1, 1, 1], (float("-inf"), 1, 1))
        self.validate([0, 10, 20], [False, False, True], [1, 1, 1], (15, -1, 0))

        self.validate([0, 10, 20], [False, True, False], [0.5, 1, 1], (15, 1, 0.5))
        self.validate([0, 10, 20], [False, True, False], [1, 1, 0.5], (5, -1, 0.5))

class TestBestFeature(unittest.TestCase):

    def validate(self, feature_grid, labels, weights, output):
        assert len(feature_grid) == len(labels) == len(weights)
        
        data = [faces.Datum(str(idx),
                            norm_label(labels[idx]),
                            faces.f_vec(feature_grid[idx]))
                for idx in xrange(len(feature_grid))]

        self.assertEqual(weak_classifier.best_feature(data, weights),
                         output)

    def test_min(self):
        self.validate([[7]], [True], [1], (0, float("-inf"), -1, 0))
        self.validate([[7, -75]], [True], [1], (0, float("-inf"), -1, 0))
        self.validate([[7],[69]], [True, True], [1,1], (0, float("-inf"), -1, 0))
        self.validate([[7],[69]], [True, False], [1,1], (0, 38, 1, 0))

    def test_square(self):
        self.validate([[7, -75],
                       [69, 30]],
                      [True, True], [1,1], (0, float("-inf"), -1, 0))
        self.validate([[7, -75],
                       [69, 30]],
                      [True, False], [1,1], (0, 38, 1, 0))
        self.validate([[7, -75],
                       [7, 30]],
                      [True, False], [1,1], (1, -22.5, 1, 0))
        self.validate([[7, -75],
                       [7, 30]],
                      [False, True], [1,1], (1, -22.5, -1, 0))

    def test_triple_square(self):
        self.validate([[7, -75, -410],
                       [69, 100, -78],
                       [180, -17, 1]],
                      [False, True, False], [1,1,1], (1, 41.5, -1, 0))

if __name__ == '__main__':
    unittest.main()
