import unittest
import faces
import weak_classifier

class TestBestThresh(unittest.TestCase):

    def validate(self, features, labels, weights, output):
        assert len(features) == len(labels) == len(weights)

        
        data = [faces.Datum(str(idx), [features[idx]], labels[idx])
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

if __name__ == '__main__':
    unittest.main()
