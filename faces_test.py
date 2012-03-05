import unittest
import faces
import numpy
from numpy import testing

class TestIntegrateImage(unittest.TestCase):

    def validate(self, input, output):
        input = numpy.array(input, dtype=numpy.int64, ndmin=2)
        output = numpy.array(output, dtype=numpy.int64, ndmin=2)
        faces.integrate_image(input)
        testing.assert_equal(input, output)
    
    def test_base(self):
        self.validate([], [])
        self.validate([[1]], [[1]])

    def test_top(self):
        self.validate([[1,1,1]], [[1,2,3]])
        self.validate([[4,2,9]], [[4, 6, 15]])

    def test_side(self):
        self.validate([[1],[1]], [[1],[2]])
        self.validate([[4],[9],[3]], [[4],[13],[16]])

    def test_full(self):
        self.validate([[1, 1],[1,1]], [[1,2],[2,4]])
        self.validate([[0,0],[0,0]], [[0,0],[0,0]])
        self.validate([[1,2],[3,4]], [[1,3],[4, 10]])
        self.validate([[1,2,3],[4,5,6]], [[1,3,6],[5,12, 21]])

class TestCoordFuncs(unittest.TestCase):
    
    def test_mid_top(self):
        def validate(in_start, in_end, out):
            self.assertEqual(faces.mid_top(in_start, in_end), out)

        validate((0,0), (1,2), (0,1))
        validate((30,20), (35, 26), (30,23))

    def test_mid_bot(self):
        def validate(in_start, in_end, out):
            self.assertEqual(faces.mid_bot(in_start, in_end), out)

        validate((0,0), (1,2), (1,1))
        validate((30,20), (35, 26), (35,23))

    def test_left_bot(self):
        def validate(in_start, in_end, out):
            self.assertEqual(faces.left_bot(in_start, in_end), out)

        validate((0,0), (1,2), (1,0))
        validate((30,20), (35, 26), (35,20))

    def test_right_top(self):
        def validate(in_start, in_end, out):
            self.assertEqual(faces.right_top(in_start, in_end), out)

        validate((0,0), (1,2), (0,2))
        validate((30,20), (35, 26), (30,26))

    def test_right_top(self):
        def validate(in_start, in_end, out):
            self.assertEqual(faces.left_mid(in_start, in_end), out)

        validate((0,0), (2,1), (1,0))
        validate((30,20), (36, 25), (33,20))

    def test_right_top(self):
        def validate(in_start, in_end, out):
            self.assertEqual(faces.right_mid(in_start, in_end), out)

        validate((0,0), (2,1), (1,1))
        validate((30,20), (36, 25), (33,25))

    def test_mid_top(self):
        def validate(in_start, in_end, out):
            self.assertEqual(faces.first_third_top(in_start, in_end), out)

        validate((0,0), (1,3), (0,1))
        validate((30,20), (35, 26), (30,22))

    def test_mid_top(self):
        def validate(in_start, in_end, out):
            self.assertEqual(faces.first_third_bot(in_start, in_end), out)

        validate((0,0), (1,3), (1,1))
        validate((30,20), (35, 26), (35,22))

class TestFeatureA(unittest.TestCase):
    
    def validate(self, ar, start, end, output):
        ar = numpy.array(ar)
        faces.integrate_image(ar)
        img = faces.IntegratedImage(ar)
        self.assertEqual(faces.feature_a(img, start, end), output)

    def test_min(self):
        self.validate([[0, 0]], (0,0), (1,2), 0)
        self.validate([[1, 0]], (0,0), (1,2), -1)
        self.validate([[0, 1]], (0,0), (1,2), 1)
        self.validate([[2, 0]], (0,0), (1,2), -2)
        self.validate([[0, 2]], (0,0), (1,2), 2)
        self.validate([[2, 1]], (0,0), (1,2), -1)
        self.validate([[1, 2]], (0,0), (1,2), 1)
    
    def test_med(self):
        self.validate([[0,0],[0,0]], (0,0), (2,2), 0)
        self.validate([[1,0],[0,1]], (0,0), (2,2), 0)
        self.validate([[0,0,0,0],[0,0,0,0]], (0,0), (2,4), 0)
        self.validate([[1,0,0,0],[0,0,0,0]], (0,0), (2,4), -1)
        self.validate([[0,0,0,0],[0,3,0,0]], (0,0), (2,4), -3)
        self.validate([[0,0,2,0],[0,0,0,0]], (0,0), (2,4), 2)

    def test_translated(self):
        self.validate([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]], (1,2), (3,4), 0)
        self.validate([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]], (1,2), (2,4), 1)
        self.validate([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]], (1,1), (3,5), 8)

class TestFeatureB(unittest.TestCase):
    
    def validate(self, ar, start, end, output):
        ar = numpy.array(ar)
        faces.integrate_image(ar)
        img = faces.IntegratedImage(ar)
        self.assertEqual(faces.feature_b(img, start, end), output)

    def test_min(self):
        self.validate([[0],[0]], (0,0), (2,1), 0)
        self.validate([[1],[0]], (0,0), (2,1), 1)
        self.validate([[0],[1]], (0,0), (2,1), -1)
        self.validate([[2],[0]], (0,0), (2,1), 2)
        self.validate([[0],[2]], (0,0), (2,1), -2)
        self.validate([[2],[1]], (0,0), (2,1), 1)
        self.validate([[1],[2]], (0,0), (2,1), -1)
    
    def test_med(self):
        self.validate([[0,0],[0,0]], (0,0), (2,2), 0)
        self.validate([[1,0],[0,1]], (0,0), (2,2), 0)
        self.validate([[0,0,0,0],[0,0,0,0]], (0,0), (2,4), 0)
        self.validate([[1,0,0,0],[0,0,0,0]], (0,0), (2,4), 1)
        self.validate([[0,0,0,0],[0,3,0,0]], (0,0), (2,4), -3)
        self.validate([[0,0,2,0],[0,0,0,0]], (0,0), (2,4), 2)

    def test_translated(self):
        self.validate([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]], (1,2), (3,4), 0)
        self.validate([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]], (1,2), (3,3), -5)
        self.validate([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],
                       [16,17,18,19,20],[21,22,23,24,25]],
                      (1,1), (5,3), -40)

if __name__ == '__main__':
    unittest.main()
