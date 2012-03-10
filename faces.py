import os
import fnmatch
from os.path import join
import json
import Image
import numpy

def integrate_image(table):
    """Creates integral image/summed area table in place."""
    for c in xrange(table.shape[1]):
        for r in xrange(table.shape[0]):
            accum = 0
            if r > 0:
                table[r, c] += table[r-1, c]
            if c > 0:
                table[r, c] += table[r, c-1]
            if r > 0 and c > 0:
                table[r, c] -= table[r-1, c-1]

class IntegratedImage(object):
    def __init__(self, table):
        self.table = table

    def __getitem__(self, coord):
        assert(coord[0] >= 0)
        assert(coord[1] >= 0)

        if coord[0] == 0 or coord[1] == 0:
            return 0

        return self.table[coord[0] - 1, coord[1] - 1]
    
    def __repr__(self):
        return repr(self.table)

    @property
    def shape(self):
        return self.table.shape

def mid_top(start, end):
    assert (end[1] - start[1]) % 2 == 0
    return (start[0],  start[1] + ((end[1] - start[1]) / 2))

def mid_bot(start, end):
    assert (end[1] - start[1]) % 2 == 0
    return (end[0],  start[1] + ((end[1] - start[1]) / 2))

def left_bot(start, end):
    return (end[0], start[1])

def right_top(start, end):
    return (start[0], end[1])

def left_mid(start, end):
    assert (end[0] - start[0]) % 2 == 0
    return (start[0] + ((end[0] - start[0]) / 2), start[1])

def right_mid(start, end):
    assert (end[0] - start[0]) % 2 == 0
    return (start[0] + ((end[0] - start[0]) / 2), end[1])

def first_third_top(start, end):
    assert (end[1] - start[1]) % 3 == 0
    return (start[0],  start[1] + ((end[1] - start[1]) / 3))

def second_third_top(start, end):
    assert (end[1] - start[1]) % 3 == 0
    return (start[0],  start[1] + (2 * (end[1] - start[1]) / 3))

def first_third_bot(start, end):
    assert (end[1] - start[1]) % 3 == 0
    return (end[0],  start[1] + ((end[1] - start[1]) / 3))

def second_third_bot(start, end):
    assert (end[1] - start[1]) % 3 == 0
    return (end[0],  start[1] + (2 * (end[1] - start[1]) / 3))

def mid_mid(start, end):
    assert (end[0] - start[0]) % 2 == 0
    assert (end[1] - start[1]) % 2 == 0
    return (start[0] + ((end[0] - start[0]) / 2),  start[1] + ((end[1] - start[1]) / 2))

def feature_a(img, start, end):
    assert(start[0] < end[0])
    assert(start[1] + 1 < end[1])

    return (-img[start]
             + 2 * img[mid_top(start, end)]
             - img[right_top(start, end)]
             + img[left_bot(start, end)]
             - 2 * img[mid_bot(start, end)]
             + img[end])

def feature_b(img, start, end):
    assert(start[0] + 1 < end[0])
    assert(start[1] < end[1])

    return (img[start]
            - img[right_top(start, end)]
            - 2 * img[left_mid(start, end)]
            + 2 * img[right_mid(start, end)]
            + img[left_bot(start, end)]
            - img[end])

def feature_c(img, start, end):
    assert(start[0] < end[0])
    assert(start[1] + 2 < end[1])

    return (-img[start]
             + 2 * img[first_third_top(start, end)]
             - 2 * img[second_third_top(start, end)]
             + img[right_top(start, end)]
             + img[left_bot(start, end)]
             - 2 * img[first_third_bot(start, end)]
             + 2 * img[second_third_bot(start, end)]
             - img[end])

def feature_d(img, start, end):
    assert(start[0] + 1 < end[0])
    assert(start[1] + 1 < end[1])

    return (-img[start]
             + 2 * img[mid_top(start, end)]
             - img[right_top(start, end)]
             + 2 * img[left_mid(start, end)]
             - 4 * img[mid_mid(start, end)]
             + 2 * img[right_mid(start, end)]
             - img[left_bot(start, end)]
             + 2 * img[mid_bot(start, end)]
             - img[end])

def all_windows(start, end):
    for start_row in xrange(start[0], end[0]):
        for start_col in xrange(start[1], end[1]):
            for end_row in xrange(start_row + 1, end[0] + 1):
                for end_col in xrange(start_col + 1, end[1] + 1):
                    yield ((start_row, start_col), (end_row, end_col))

def all_features(img, all_start, all_end):
    for start, end in all_windows(all_start, all_end):
        if (end[1] - start[1]) % 2 == 0:
            yield feature_a(img, start, end)

        if (end[0] - start[0]) % 2 == 0:
            yield feature_b(img, start, end)

        if (end[1] - start[1]) % 3 == 0:
            yield feature_c(img, start, end)

        if (end[0] - start[0]) % 2 == 0 and (end[1] - start[1]) % 2 == 0:
            yield feature_d(img, start, end)

def list_img_features(int_img):
    return list(all_features(int_img, (0,0), int_img.shape))

class Datum(object):
    
    def __init__(self, img_path, features, label):
        self.img_path = img_path
        self.features = features
        self.label = label

    def __repr__(self):
        return ("Datum(img_path=%s, features=%s..., label=%s)" %
                (self.img_path, self.features[:5], self.label))

def f_vec(features):
    return numpy.array(features, dtype=numpy.int64)

def load_datum(path, label):
    img = Image.open(path)
    img = img.convert("L")
    #img = img.resize((24,24), Image.ANTIALIAS)
    ar = f_vec(img)
    integrate_image(ar)
    int_img = IntegratedImage(ar)
    return Datum(path,
                 f_vec(list_img_features(int_img)),
                 label)

def load_data_dir(dir, label, data_out, max_load=5):
    for root, dirs, files in os.walk(dir):
        for name in fnmatch.filter(files, "*.bmp"):
            rel_path = join(root, name)
            print "loading:", rel_path
            data_out.append(load_datum(rel_path, label))
            max_load -= 1
            if not max_load:
                break

class DatumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Datum):
            return {"img_path": obj.img_path,
                    "features": obj.features.tolist(),
                    "label": obj.label}
        return json.JSONEncoder.default(self, obj)

def dump(obj, fp):
    json.dump(obj, fp, cls=DatumEncoder)

def datum_decoder(dct):
    if "features" in dct:
        return Datum(dct["img_path"],
                     f_vec(dct["features"]),
                     dct["label"])
    return dct

def load(fp):
    return json.load(fp, object_hook=datum_decoder)

