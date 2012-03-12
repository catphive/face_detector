#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <utility>
#include <assert.h>

using std::cout;
using std::endl;

using std::make_pair;
typedef std::pair<int,int> coord_type;

namespace {
    template <typename T>
    struct obj_ref {
    
        obj_ref(T* obj = NULL): m_obj(obj) {
        }

        ~obj_ref() {
            Py_CLEAR(m_obj);
        }

        void incr() {
            Py_INCREF(m_obj);
        }

        T*& get() {
            return m_obj;
        }

    private:
        T* m_obj;
        
        obj_ref(const obj_ref&);
        obj_ref& operator=(const obj_ref&);
    };

    struct image {
        image(PyArrayObject* img): m_img_ref(img) {
            m_img_ref.incr();
        }

        bool validate() {
            if (m_img_ref.get()->descr->type_num != NPY_INT32 ||
                m_img_ref.get()->nd != 2 ||
                !PyArray_ISCONTIGUOUS(m_img_ref.get()) ||
                !PyArray_ISALIGNED(m_img_ref.get()) ||
                !PyArray_ISCARRAY(m_img_ref.get())) {
                PyErr_SetString(PyExc_ValueError,
                                "Integrated image array must be 2d int array.");
                return false;
            }
        
            return true;
        }

        int operator[](coord_type coord) {
            return (*this)(coord.first, coord.second);
        }
        
        // This uses python imaging library style indexing.
        int operator()(int row, int col) {
            assert(col >= 0);
            assert(row >= 0);
            
            if (col == 0 || row == 0) {
                return 0;
            }
            
            --col;
            --row;

            int* data = static_cast<int*>(PyArray_DATA(m_img_ref.get()));
            int num_cols = dim(1);
            int idx = row * num_cols + (col % num_cols);
            assert(idx < dim(0) * dim(1));
            
            return data[idx];
        }

        int dim(int n) {
            return PyArray_DIM(m_img_ref.get(), n);
        }

    private:
        obj_ref<PyArrayObject> m_img_ref;

        image(const image&);
        image& operator=(const image&);
    };

    std::ostream& operator<<(std::ostream& stream, image& img) {
        stream << "[";
        for (int row = 1; row <= img.dim(0); ++row) {
            stream << "[";
            for (int col = 1; col <= img.dim(1); ++col) {
                stream << img(row, col) << ", ";
            }
            stream << "]\n";
        }
        stream << "]" << endl;
        return stream;
    }

    coord_type mid_top(coord_type start, coord_type end) {
        assert((end.second - start.second) % 2 == 0);
        return make_pair(start.first, start.second + ((end.second - start.second) / 2));
    }

    coord_type mid_bot(coord_type start, coord_type end) {
        assert((end.second - start.second) % 2 == 0);
        return make_pair(end.first, start.second + ((end.second - start.second) / 2));
    }

    coord_type left_bot(coord_type start, coord_type end) {
        return make_pair(end.first, start.second);
    }

    coord_type right_top(coord_type start, coord_type end) {
        return make_pair(start.first, end.second);
    }

    coord_type left_mid(coord_type start, coord_type end) {
        assert((end.first - start.first) % 2 == 0);
        return make_pair(start.first + ((end.first - start.first) / 2), start.second);
    }

    coord_type right_mid(coord_type start, coord_type end) {
        assert((end.first - start.first) % 2 == 0);
        return make_pair(start.first + ((end.first - start.first) / 2), end.second);
    }

    coord_type first_third_top(coord_type start, coord_type end) {
        assert((end.second - start.second) % 3 == 0);
        return make_pair(start.first, start.second + ((end.second - start.second) / 3));
    }
    
    coord_type second_third_top(coord_type start, coord_type end) {
        assert((end.second - start.second) % 3 == 0);
        return make_pair(start.first, start.second + (2 * (end.second - start.second) / 3));
    }

    coord_type first_third_bot(coord_type start, coord_type end) {
        assert((end.second - start.second) % 3 == 0);
        return make_pair(end.first, start.second + ((end.second - start.second) / 3));
    }

    coord_type second_third_bot(coord_type start, coord_type end) {
        assert((end.second - start.second) % 3 == 0);
        return make_pair(end.first, start.second + (2 * (end.second - start.second) / 3));
    }

    coord_type mid_mid(coord_type start, coord_type end) {
        assert((end.first - start.first) % 2 == 0);
        assert((end.second - start.second) % 2 == 0);
        return make_pair(start.first + ((end.first - start.first) / 2),
                         start.second + ((end.second - start.second) / 2));
    }

    int feature_a(image& img, coord_type start, coord_type end) {
        assert(start.first < end.first);
        assert(start.second + 1 < end.second);
        return (-img[start]
                + 2 * img[mid_top(start, end)]
                - img[right_top(start, end)]
                + img[left_bot(start, end)]
                - 2 * img[mid_bot(start, end)]
                + img[end]);
    }

    int feature_b(image& img, coord_type start, coord_type end) {
        assert(start[0] + 1 < end[0]);
        assert(start[1] < end[1]);

        return (img[start]
                - img[right_top(start, end)]
                - 2 * img[left_mid(start, end)]
                + 2 * img[right_mid(start, end)]
                + img[left_bot(start, end)]
                - img[end]);
    }

    int feature_c(image& img, coord_type start, coord_type end) {
        assert(start[0] < end[0]);
        assert(start[1] + 2 < end[1]);

        return (-img[start]
                + 2 * img[first_third_top(start, end)]
                - 2 * img[second_third_top(start, end)]
                + img[right_top(start, end)]
                + img[left_bot(start, end)]
                - 2 * img[first_third_bot(start, end)]
                + 2 * img[second_third_bot(start, end)]
                - img[end]);
    }

    int feature_d(image& img, coord_type start, coord_type end) {
        assert(start[0] + 1 < end[0]);
        assert(start[1] + 1 < end[1]);

        return (-img[start]
                + 2 * img[mid_top(start, end)]
                - img[right_top(start, end)]
                + 2 * img[left_mid(start, end)]
                - 4 * img[mid_mid(start, end)]
                + 2 * img[right_mid(start, end)]
                - img[left_bot(start, end)]
                + 2 * img[mid_bot(start, end)]
                - img[end]);
    }
}

static
PyObject* c_faces_lazy_get_feature(PyObject *self, PyObject *args) {
    int feature_func_idx;
    int start0 = -50;
    int start1 = -50;
    int end0 = -50;
    int end1 = -50;
    PyArrayObject* int_img_array;

    if (!PyArg_ParseTuple(args, "i(ii)(ii)O!",
                          &feature_func_idx,
                          &start0,
                          &start1,
                          &end0,
                          &end1,
                          &PyArray_Type,
                          &int_img_array)) {
        return NULL;
    }

    image int_img(int_img_array);

    #ifndef NDEBUG
    if(!int_img.validate()) {
        return NULL;
    }
    #endif

    coord_type start(start0, start1);
    coord_type end(end0, end1);

    int res = 0;
    switch(feature_func_idx) {
    case 0:
        res = feature_a(int_img, start, end);
        break;
    case 1:
        res = feature_b(int_img, start, end);
        break;
    case 2:
        res = feature_c(int_img, start, end);
        break;
    case 3:
        res = feature_d(int_img, start, end);
        break;
    default:
        // should never get here.
        return NULL;
    }

    return PyInt_FromLong(res);
}

static PyMethodDef FacesMethods[] = {
    {"lazy_get_feature",  c_faces_lazy_get_feature, METH_VARARGS,
     "Extract a featured from an integrated image."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initc_faces(void)
{
    (void) Py_InitModule("c_faces", FacesMethods);
    import_array();
}




