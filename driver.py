from optparse import OptionParser
import faces
import weak_classifier
import boost

def read_opts():
    parser = OptionParser()
    parser.add_option("-f", "--nfaces", dest="num_faces",
                      help="number of faces to read from the training set",
                      type="int", default=5)
    parser.add_option("-o", "--nother", dest="num_other",
                      help="number of non-faces to read from the training set",
                      type="int", default=5)
    parser.add_option("-i", "--niter", dest="num_iterations",
                      help="number of boosting iterations to perform",
                      type="int", default=1)
    (options, args) = parser.parse_args()
    return options

def main():
    opts = read_opts()
    
    feature_descriptors = faces.list_feature_descriptors((16,16))
    data = []
    faces.load_data_dir('Face16', 1, feature_descriptors, data, opts.num_faces)
    faces.load_data_dir('Nonface16', -1, feature_descriptors, data, opts.num_other)

    print "training boosted classifier..."
    boost_cls = boost.train_classifier(data, opts.num_iterations)
    print boost_cls
    guesses = list(boost_cls.classify(data))
    print "guesses = %s" % guesses
    num_errors = sum(guess != data[idx].label for idx, guess in enumerate(guesses))
    print "%s error out of %s (%s%%)" % (num_errors, len(data), num_errors / float(len(data)))

if __name__ == "__main__":
    main()



