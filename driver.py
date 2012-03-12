from optparse import OptionParser
import random
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
    parser.add_option("-s", "--nsample", dest="sample_size",
                      help="randomly select sample of training data to use.",
                      type="int", default=None)
    parser.add_option("-v", "--nvalidate", dest="validate_size",
                      help=("Hold out this many elements for validation. " +
                            "Validation not performed if sample size not specified."),
                      type="int", default=10)
    parser.add_option("-c", "--cbackend", action="store_true", dest="c_backend",
                      help=("Use C++ backend for computing features rapidly. " +
                            "To use this you must first compile the c_faces " +
                            "module with the build.sh script."))

    (options, args) = parser.parse_args()
    return options

def classify(classifier, data):
    guesses = list(classifier.classify(data))
    num_errors = sum(guess != data[idx].label for idx, guess in enumerate(guesses))
    print "%d error out of %d (%s%%)" % (num_errors, len(data), (num_errors / float(len(data))) * 100)

    false_positives = sum(guess != data[idx].label and guess == 1 for idx, guess in enumerate(guesses))
    false_negatives = sum(guess != data[idx].label and guess == -1 for idx, guess in enumerate(guesses))

    print "%d false positives" % false_positives
    print "%d false negatives" % false_negatives

def main():
    opts = read_opts()

    if not opts.c_backend:
        print "WARNING: training in pure python. Run with -c option to enable the (much faster) C++ backend"
    
    feature_descriptors = faces.list_feature_descriptors((16,16))
    data = []
    print "loading faces..."
    faces.load_data_dir('Face16', 1, feature_descriptors, data, opts.num_faces, opts.c_backend)
    faces.load_data_dir('Nonface16', -1, feature_descriptors, data, opts.num_other, opts.c_backend)

    print "suffling..."
    random.shuffle(data)
    if opts.sample_size:
        train_data = data[:opts.sample_size]
        validation_data = data[opts.sample_size : opts.sample_size + opts.validate_size]
    else:
        train_data = data
        validation_data = []

    print "training boosted classifier..."
    classifier = boost.train_classifier(train_data, opts.num_iterations)
    print classifier

    print "training error:"
    classify(classifier, train_data)

    if opts.validate_size and validation_data:
        print "validation error:"
        classify(classifier, validation_data)

if __name__ == "__main__":
    main()
    # import cProfile
    # cProfile.run('main()', 'mainprof')
