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

    (options, args) = parser.parse_args()
    return options

def classify(classifier, data):
    guesses = list(classifier.classify(data))
    num_errors = sum(guess != data[idx].label for idx, guess in enumerate(guesses))
    print "%s error out of %s (%s%%)" % (num_errors, len(data), (num_errors / float(len(data))) * 100)

def main():
    opts = read_opts()
    
    feature_descriptors = faces.list_feature_descriptors((16,16))
    data = []
    print "loading faces..."
    faces.load_data_dir('Face16', 1, feature_descriptors, data, opts.num_faces)
    faces.load_data_dir('Nonface16', -1, feature_descriptors, data, opts.num_other)

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



