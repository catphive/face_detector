from optparse import OptionParser
import random
import Image
import ImageDraw
import time

import faces
import weak_classifier
import boost
import serializer
import search


try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass


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
                      type="int", default=None)
    parser.add_option("-c", "--cbackend", action="store_true", dest="c_backend",
                      help=("Use C++ backend for computing features rapidly. " +
                            "To use this you must first compile the c_faces " +
                            "module with the build.sh script."))
    parser.add_option("--load-classifier", dest="load_classifier",
                      help="Read classifier from file instead of training it.",
                      metavar="FILE")
    parser.add_option("--save-classifier", dest="save_classifier",
                      help="Write classifier to file after training.", metavar="FILE")
    parser.add_option("--each-iter", action="store_true", dest="each_iter",
                      help=("Show validation accuracy after each iteration of " +
                            "boosting, rather than just the final classifier. "))
    parser.add_option("--save-plot-iters", action="store_true", dest="plot_iters",
                      help=("Implies --each-iter. Requires matplotlib."))
    parser.add_option("--save-plot-top-features", action="store_true", dest="plot_features",
                      help=("Saves visual representation of best features. Requires matplotlib."))
    parser.add_option("--search-image", dest="search_image",
                      help="Search image for faces and output marked up file to search_out.png." +
                      "Note that there will be many false positives in some images as we do not" +
                      " implement the cascade.",
                      metavar="FILE")

    (options, args) = parser.parse_args()

    if options.plot_iters:
        options.each_iter = True

    return options

def classify(classifier, data):
    t1 = time.time()
    guesses = list(classifier.classify(data))
    t2 = time.time()
    num_errors = sum(guess != data[idx].label for idx, guess in enumerate(guesses))
    pct_err = (num_errors / float(len(data))) * 100
    print "%d error out of %d (%s%%)" % (num_errors, len(data), pct_err)

    false_positives = sum(guess != data[idx].label and guess == 1 for idx, guess in enumerate(guesses))
    false_negatives = sum(guess != data[idx].label and guess == -1 for idx, guess in enumerate(guesses))

    print "%d false positives" % false_positives
    print "%d false negatives" % false_negatives
    print 'classification took %0.3f ms' % ((t2-t1) * 1000.0)

    return (pct_err, false_positives, false_negatives)

def plot_perf(perf_data):    
    import matplotlib.pyplot as plt

    xs = range(len(perf_data))
    pct_errs = [p[0] for p in perf_data]
    false_positives = [p[1] for p in perf_data]
    false_negatives = [p[2] for p in perf_data]

    plt.plot(xs, pct_errs)
    plt.ylabel("% error")
    plt.xlabel("# iterations")
    plt.savefig("pct_err.png")

    plt.clf()
    plt.cla()

    plt.plot(xs, false_positives)
    plt.ylabel("# false positives")
    plt.xlabel("# iterations")
    plt.savefig("false_positives.png")

    plt.clf()
    plt.cla()

    plt.plot(xs, false_negatives)
    plt.ylabel("# false negatives")
    plt.xlabel("# iterations")
    plt.savefig("false_negatives.png")

    plt.clf()
    plt.cla()

def scale(coord, factor):
    return (coord[0] * factor, coord[1] * factor)

def plot_features(classifier, feature_descriptors):
    factor = 10
    img = Image.open("Face16/c000002.bmp")
    img = img.convert("RGB")
    img = img.resize(scale(img.size, factor))
    draw = ImageDraw.Draw(img)
    
    colors = ["red", "green", "blue", "orange"]

    def cvt(coord):
        return (coord[1], coord[0])

    for base_h in classifier.base_h[:4]:
        feature_func_idx, start, end = feature_descriptors[base_h.f_idx]
        start = cvt(start)
        end = cvt(end)

        draw.rectangle([scale(start, factor), scale(end, factor)],
                       outline=colors[feature_func_idx])
    
    img.save("features_plot.png")

def classify_with_all_iterations(classifier, data, do_plot):
    results = []
    for iters in xrange(1, classifier.iterations + 1):
        print "boosted classifier after %d iterations:" % iters
        # Reconstruct classifier at earlier iterations.
        cls = boost.BoostClassifier(base_h=classifier.base_h[:iters],
                                    alpha=classifier.alpha[:iters],
                                    iterations=iters)

        results.append(classify(cls, data))
    
    if do_plot:
        plot_perf(results)

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
    elif opts.validate_size:
        train_data = []
        validation_data = data[:opts.validate_size]
    else:
        train_data = data
        validation_data = []
        
    if opts.load_classifier:
        with open(opts.load_classifier) as in_file:
            classifier = serializer.load(in_file)
    else:
        print "training boosted classifier..."
        if not train_data:
            print "specify some training data with the -s flag."
            exit(1)
        classifier = boost.train_classifier(train_data, opts.num_iterations)
        print classifier

    if train_data:
        print "training error:"
        classify(classifier, train_data)

    if validation_data:
        print "validation error:"
        if opts.each_iter:
            classify_with_all_iterations(classifier, validation_data, opts.plot_iters)
        else:
            classify(classifier, validation_data)

    if opts.plot_features:
        plot_features(classifier, feature_descriptors)

    if opts.search_image:
        search.search(classifier, opts.search_image, feature_descriptors, opts.c_backend)

    if opts.save_classifier:
        with open(opts.save_classifier, "w") as out_file:
            serializer.dump(classifier, out_file)

if __name__ == "__main__":
    main()
    #import cProfile
    #cProfile.run('main()', 'mainprof')
