import faces
import weak_classifier

def main():

    with open("train.json") as train_file:
        data = faces.load(train_file)
    
    print "weak_classifier = %s" % repr(weak_classifier.best_feature(data, [1.0 / len(data) for _ in data]))

if __name__ == "__main__":
    main()



