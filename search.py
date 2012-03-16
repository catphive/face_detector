import Image
import ImageDraw
import faces

def plot_search(img, rects):
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    
    colors = ["red", "green", "blue", "orange"]

    def cvt(coord):
        return (coord[1], coord[0])

    for start, end in rects:
        start = cvt(start)
        end = cvt(end)

        draw.rectangle([start, end],
                       outline="red")
    
    img.save("search_out.png")


def all_windows(start, end):
    # Mainly searching in this funky way to get better examples of
    # false positives.
    for size in xrange(16, min(end[0] + 1 - start[0], end[1] + 1 - start[1])):
        for start_row in xrange(start[0], end[0], 2):
            for start_col in xrange(start[1], end[1], 2):
                if size < end[0] + 1 - start_row and size < end[1] + 1 - start_col:
                    yield ((start_row, start_col), (start_row + size, start_col + size))

def search(classifier, img_path, feature_descriptors, c_backend):
    img = Image.open(img_path)
    img = img.convert("L")
    
    print "searching image"
    results = []
    count = 0
    for start, end in all_windows((0,0),
                                  (img.size[1], img.size[0])):
        
        # TODO: refactor code in faces.py so I can call that
        # instead of having this mess right here.
        window = img.crop((start[1], start[0], end[1], end[0]))
        window_pre_resize = window
        window = window.resize((16, 16))
        ar = faces.f_vec(window)
        faces.integrate_image(ar)
        int_img = faces.IntegratedImage(ar)
        
        # Yuck. Needs a little refactoring to make this cleaner.
        if 1 == list(classifier.classify([faces.Datum(
                        img_path,
                        -1,
                        faces.LazyFeatureVec(int_img,
                                             feature_descriptors,
                                             c_backend))]))[0]:
            print "match %s" % repr((start, end))

            results.append((start, end))
            if len(results) > 10:
                break

    plot_search(img, results)
