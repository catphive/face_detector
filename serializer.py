import json
import weak_classifier
import boost

class ClassifierEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, weak_classifier.WeakClassifier):
            return {"f_idx": obj.f_idx,
                    "thresh": obj.thresh,
                    "parity": obj.parity,
                    "expected_err": obj.expected_err}
        if isinstance(obj, boost.BoostClassifier):
            return {"iterations": obj.iterations,
                    "base_h": obj.base_h,
                    "alpha": obj.alpha}
        return json.JSONEncoder.default(self, obj)

def dump(obj, fp):
    json.dump(obj, fp, cls=ClassifierEncoder)

def classifier_decoder(dct):
    if "f_idx" in dct:
        return weak_classifier.WeakClassifier(dct["f_idx"],
                                              dct["thresh"],
                                              dct["parity"],
                                              dct["expected_err"])
    if "base_h" in dct:
        return boost.BoostClassifier(dct["base_h"],
                                     dct["alpha"],
                                     dct["iterations"])
    return dct

def load(fp):
    return json.load(fp, object_hook=classifier_decoder)
