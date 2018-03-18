from __future__ import division
from collections import defaultdict
import pickle
import os
import random

# Global class labels.
POS_LABEL = 'pos'
NEG_LABEL = 'neg'

# Path to dataset
PATH_TO_DATA = "large_movie_review_dataset"
TRAIN_DIR = os.path.join(PATH_TO_DATA, "train")
TEST_DIR = os.path.join(PATH_TO_DATA, "test")


###################################
# Utilities

def dict_add(vec1, vec2):
    """treat vec1 and vec2 as dict representations of sparse vectors"""
    out = defaultdict(float)
    out.update(vec1)
    for k in vec2: out[k] += vec2[k]
    return dict(out)
def dict_subtract(vec1, vec2):
    """treat vec1 and vec2 as dict representations of sparse vectors"""
    out = defaultdict(float)
    out.update(vec1)
    for k in vec2: out[k] -= vec2[k]
    return dict(out)

def dict_argmax(dct):
    """Return the key whose value is largest. In other words: argmax_k dct[k]"""
    return max(dct.iterkeys(), key=lambda k: dct[k])

def dict_dotprod(d1, d2):
    """Return the dot product (aka inner product) of two vectors, where each is
    represented as a dictionary of {index: weight} pairs, where indexes are any
    keys, potentially strings.  If a key does not exist in a dictionary, its
    value is assumed to be zero."""
    smaller = d1 if len(d1)<len(d2) else d2  # BUGFIXED 20151012
    total = 0
    for key in smaller.iterkeys():
        total += d1.get(key,0) * d2.get(key,0)
    return total

def tokenize_doc(doc):
    """
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """

    bow = defaultdict(float)
    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1.0
    return bow

def construct_dataset(train=True):
    """
    Build a dataset. If train is true, build training set otherwise build test set.
    The training set is a list of training examples. A training example is a tuple
    containing a dictionary (that represents features of the training example) and
    a label for that training instance (either 1 or -1).
    """

    dataset = []
    DATA_DIR = TRAIN_DIR if train == True else TEST_DIR
    pos_path = os.path.join(DATA_DIR, POS_LABEL)
    neg_path = os.path.join(DATA_DIR, NEG_LABEL)

    num_inst = 1000
    print "[constructing dataset...]"
    for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
        numeric_label = 1 if p == pos_path else -1
        print "\treading data from %s" % p
        for f in sorted(os.listdir(p))[:num_inst]:
            with open(os.path.join(p,f),'r') as doc:
                content = doc.read()
                bow = tokenize_doc(content)
                dataset.append((bow, numeric_label))
    print "[dataset constructed.]"
    return dataset

###############################


def features_for_label(bow, label):
    """

    The full f(x,y) function. Pass in the bag-of-words for one document
    (represented as a dictionary) and a string label. Returns one big feature
    vector.  The space of keys for this feature vector is used for all classes.
    But for any single call to this function, half of all the features have to be zero.
    """
    feat_vec = defaultdict(float)
    for word, value in bow.iteritems():
        feat_vec["%s_%s" % (label, word)] = value
    return feat_vec

def predict_multiclass(bow, weights):
    """
    Takes a set of features represented as a dictionary and a weight vector that contains
    weights for features of each label (represented as a dictionary) and
    performs perceptron multi-class classification (i.e., finds the class with the highest
    score under the model.
    
    """
    pos_feat_vec = features_for_label(bow, "1")
    neg_feat_vec = features_for_label(bow, "-1")
    scores = { 1: dict_dotprod(pos_feat_vec, weights),
               -1: dict_dotprod(neg_feat_vec, weights) }
    return dict_argmax(scores)

def train(examples, stepsize=1, numpasses=10, do_averaging=False, devdata=None):
    """
    Trains a perceptron.
      examples: list of (featvec, label) pairs; featvec is a dictionary and label is a string
      stepsize: hyperparameter; this affects how the weights are updated
      numpasses: the number of times to loop through the dataset during training
      do_averaging: boolean that determines whether to use averaged perceptron
      devdata: the test set of examples

    returns a dictionary containing the vanilla perceptron accuracy on the train set and test set
    and the averaged perceptron's accuracy on the test set.
    """

    weights = defaultdict(float)
    train_acc = []
    test_acc = []
    avg_test_acc = []
    S_t=defaultdict(float)
    current_iter=None

    def get_averaged_weights():
        return {k:weights.get(k,0)-S_t.get(k,0)/current_iter for k in set(weights) | set(S_t)}

    print "[training...]"
    for pass_iteration in range(numpasses):
        print "\tTraining iteration %d" % pass_iteration
        random.shuffle(examples)
        for bow, goldlabel in examples:
            predlabel = predict_multiclass(bow, weights)
            g=dict_subtract(features_for_label(bow, str(goldlabel)),features_for_label(bow, str(predlabel)))
                  
            weights=dict_add(weights,g)
            S_t={k : S_t.get(k,0)+g.get(k,0)*pass_iteration for k in set(S_t) | set(g)}
            
            
            
                
                
                #For perceptron with no average weight implementation#
#             if predlabel != goldlabel:
#                 g=dict_subtract(features_for_label(bow, str(goldlabel)),features_for_label(bow, str(predlabel)))
#                 weights=dict_add(weights, g)
        current_iter=pass_iteration+1
        print "TR RAW EVAL:",
        train_acc.append(do_evaluation(examples, weights))

        if devdata:
            print "DEV RAW EVAL:",
            test_acc.append(do_evaluation(devdata, weights))

        if devdata and do_averaging:
            print "DEV AVG EVAL:",
            avg_test_acc.append(do_evaluation(devdata, get_averaged_weights()))

    print "[learned weights for %d features from %d examples.]" % (len(weights), len(examples))

    return { 'train_acc': train_acc,
             'test_acc': test_acc,
             'avg_test_acc': avg_test_acc,
             'weights': weights if not do_averaging else get_averaged_weights() }

def do_evaluation(examples, weights):
    """
    Compute the accuracy of a trained perceptron.
    """
    num_correct, num_total = 0, 0
    for feats, goldlabel in examples:
        predlabel = predict_multiclass(feats, weights)
        if predlabel == goldlabel:
            num_correct += 1.0
        num_total += 1.0
    print "%d/%d = %.4f accuracy" % (num_correct, num_total, num_correct/num_total)
    return num_correct/num_total

def plot_accuracy_vs_iteration(train_acc, test_acc, avg_test_acc, naive_bayes_acc):
    """

    Plot the vanilla perceptron accuracy on the trainning set and test set
    and the averaged perceptron accuracy on the test set.
    """
    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    red_patch = mpatches.Patch(color='red', label='Naive Bayes Accuracy')
    blue_patch = mpatches.Patch(color='blue', label='Training Accuracy')
    green_patch = mpatches.Patch(color='green', label='Test Accuracy')
    magenta_patch = mpatches.Patch(color='magenta', label='Average Test Accuracy')
    plt.legend(handles=[red_patch, blue_patch, green_patch, magenta_patch])
    
    
    numpasses=range(1,11)
    naive_bayes_acc=[naive_bayes_acc/100] * 10
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy Score VS Iteration')
    plt.plot(numpasses,naive_bayes_acc,'r')
    plt.plot(numpasses,train_acc,'b')
    plt.plot(numpasses,test_acc,'g')
    plt.plot(numpasses,avg_test_acc,'m')
    plt.show()
     # moved into this function only so it doesnt slow down the overall script if this function doesnt get called
    pass

if __name__=='__main__':
    training_set = construct_dataset(train=True)
    test_set = construct_dataset(train=False)
    sol_dict = train(training_set, do_averaging=True, devdata=test_set)
    with open("av_pe_te_acc.txt", "rb") as fp:
        avg_test_acc=pickle.load(fp)
    
    with open("vanil_te_acc.txt", "rb") as fp:
        test_acc=pickle.load(fp)
    with open("vanil_tr_acc.txt", "rb") as fp:
        train_acc=pickle.load(fp)
    naive_bayes_acc=83.028
    plot_accuracy_vs_iteration(train_acc, test_acc, avg_test_acc, naive_bayes_acc)
    # sol_dict = train(training_set, do_averaging=True, devdata=test_set)
    # plot_accuracy_vs_iteration(sol_dict['train_acc'], sol_dict['test_acc'], sol_dict['avg_test_acc'])
