from __future__ import division
import sys,re,random
from collections import defaultdict
from pprint import pprint
import pickle
##########################
from vit_starter import viterbi  # your vit.py from part 1
OUTPUT_VOCAB = set(""" ! # $ & , @ A D E G L M N O P R S T U V X Y Z ^ """.split())

##########################
# Utilities
def dict_add(vec1, vec2):
    """treat vec1 and vec2 as dict representations of sparse vectors"""
    out = defaultdict(float)
    out.update(vec1)
    for k in vec2: out[k] += vec2[k]
    return out
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

def read_tagging_file(filename):
    """Returns list of sentences from a two-column formatted file.
    Each returned sentence is the pair (tokens, tags) where each of those is a
    list of strings.
    """
    sentences = open(filename).read().strip().split("\n\n")
    ret = []
    for sent in sentences:
        lines = sent.split("\n")
        pairs = [L.split("\t") for L in lines]
        tokens = [tok for tok,tag in pairs]
        tags = [tag for tok,tag in pairs]
        ret.append( (tokens,tags) )
    return ret
###############################

def do_evaluation(examples, weights):
    num_correct,num_total=0,0
    for tokens,goldlabels in examples:
        N = len(tokens); assert N==len(goldlabels)
        predlabels = predict_seq(tokens, weights)
        num_correct += sum(predlabels[t]==goldlabels[t] for t in range(N))
        num_total += N
    print "%d/%d = %.4f accuracy" % (num_correct, num_total, num_correct/num_total)
    return num_correct/num_total

def fancy_eval(examples, weights):
    confusion = defaultdict(float)
    bygold = defaultdict(lambda:{'total':0,'correct':0})
    for tokens,goldlabels in examples:
        predlabels = predict_seq(tokens, weights)
        for pred,gold in zip(predlabels, goldlabels):
            confusion[gold,pred] += 1
            bygold[gold]['correct'] += int(pred==gold)
            bygold[gold]['total'] += 1
        show_predictions(tokens, goldlabels, predlabels)    
    goldaccs = {g: bygold[g]['correct']/bygold[g]['total'] for g in bygold}
    for gold in sorted(goldaccs, key=lambda g: -goldaccs[g]):
        print "gold %s acc %.4f (%d/%d)" % (gold,
                goldaccs[gold],
                bygold[gold]['correct'],bygold[gold]['total'],)
        
def show_predictions(tokens, goldlabels, predlabels):
    print "%-20s %-4s %-4s" % ("word", "gold", "pred")
    print "%-20s %-4s %-4s" % ("----", "----", "----")
    for w, goldy, predy in zip(tokens, goldlabels, predlabels):
        out = "%-20s %-4s %-4s" % (w,goldy,predy)
        if goldy!=predy:
            out += "  *** Error"
        print out

###############################


def train(examples, stepsize=1, numpasses=10, do_averaging=False, devdata=None):
    """
    IMPLEMENT ME !
    Train a perceptron. This is similar to the classifier perceptron training code
    but for the structured perceptron. Examples are now pairs of token and label
    sequences. The rest of the function arguments are the same as the arguments to
    the training algorithm for classifier perceptron.
    """

    weights = defaultdict(float)
    S_t=defaultdict(float)
    current_iter=None
    def get_averaged_weights():
        # IMPLEMENT ME!
        return {k:weights.get(k,0)-S_t.get(k,0)/current_iter for k in set(weights) | set(S_t)}

    for pass_iteration in range(numpasses):
        print "Training iteration %d" % pass_iteration

        for tokens,goldlabels in examples:
            predlabel = predict_seq(tokens, weights)
            g=dict_subtract(features_for_seq(tokens, goldlabels),features_for_seq(tokens, predlabel))
            S_t={k : S_t.get(k,0)+g.get(k,0)*pass_iteration for k in set(S_t) | set(g)}
            weights=dict_add(weights, g)
            
            
        #For perceptron with unaveraged weights# 
            
            
#             predlabel = predict_seq(tokens, weights)
#             g=dict_subtract(features_for_seq(tokens, goldlabels),features_for_seq(tokens, predlabel))
#             weights=defaultdict(float)
#             weights.update(dict_add(weights, g))
        current_iter=pass_iteration+1
        # Evaluation at the end of a training iter
        print "TR  RAW EVAL:",
        do_evaluation(examples, weights)
        if devdata:
            print "DEV RAW EVAL:",
            do_evaluation(devdata, weights)
        if devdata and do_averaging:
            print "DEV AVG EVAL:",
            do_evaluation(devdata, get_averaged_weights())
            
    print "Learned weights for %d features from %d examples" % (len(weights), len(examples))
    
    # NOTE different return value then classperc.py version.
    return weights if not do_averaging else get_averaged_weights()

def predict_seq(tokens, weights):
    """
    IMPLEMENT ME!
    takes tokens and weights, calls viterbi and returns the most likely
    sequence of tags
    """
    
    # predlabels = greedy_decode(Ascores, Bscores, OUTPUT_VOCAB)
    Ascores,Bscores=calc_factor_scores(tokens, weights)
    return viterbi(Ascores,Bscores,OUTPUT_VOCAB)
    

def greedy_decode(Ascores, Bscores, OUTPUT_VOCAB):
    """Left-to-right greedy decoding.  Uses transition feature for prevtag to curtag."""
    N=len(Bscores)
    if N==0: return []
    out = [None]*N
    out[0] = dict_argmax(Bscores[0])
    for t in range(1,N):
        tagscores = {tag: Bscores[t][tag] + Ascores[out[t-1], tag] for tag in OUTPUT_VOCAB}
        besttag = dict_argmax(tagscores)
        out[t] = besttag
    return out

def local_emission_features(t, tag, tokens):
    """
    Feature vector for the B_t(y) function
    t: an integer, index for a particular position
    tag: a hypothesized tag to go at this position
    tokens: the list of strings of all the word tokens in the sentence.
    Retruns a set of features.
    """
    curword = tokens[t]
    feats = {}
    feats["tag=%s_biasterm" % tag] = 1
    feats["tag=%s_curword=%s" % (tag, curword)] = 1

    return feats

def features_for_seq(tokens, labelseq):
    """

    tokens: a list of tokens
    labelseq: a list of output labels
    The full f(x,y) function. Returns one big feature vector.
    This returns a feature vector represented as a dictionary.
    """
    
    f_x_y=defaultdict(int)
    for i in range(len(tokens)):
        b_map=local_emission_features(i, labelseq[i], tokens)
        for keys in b_map.keys():
            f_x_y[keys]+=1
        
        
        pass
    for t in range(len(tokens)-1):
        f_x_y[(labelseq[t],labelseq[t+1])]+=1
    return f_x_y

def calc_factor_scores(tokens, weights):
    """

    tokens: a list of tokens
    weights: perceptron weights (dict)

    returns a pair of two things:
    Ascores which is a dictionary that maps tag pairs to weights
    Bscores which is a list of dictionaries of tagscores per token
    """
    N = len(tokens)
    Ascores = { (tag1,tag2): weights.get((tag1,tag2),0) for tag1 in OUTPUT_VOCAB for tag2 in OUTPUT_VOCAB }
    Bscores = []
    for t in range(N):
        simple_dict=defaultdict(float)
        for tags in OUTPUT_VOCAB:
            key_potential="tag=%s_curword=%s" %(tags,tokens[t])
            key_bias="tag=%s_biasterm" %(tags)
            l_e=local_emission_features(t, tags, tokens)
            simple_dict[tags]=l_e[key_potential]*weights.get(key_potential,0)
            simple_dict[key_bias]=l_e[key_bias]*weights.get(key_bias,0)
        Bscores.append(simple_dict)   
    assert len(Bscores) == N
    return Ascores, Bscores

if __name__ == '__main__':
    training_data=read_tagging_file("oct27.train")
    test_data=read_tagging_file("oct27.dev")
    with open("weights.txt","rb") as fp:
        weights=pickle.load(fp)
    
        
    fancy_eval(test_data[6:7], weights)

#     random.shuffle(training_data)
#     sol_dict = train(training_data, do_averaging=True, devdata=test_data)
#     with open("weights.txt","wb") as fp:
#         pickle.dump(sol_dict, fp)
#     
#     tag_counter=defaultdict(int)
#     a=read_tagging_file("oct27.dev")
#     for i in a:
#         for t in i[1]:
#             tag_counter[t]+=1
#     print "Most common tag: ", dict_argmax(tag_counter)
#     sum_freq=sum(tag_counter.values())
#     acc=tag_counter[dict_argmax(tag_counter)]/sum_freq
#     
#     
#     pass
