# Predicting POS tagging sequence of tweets using Structured Perceptron and Viterbi

This is an implementation of a structured perceptron to predict **Parts of Speech** (POS) tags of tokens in tweets. It contains **Viterbi algorithm** as a subroutine to infer *POS* sequences and calculate goodness value for loss minimization. 

## POS tags, HMM and Viterbi
**Parts Of Speech** tags are lexical categories under which tokens in a sentence/phrase fall. For instance, *man* is a *Noun*, hence its *POS* tag is *NN* (short for Singular Noun). Similarly, *POS* tag of *going* is *VVG* (short for verb, gerund/present participle).

Check out [this website](https://www.sketchengine.co.uk/penn-treebank-tagset/) for a comprehensive list of *POS* tags.

A sequence of tokens forming a sentence has multiple *POS* tag sequences. A **Hidden Markov Model** (HMM) of the sequence of tokens can be used to determine its most probable POS tag sequence. This HMM is a probabilistic model of all possible POS tag sequences. The log likelihood of a given POS tag sequence *Y* for a sequence of tokens *X* can be determined using the equation below:

![img](https://raw.githubusercontent.com/Tapojit/structured-perceptron-and-viterbi/master/HMM.png)

Here, *P<sub>E</sub>* is emission probability and *P<sub>T</sub>* is transition probability. Emission probability is the likelihood of a tag *y<sub>i</sub>* being assigned to token *x<sub>i</sub>*, whereas transition probability is the likelihood that a tag *y<sub>i-1</sub>* is followed by the tag *y<sub>i</sub>*.

Using the equation above, **Viterbi** algorithm dynamically searches for the POS tag sequence with the highest log likelihood in the HMM. You can check out [this video](https://www.youtube.com/watch?v=_568XqOByTs) to understand how this algorithm works.

**vit_starter.py** contains an implementation of the algorithm in the **viterbi** function, which takes as arguments:
1. Dictionary of log transition probabilities with tuple of POS tags as keys.
2. List of dictionaries containing log emission probabilities with POS tags as keys.
3. Set of all possible POS tags.

It returns a list of POS tag sequence which has the highest log likelihood in the HMM.

A test function called **randomized_test** has also been implemented to test the *viterbi* algorithm using arbitrary POS tags, tokens and randomized log probabilities. You can test it out in the python shell of *bash* by running the following lines:

```
>>> import vit_starter.py
>>> vit_starter.randomized_test()

```

Below is the printed output:

```
output_vocab= [0, 1, 2, 3, 4]
A= {(1, 3): 0.48754561565306775, (3, 0): 0.0789558841595761, (2, 1): 0.39402592642278156, (0, 3): 0.6218107404549282, (4, 0): 
0.5099816428700363, (1, 2): 0.7305152028190156, (3, 3): 0.7619172938965952, (4, 4): 0.916092334010546, (2, 2): 
0.3937099783126343, (4, 1): 0.15496847382010293, (1, 1): 0.27704298705927666, (3, 2): 0.36541076384440574, (0, 0): 
0.5026790792580142, (0, 4): 0.031174665332460938, (1, 4): 0.16860428725073673, (2, 3): 0.5361106242084056, (4, 2): 
0.5091227243620162, (1, 0): 0.27038244854390214, (0, 1): 0.5392852188827664, (3, 1): 0.5458502725685433, (2, 4): 
0.5861068442124312, (2, 0): 0.7141848566320921, (4, 3): 0.863195968725216, (3, 4): 0.61212772462278, (0, 2): 
0.33293486231787084}
Bs= [[0.7615196207645206, 0.5937904145636841, 0.7013693739122513, 0.20360693744237235, 0.4506761144710405], 
[0.3812349396218092, 0.4259549050017569, 0.5317918649658191, 0.49813438238468655, 0.7219096114448478], [0.9926458498806451, 
0.9461133009591269, 0.23858466256187394, 0.887562455526099, 0.47061274169180956]]
Worked!

```
*output_vocab* is the POS tag sequence returned by Viterbi. *A* is the transition probability dictionary, whereas *Bs* is the emission probability list. **randomized_test** function uses an exhaustive search algorithm to find the correct POS tag sequence, which is cross-verified against output of **viterbi**. If **viterbi**'s output is incorrect, an assertion error is thrown, otherwise the last line of the printed output is *Worked!*

## Structured Perceptron

A **perceptron** is a machine learning algorithm which is trained on a set of features *x* and labels *y* to make future label predictions on input features. Training occurs in multiple iterations, each of which involves two steps: *inference* & *weight updates*. Before training starts, weight *&theta;* is initialized as a vector of zeroes. Inference step involves predicting labels *y<su>*</su>* of training features using the weight vector and feature vector *f(x, y)*. The predicted labels in the inference step *y<su>*</su>* is compared to the gold labels *y* to determine *loss*; if a predicted label *y<su>*</su><sub>i</sub>* is incorrect, corresponding feature *f(x<sub>i</sub>, y<sub>i</sub>)* is punished in the weight vector by the loss value, otherwise nothing is done to the weight vector. This is the *weight update* step. Each iteration involves different training features and labels, hence weight vector gets updated at each iteration. Over each iteration, the perceptron improves in predicting labels by minimizing loss.

An **averaged perceptron** keeps track of *cumulative loss values* over each iteration, while carrying out weight updates as in a typical perceptron. *Averaged weight* is calculated at the end of each iteration using the *cumulative loss values*. Averaged weights allow for better generalization in the long run.

The purpose of the **structured perceptron** is to predict the POS tag sequence, given a string of tokens. Transition and emission log probabilities, along with a bias are used as weights here. The viterbi algorithm is used to determine the gold POS tag sequence, whose log likelihood is compared to the log likelihood of the predicted POS tag sequence to determine the loss value. This loss value is used to punish and reward the weights during update.

## How to carry out training of structured perceptron

**structperc.py** contains an implementation of *averaged structured perceptron*. You can start training by running **train** function, which takes as arguments:

1. examples: Training dataset
2. stepsize: set to *1* by default
3. numpasses: number of iterations; set to *10* by default
4. do_averaging: takes boolean. Averages weight vector if *True*. Set to *False* by default.
5. devdata: Test dataset, if you want to carry out test at the end of training. Set to *None* by default

In order to carry out training, you can use bash to run the line below from within the repository directory:

```
>>> python structperc.py

```
The training function returns a dictionary of trained weights and prints out the prediction accuracy score for training and test data.

Below is the print output obtained from a training and test session. Training took place over 10 iterations. *TR  RAW EVAL* is training score with unaveraged weights, *DEV RAW EVAL* is test score with unaveraged weights and *DEV AVG EVAL* is test score with averaged weights. The scores are between 0 and 1.

```
Training iteration 0
TR  RAW EVAL: 9951/14619 = 0.6807 accuracy
DEV RAW EVAL: 2783/4823 = 0.5770 accuracy
DEV AVG EVAL: 2783/4823 = 0.5770 accuracy
Training iteration 1
TR  RAW EVAL: 11409/14619 = 0.7804 accuracy
DEV RAW EVAL: 3054/4823 = 0.6332 accuracy
DEV AVG EVAL: 3066/4823 = 0.6357 accuracy
Training iteration 2
TR  RAW EVAL: 11933/14619 = 0.8163 accuracy
DEV RAW EVAL: 2941/4823 = 0.6098 accuracy
DEV AVG EVAL: 3121/4823 = 0.6471 accuracy
Training iteration 3
TR  RAW EVAL: 12377/14619 = 0.8466 accuracy
DEV RAW EVAL: 3072/4823 = 0.6369 accuracy
DEV AVG EVAL: 3196/4823 = 0.6627 accuracy
Training iteration 4
TR  RAW EVAL: 12726/14619 = 0.8705 accuracy
DEV RAW EVAL: 3129/4823 = 0.6488 accuracy
DEV AVG EVAL: 3235/4823 = 0.6707 accuracy
Training iteration 5
TR  RAW EVAL: 12561/14619 = 0.8592 accuracy
DEV RAW EVAL: 3036/4823 = 0.6295 accuracy
DEV AVG EVAL: 3265/4823 = 0.6770 accuracy
Training iteration 6
TR  RAW EVAL: 12862/14619 = 0.8798 accuracy
DEV RAW EVAL: 3147/4823 = 0.6525 accuracy
DEV AVG EVAL: 3270/4823 = 0.6780 accuracy
Training iteration 7
TR  RAW EVAL: 12827/14619 = 0.8774 accuracy
DEV RAW EVAL: 3146/4823 = 0.6523 accuracy
DEV AVG EVAL: 3272/4823 = 0.6784 accuracy
Training iteration 8
TR  RAW EVAL: 12880/14619 = 0.8810 accuracy
DEV RAW EVAL: 3146/4823 = 0.6523 accuracy
DEV AVG EVAL: 3270/4823 = 0.6780 accuracy
Training iteration 9
TR  RAW EVAL: 12868/14619 = 0.8802 accuracy
DEV RAW EVAL: 3131/4823 = 0.6492 accuracy
DEV AVG EVAL: 3271/4823 = 0.6782 accuracy
Learned weights for 21606 features from 1000 examples

```

By default, training in **structperc.py** is carried out on **oct27.train** and testing on **oct27.dev**. Both datasets are parsed by the function **read_tagging_file** in the script. If you want to use your own dataset(s), check out the default datasets and the dataset parser function.

## NOTES

