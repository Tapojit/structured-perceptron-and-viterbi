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

A **perceptron** is a machine learning algorithm which is trained on a set of features *x* and labels *y* to make future label predictions on input features. Training occurs in multiple iterations, each of which involves two steps: *inference* & *weight updates*. Before training starts, weight *&theta;* is initialized as a vector of zeroes. Inference step involves predicting labels *y<su>*</su>* of training features 
