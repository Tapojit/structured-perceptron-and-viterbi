# Predicting POS tagging sequence of tweets using Structured Perceptron and Viterbi

This is an implementation of a structured perceptron to predict **Parts of Speech** (POS) tags of tokens in tweets. It contains **Viterbi algorithm** as a subroutine to infer *POS* sequences and calculate goodness value for loss minimization. 

## POS tags, HMM and Viterbi
**Parts Of Speech** tags are lexical categories under which tokens in a sentence/phrase fall. For instance, *man* is a *Noun*, hence its *POS* tag is *NN* (short for Singular Noun). Similarly, *POS* tag of *going* is *VVG* (short for verb, gerund/present participle).

Check out [this website](https://www.sketchengine.co.uk/penn-treebank-tagset/) for a comprehensive list of *POS* tags.

A sequence of tokens forming a sentence has multiple *POS* tag sequences. A **Hidden Markov Model** (HMM) of the sequence of tokens can be used to determine its most probable POS tag sequence. This HMM is a probabilistic model of all possible POS tag sequences. The log likelihood of a given POS tag sequence *Y* for a sequence of tokens *X* can be determined using the equation below:

![img](https://raw.githubusercontent.com/Tapojit/structured-perceptron-and-viterbi/master/HMM1.png)
