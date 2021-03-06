# Loss2Vec

## Word2Vec objective customization for improving synonyms and antonyms oriented tasks

By: Eran Nussinovitch, Gregory Pasternak

Meaning & Computation Final Project

![](https://github.com/ednussi/3deception/blob/master/display/figure1.PNG)

## Introduction
From the moment we were introduced to the simplicity and power of word2vec for distributional vector representation of words, we were eager to get our hands dirty with the model through the exploration of its vector space properties in order to understand why it is so successful on analogy tasks, where it comes short, and what can be done.

Throughout exercise 2 and further experiments we noticed that word2vec model performs rather poor when trying to distinguish between synonyms and antonyms of a target word. This seems reasonable when we look at the definition of conceptual similarity that is used wide in basic distributional semantics models: two words are considered to be similar if they appear near the same words. But in case of antonyms this is a weak claim, as for example, adjectives “good” and “bad” may both describe a person or an object in the very same sentences.

In essence we are to create a framework that allows simple modifications to the objective function while training word2vec skip-gram model in order to span vector spaces better suited for tasks other that analogies, e.g. singular/plural distinction, WordNet path embedding, hypernyms/hyponyms distinction, word sense disambiguation, etc. In this project we concentrated on synonyms/antonyms distinction.

## Methods and Materials
### Background & Theory

In regular skip-gram model with negative sampling (SGNS) such as presented in [3] we can write the objective as

<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{w\inV}\sum_{c\inV}\&hash;(w,c)&space;\cdot&space;log(\sigma(sim(w,c)))&plus;k\cdot&space;\&hash;(w\cdot&space;P_0(c)\cdot&space;log(\sigma(-sim(w,c))))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{w\inV}\sum_{c\inV}\&hash;(w,c)&space;\cdot&space;log(\sigma(sim(w,c)))&plus;k\cdot&space;\&hash;(w\cdot&space;P_0(c)\cdot&space;log(\sigma(-sim(w,c))))" title="\sum_{w\inV}\sum_{c\inV}\#(w,c) \cdot log(\sigma(sim(w,c)))+k\cdot \#(w\cdot P_0(c)\cdot log(\sigma(-sim(w,c))))" /></a>

Where V is the vocabulary; 
w,c are target and context words respectively; 
#(w,c)is the number of appearances of the target word with the context word; 
<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma(x)=\frac{1}{1&plus;e^{-x}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma(x)=\frac{1}{1&plus;e^{-x}}" title="\sigma(x)=\frac{1}{1+e^{-x}}" /></a> is a sigmoid function;
<a href="https://www.codecogs.com/eqnedit.php?latex=sim(w_i,w_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?sim(w_i,w_j)" title="sim(w_i,w_j)" /></a> is the cosine similarity between the two embedded vectors of the corresponding words <a href="https://www.codecogs.com/eqnedit.php?latex=w_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_i" title="w_i" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=w_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_j" title="w_j" /></a>.
Thus the first term represents the co-occurrences between a word w and a context c within a predefined context window. The second term refers to negative sampling, where k is the number of negatively sampled words, #(w) is the number of appearances of w as a target word in the unigram distribution <a href="https://www.codecogs.com/eqnedit.php?latex=P_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P_0" title="P_0" /></a> of its negative context c.
We chose to implement a similar approach to that which was presented in [2]. This paper suggest to add a distributional lexical-contrast embedding (dLCE) to the objective: 

<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{w\in&space;V}&space;\sum_{c\in&space;V}&space;\left&space;(\&hash;(w,c)\cdot&space;log(\sigma(sim(w,c)))&plus;&space;k\cdot\&hash;(wP_0(c)\cdot&space;log(\sigma(-sim(w,c)))&space;&plus;\frac{1}{\&hash;(w,u)}&space;\sum_{u\in&space;W(c)\bigcap&space;S(w)}sim(w,u)&plus;&space;\frac{1}{\&hash;(w,v)}&space;\sum_{v\in&space;W(c)\bigcap&space;A(w)}sim(w,v)&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{w\in&space;V}&space;\sum_{c\in&space;V}&space;\left&space;(\&hash;(w,c)\cdot&space;log(\sigma(sim(w,c)))&plus;&space;k\cdot\&hash;(wP_0(c)\cdot&space;log(\sigma(-sim(w,c)))&space;&plus;\frac{1}{\&hash;(w,u)}&space;\sum_{u\in&space;W(c)\bigcap&space;S(w)}sim(w,u)&plus;&space;\frac{1}{\&hash;(w,v)}&space;\sum_{v\in&space;W(c)\bigcap&space;A(w)}sim(w,v)&space;\right&space;)" title="\sum_{w\in V} \sum_{c\in V} \left (\#(w,c)\cdot log(\sigma(sim(w,c)))+ k\cdot\#(wP_0(c)\cdot log(\sigma(-sim(w,c))) +\frac{1}{\#(w,u)} \sum_{u\in W(c)\bigcap S(w)}sim(w,u)+ \frac{1}{\#(w,v)} \sum_{v\in W(c)\bigcap A(w)}sim(w,v) \right )" /></a>

Where the term in the first line is the same as in the original SGNS model, and the second term integrates the lexical contrast information: A(w)and S(w)are sets of antonyms and synonyms of the word w respectively, and W(c)represents words that have positive LMI score, defined as follows [4]:

<a href="https://www.codecogs.com/eqnedit.php?latex=W(c)=&space;\{w&space;\,s.t.\,\forall&space;x\in&space;W&space;\,&space;f(x,c)\cdot&space;log_2(\frac{f(x,c)}{f(x)f(c)}\geq0)\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W(c)=&space;\{w&space;\,s.t.\,\forall&space;x\in&space;W&space;\,&space;f(x,c)\cdot&space;log_2(\frac{f(x,c)}{f(x)f(c)}\geq0)\}" title="W(c)= \{w \,s.t.\,\forall x\in W \, f(x,c)\cdot log_2(\frac{f(x,c)}{f(x)f(c)}\geq0)\}" /></a>

Where f(x) is the frequency of a word x in the corpus.

### Implementation
In order to simplify the problem, we introduced the following modifications:

* Criteria for a word to be considered viable for computing dLCE objective:
  * 100 or more occurrences in corpus
  * Has 1 or more synonyms
  * Has 1 or more antonyms
  For the words that only appeared 100 or more times but don’t meet other requirements, we compute regular SGNS objective. Synonyms and antonyms for all the words in original vocabulary are extracted using NLTK’s WordNet interface. As there are far less antonyms than synonyms for most of the words, we also considered *synonyms* of *antonyms* as antonyms for the target word.
* We implemented a simplified dLCE objective and updated the loss according to:
<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{w\in&space;V}&space;\sum_{c\in&space;V}&space;\left&space;(\&hash;(w,c)\cdot&space;log(\sigma(sim(w,c)))&plus;&space;k\cdot\&hash;(wP_0(c)\cdot&space;log(\sigma(-sim(w,c)))&space;&plus;\frac{1}{\&hash;(w,u)}&space;\sum_{u\in&space;W(c)\bigcap&space;S(w)}sim(w,u)&plus;&space;\frac{1}{\&hash;(w,v)}&space;\sum_{v\in&space;W(c)\bigcap&space;A(w)}sim(w,v)&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{w\in&space;V}&space;\sum_{c\in&space;V}&space;\left&space;(\&hash;(w,c)\cdot&space;log(\sigma(sim(w,c)))&plus;&space;k\cdot\&hash;(wP_0(c)\cdot&space;log(\sigma(-sim(w,c)))&space;&plus;\frac{1}{\&hash;(w,u)}&space;\sum_{u\in&space;W(c)\bigcap&space;S(w)}sim(w,u)&plus;&space;\frac{1}{\&hash;(w,v)}&space;\sum_{v\in&space;W(c)\bigcap&space;A(w)}sim(w,v)&space;\right&space;)" title="\sum_{w\in V} \sum_{c\in V} \left (\#(w,c)\cdot log(\sigma(sim(w,c)))+ k\cdot\#(wP_0(c)\cdot log(\sigma(-sim(w,c))) +\frac{1}{\#(w,u)} \sum_{u\in W(c)\bigcap S(w)}sim(w,u)+ \frac{1}{\#(w,v)} \sum_{v\in W(c)\bigcap A(w)}sim(w,v) \right )" /></a>
  Note: we have implemented the LMI part, but due to runtime/memory problems it was extremely hard to test it on somewhat reasonable corpus.
* When getting the context word for synonyms and antonyms, we only sampled from at most 10000 of all of their contexts (uniformly sampled from the corpus), again because of memory issues.

The training was performed on regular machine with 32GB of RAM memory and 12 cores.
We trained on British National Corpus (99736912 words, 490900 unique words, 34639 frequent words), using window size of 5 on both sides of target word, and a dimension of 200; the training took about 14 hours to train 15 epochs.

## Results
### Preprocessing
Previous to the training, we extracted the following from BNC:
* Word frequencies (using spaCy tokenizer)
* Word synonyms and antonyms (using nltk.wordnet)
* Word contexts for all vocabulary words in order to get context words for synonyms and antonyms in training time
* LMI values for all pairs of vocabulary words

### Building Test Cases
The test consists from word quartets [w1, w2 , w3, w4] such that  w1, w2  and w3, w4are pairs of a word and its antonym respectively. Examples: [absence, presence, comfort, discomfort], [hate, love, forget, remember]. For each such quartet we create a direction vector using the first pair, apply it on the 3rd word and check if the 4th word is in the top 5 proximity of the result <a href="https://www.codecogs.com/eqnedit.php?latex=w_3&plus;(w_2-w_1)\approx&space;w_4" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_3&plus;(w_2-w_1)\approx&space;w_4" title="w_3+(w_2-w_1)\approx w_4" /></a> ,where the approximity sign refers to top 5 proximity.

We refer to this task as “antonym accordance”; despite being somewhat non-intuitive for human definition of antonyms, this task is still a good indicator of relative antonym placement in resulting vector space. In total we had 74526 such test cases, from which 63302 were selected to use on BNC (others included non-vocabulary words as per our definition).

Examples (after PCA dimensionality reduction to 2D):
![](https://github.com/ednussi/loss2vec/blob/master/display/Figure%200.PNG)

### Comparison
As an additional evaluation we also compared between similarity measure of antonyms as computed by SGNS and by our model. Examples (cosine similarity, lower is more similar):

| Word pair  | SGNS | Our  |
| ------------- | ------------- | ------------- |
| absence-presence  | 0.395  | 0.43 |
| accept-refuse | 0.582  | 0.622 |
| advantage-disadvantage  |  0.481  | 0.52 |
| dark-light  | 0.556  | 0.645|
| definite-indefinite  | 0.7  | 0.751 |
| new-old  | 0.663  | 0.99 |
| private-public  | 0.45  | 0.5 | 

![](https://github.com/ednussi/loss2vec/blob/master/display/Figure%201.PNG)

Figure 1: Test accuracy of SGNS model (Word2Vec) from [5] against our improved version (dLCE) as a function of epochs number

### Discussion
As may be seen from the figure above, our model outperformed word2vec SGNS with NCE loss by 1.3% on antonym accordance task (16.3% for our model vs 15.0% for SNGS).
The test accuracy grows consistently with number of training epochs, and with the size of a corpus.

The power of our model comes in the form that we only had a partial list of antonyms and synonyms compared to the test we created, meaning it got the concept of antonyms even for word pairs which it had not seen during the training phase.

We have used the code provided from [5] as SGNS baseline implementation, and as a starting point for loss function customization. The code is attached to this submission. 
To run preprocessing tasks on a corpus:

```
python3 loss2vec/data/scripts/extract_wn_syn_ant.py /path/to/corpus.txt 
(requires nltk with wordnet, spaCy, numpy)
python3 loss2vec/data/scripts/extract_counts.py /path/to/corpus.txt 
(requires spaCy, numpy, pandas)
```

To run our model training (requires tensorflow):
First compile the sampling tensorflow kernels:
```
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0
```

Then run our model training:
```
  python3 word2vec_dlce.py 
--train_data=/path/to/corpus.txt --eval_data=/path/to/loss2vec/data/test-antonyms.txt --save_path=/path/to/save/tensorflow/model 
--syn_threshold=10 
--ant_threshold=2 
--vocabs_root=/path/to/syn/ant/count/context/pickles/folder
```

Please note that the code is not optimized, hence memory usage is not too efficient, and training on larger corpora becomes even more time- and memory-demanding.

## References

[1] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013.

[2] Kim Anh Nguyen, Sabine Schulte im Walde, and Ngoc Thang Vu. 2016. Integrating distributional lexical contrast into word embeddings for antonymsynonym distinction. In Proc. of ACL. pages 454– 459. Jeffrey Penni

[3] Goldberg, Y. & Levy, O. (2014), 'word2vec Explained: deriving Mikolov et al.'s negative-sampling 
word-embedding method' , cite arxiv:1402.3722 .

[4] Chris Biemann and Martin Riedl Computer Science Department, FG Language Technology, TU Darmstadt, Germany Technical Report, TU Darmstadt, Germany, April 2013

[5] Efficient Estimation of Word Representations in Vector Space, Tomas Mikolov , Kai Chen ,Greg Corrado ,Jeffrey Dean, Google Inc. 2013, https://arxiv.org/pdf/1301.3781.pdf
https://github.com/tensorflow/models/tree/master/tutorials/embedding
