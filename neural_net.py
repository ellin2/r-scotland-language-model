import math
print('imported math')
import tensorflow
print('imported tensorflow')
import numpy as np
print('imported numpy')
from preprocess import *

unknowns = []   #it shouldn't need to get used, so I'm resetting unknowns to an empty array
                #so I don't accidentally use it

vocab_size = len(knowns)

word_indices = {}
for i  in range(vocab_size):
    word_indices[knowns[i]] = i

def word2int(word):
    if is_known(word):
        return word_indices[word]
    else:
        return vocab_size
    return vec

def word2vec(word):
    vec = [0] * (vocab_size + 1)
    vec[word2int(word)] = 1
    return vec

def comment2ints(comment):
    assert(comment[0] == '<comment>' and comment[-1] == '</comment>')
    return [word2int(word) for word in comment[1:-1]]

def comment2mtx(comment):
    assert(comment[0] == '<comment>' and comment[-1] == '</comment>')
    return [word2vec(word) for word in comment[1:-1]]

def vec2word(vec):
    if vec[-1] == 1:
        return '<unk/>'
    else:
        return knowns[vec.index(1)]

def softmax(array):
    exps = [math.e**n for n in array]
    total = sum(exps)
    return [n/total for n in exps]

class RNNNumpy:
    def __init__(self, io_size, hidden_size=100, bptt_truncate=4):
        self.io_size = io_size              #size of input and output vectors (vocab size plus one for unknown token)
        self.hidden_size = hidden_size      #size of hidden layer
        self.bptt_truncate = bptt_truncate  #idk yet

        self.U = np.random.uniform(-np.sqrt(1./io_size), np.sqrt(1./io_size), (hidden_size, io_size))
        self.V = np.random.uniform(-np.sqrt(1./hidden_size), np.sqrt(1./hidden_size), (io_size, hidden_size))
        self.W = np.random.uniform(-np.sqrt(1./hidden_size), np.sqrt(1./hidden_size), (hidden_size, hidden_size))

    #x: an array of ints corresponding to words
    def forward_propagate(self, x):
        steps = len(x)
        #an array of hidden layer vectors for each step, plus one for initial hidden layer value
        s = np.zeros((steps + 1, self.hidden_size))
        #an array of output vectors for each step
        o = np.zeros((steps, self.io_size))
        
        for step in range(steps):
            s[step] = np.tanh(self.U[:,x[step]] + self.W.dot(s[step-1]))
            o[step] = softmax(self.V.dot(s[step]))
        return (o, s)

rnn = RNNNumpy(vocab_size+1)
outputs, hiddens = rnn.forward_propagate(comment2ints(tokenized_unks[0]))
