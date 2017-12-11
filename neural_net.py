import theano
print('imported theano')
import pickle
print('imported pickle')

with open('data.pkl','rb') as f:
    tokenized, knowns, unknowns, corpus_word_count = pickle.load(f)

