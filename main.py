import theano #might try tensorflow instead
print('imported theano')
import numpy
print('imported numpy')
import nltk.tokenize as tokenize
print('imported tokenize')
import re
print('imported re')
import ngrams #we made this file

with open('rscotland_corpus.txt','r',encoding = 'utf-8') as f:
    text = f.read()

print('finished reading file')

tokenized = [['<comment>'] + tokenize.word_tokenize(re.sub(r'[\*~\^]',r'',comment.lower())) + ['</comment>'] for comment in text.split('\n')]
total_count = sum(len(sent) for sent in tokenized)

print('finished tokenizing')

words = ngrams.ngram(tokenized,1)
print('counted words')
