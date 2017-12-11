import pickle
print('imported pickle')
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

tokenized = [tokenize.word_tokenize(re.sub(r'[\*~\^]',r'',comment.lower())) for comment in text.split('\n')]
total_count = sum(len(sent) for sent in tokenized)
tokenized = [['<comment>'] + comment + ['</comment>'] for comment in tokenized]

print('finished tokenizing')

words = ngrams.ngram(tokenized,1)
print('counted words')

flipped = ngrams.flip_dict(words)
print('flipped')

unknowns = flipped[1] + flipped[2] + flipped[3]
tokenized = [[('<unk/>' if word in unknowns else word) for word in comment] for comment in tokenized]
vocab_size = sum(len(flipped[count]) for count in flipped) - len(unknowns)
print('marked unknowns')

with open('data.pkl','wb') as f:
    pickle.dump([tokenized,unknowns,vocab_size,total_count],f)
print('pickled')
