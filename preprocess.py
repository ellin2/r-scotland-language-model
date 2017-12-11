import pickle
print('imported pickle')
import numpy
print('imported numpy')
import nltk.tokenize as tokenize
print('imported tokenize')
import re
print('imported re')
import ngrams #we made this file

def binary_search(key,l):
    low_bound = 0
    high_bound = len(l)
    while True:
        check = (low_bound + high_bound)//2
        if check == low_bound:
            return False
        item = l[check]
        if item == key:
            return True
        elif item < key:
            low_bound = check
        else:
            high_bound = check

with open('rscotland_corpus.txt','r',encoding = 'utf-8') as f:
    text = f.read()

print('finished reading file')

tokenized = [tokenize.word_tokenize(re.sub(r'[\*~\^]',r'',comment.lower())) for comment in text.split('\n')]
corpus_word_count = sum(len(sent) for sent in tokenized)
tokenized = [['<comment>'] + comment + ['</comment>'] for comment in tokenized]

print('finished tokenizing')

words = ngrams.ngram(tokenized,1)
print('counted words')

flipped = ngrams.flip_dict(words)
counts = sorted(list(flipped))
print('flipped')

unknowns = [word for i in counts[:3] for word in flipped[i]]
print('enumerated unknowns')

knowns = [word for i in counts[3:] for word in flipped[i]]
print('enumerated knowns')

tokenized = [[(word if binary_search(word,unknowns) else '<unk/>') for word in comment] for comment in tokenized]
print('marked unknowns')

with open('data.pkl','wb') as f:
    pickle.dump([tokenized,knowns,unknowns,corpus_word_count],f)
print('pickled')
