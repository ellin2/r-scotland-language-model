def ngram(corpus, gram_size):
    ngrams = {}
    for comment in corpus:
        for i in range(len(comment) - gram_size + 1):
            ngram = ' '.join(comment[i:i+gram_size])
            if ngram in ngrams:
                ngrams[ngram] += 1
            else:
                ngrams[ngram] = 1
    return ngrams

def flip_dict(d):
    counts = {}
    for word, count in d.items():
        if count in counts:
            counts[count] = counts[count] + [word]
        else:
            counts[count] = [word]
    return counts
