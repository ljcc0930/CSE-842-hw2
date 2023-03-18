import numpy as np
from nltk.corpus import brown
import nltk

# load the data
nltk.download('universal_tagset')


data = brown.tagged_words(categories='news', tags='universal')
words, word_class = zip(*data)

# calculate transition:
A = np.zeros((12, 12))
tags = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET',
          'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
tag_dict = {t: idx for idx, t in enumerate(tags)}
for i in range(len(words) - 1):
    A[tag_dict[word_class[i]], tag_dict[word_class[i+1]]] += 1
A /= A.sum(axis=1)[:, np.newaxis]
print(f"A: {A}")

# calculate emission:
B = np.zeros([12, 7])
obs = ['science', 'all', 'well', 'like', 'but', 'red', 'dog']
obs_dict = {t: idx for idx, t in enumerate(obs)}
for i in range(len(words) - 1):
    if words[i] in obs:
        B[tag_dict[word_class[i]], obs_dict[words[i]]] += 1
for i in range(12):
    if B[i].sum() > 0:
        B[i] /= B[i].sum()
print(f"B: {B}")
