import numpy as np
from nltk.corpus import brown
import nltk

# load the data
nltk.download('brown')
nltk.download('universal_tagset')
data = brown.tagged_words(categories='news', tagset='universal')

words, word_class = zip(*data)

# calculate transition:
A = np.zeros((12, 12))
tags = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET',
        'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
tag_dict = {t: idx for idx, t in enumerate(tags)}
for i in range(len(words) - 1):
    A[tag_dict[word_class[i]], tag_dict[word_class[i+1]]] += 1
A /= A.sum(axis=1)[:, np.newaxis]
# print(f"A: {A}")
print("| source \\ target | " + " | ".join(tags) + " |")
print("---".join("|" * (len(tags) + 2)))
for i, j in zip(A, tags):
    print(f"| {j}" + "".join([" | {:.2f}".format(x) for x in i]) + " |")

# calculate emission:
B = np.zeros([12, 7]) + 1e-16
obs = ['science', 'all', 'well', 'like', 'but', 'red', 'dog']
obs_dict = {t: idx for idx, t in enumerate(obs)}
for i in range(len(words) - 1):
    if words[i] in obs:
        B[tag_dict[word_class[i]], obs_dict[words[i]]] += 1
for i in range(12):
    if B[i].sum() > 0:
        B[i] /= B[i].sum()
print("| word \\ observation | " + " | ".join(obs_dict) + " |")
print("---".join("|" * (len(obs_dict) + 2)))
for i, j in zip(B, tags):
    print(f"| {j}" + "".join([" | {:.2f}".format(x) for x in i]) + " |")
