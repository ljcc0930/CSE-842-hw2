# CSE 842 Homework 2
## Question 1

Run

```
python liujia45_hmm.py
```

And you can get a sample of the 2 matrices as below:
### A
| source \ target | ADJ | ADP | ADV | CONJ | DET | NOUN | NUM | PRT | PRON | VERB | . | X |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADJ | 0.06 | 0.07 | 0.01 | 0.03 | 0.01 | 0.71 | 0.02 | 0.02 | 0.00 | 0.02 | 0.07 | 0.00 |
| ADP | 0.08 | 0.02 | 0.01 | 0.00 | 0.44 | 0.31 | 0.06 | 0.01 | 0.03 | 0.04 | 0.01 | 0.00 |
| ADV | 0.12 | 0.16 | 0.08 | 0.02 | 0.08 | 0.06 | 0.02 | 0.03 | 0.04 | 0.27 | 0.13 | 0.00 |
| CONJ | 0.11 | 0.06 | 0.06 | 0.00 | 0.14 | 0.35 | 0.03 | 0.02 | 0.04 | 0.18 | 0.02 | 0.00 |
| DET | 0.23 | 0.01 | 0.01 | 0.00 | 0.01 | 0.65 | 0.02 | 0.00 | 0.01 | 0.05 | 0.01 | 0.00 |
| NOUN | 0.02 | 0.21 | 0.02 | 0.05 | 0.01 | 0.26 | 0.01 | 0.02 | 0.01 | 0.14 | 0.25 | 0.00 |
| NUM | 0.07 | 0.13 | 0.04 | 0.03 | 0.01 | 0.41 | 0.02 | 0.01 | 0.00 | 0.05 | 0.24 | 0.00 |
| PRT | 0.02 | 0.10 | 0.03 | 0.01 | 0.08 | 0.04 | 0.01 | 0.01 | 0.00 | 0.65 | 0.04 | 0.00 |
| PRON | 0.01 | 0.05 | 0.06 | 0.01 | 0.01 | 0.01 | 0.00 | 0.02 | 0.01 | 0.76 | 0.06 | 0.00 |
| VERB | 0.05 | 0.17 | 0.07 | 0.01 | 0.18 | 0.13 | 0.02 | 0.07 | 0.03 | 0.20 | 0.06 | 0.00 |
| . | 0.04 | 0.10 | 0.05 | 0.06 | 0.16 | 0.23 | 0.03 | 0.02 | 0.08 | 0.10 | 0.11 | 0.00 |
| X | 0.00 | 0.05 | 0.01 | 0.01 | 0.00 | 0.12 | 0.00 | 0.01 | 0.00 | 0.02 | 0.22 | 0.55 |
### B
| word \ observation | science | all | well | like | but | red | dog |
|---|---|---|---|---|---|---|---|
| ADJ | 0.00 | 0.00 | 0.50 | 0.25 | 0.00 | 0.25 | 0.00 |
| ADP | 0.00 | 0.00 | 0.00 | 0.79 | 0.21 | 0.00 | 0.00 |
| ADV | 0.00 | 0.23 | 0.77 | 0.00 | 0.00 | 0.00 | 0.00 |
| CONJ | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| DET | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| NOUN | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 |
| NUM | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| PRT | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| PRON | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| VERB | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| . | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| X | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

For B, we only consider 7 words including ['science','all','well','like','but','red','dog'].

## Question 2

1. Run `python liujia45_nn.py` to get the results for one demo setting "embedding_dim=128, hidden_size=64, batch_size=32, epochs=1" and including all the features ("is_capitalized", "word_length", "contains_digit", "contains_hyphen", "is_title").
2. My model architecture is a keras simpleRNN with an embedding layer in the beginning and a dense layer in the end. 
3. For more experiment results, I include all the results as below:

Fixing embedding_dim = 128, hidden_size = 64, batch_size = 32, and epochs = 1:

|                               | Test Acc |
|-------------------------------|----------|
| no features                   | 93.74%   |
| 1 feature (+is_capitalize)    | 93.08%   |
| 2 features (+contains_digit)  | 93.84%   |
| 3 features (+contains_hyphen) | 93.45%   |
| 4 features (+is_title)        | 93.17%   |
| 5 features (+word_length)     | 95.81%   |

Having all 5 features, hidden_size = 64, batch_size = 32, and epochs = 1:

| embedding_dim                      | Test Acc |
|-------------------------------|----------|
| 32                   | 94.41%   |
| 64    |  95.59%   |
| 128  | 95.81%   |

Having all 5 features, embedding_dim = 128, batch_size = 32, and epochs = 1:

| hidden_size                     | Test Acc |
|-------------------------------|----------|
| 16                  | 95.15% |
| 32                   |  96.14% |
| 64    |  95.81%  |

Having all 5 features, embedding_dim = 128, hidden_size = 32, and epochs = 1:

| batch_size                     | Test Acc |
|-------------------------------|----------|
| 4                  | 99.24% |
| 8                  | 98.47% |
| 16                  | 97.56% |
| 32                   |  96.14% |

Having all 5 features, embedding_dim = 128, hidden_size = 32, batch_size= 4:

| Epoch                    | Val Acc |
|-------------------------------|----------|
| 1                  |  99.30% |
| 2                   | 99.57% |
| 3                   | 99.60%  |
| 4                   |  99.63% |
| 5                   |  99.64% |
| 6                  |  99.62% |
| 7                   | 99.62% |
| 8                   | 99.60%  |
| 9                   | 99.59%  |
| 10                   |  99.58% |
| Final Test Acc        | 99.59%  |
