# CSE 842 Homework 2
## Question 1

In this problem, we calculate the Maximum Likelihood Estimation (MLE) for the transition probability matrix (A) and the emission probability matrix (B) of a hidden Markov model (HMM). We use the news category of the Brown corpus with the universal tagset, which consists of 12 part-of-speech (POS) tags.

To run the code, execute the following command:

```
python liujia45_hmm.py
```

And you can get a sample of the 2 matrices as below:
### Transition Probability Matrix (A)
The transition probability matrix A represents the probabilities of moving from one POS tag to another. The rows represent the source tag, and the columns represent the target tag. The values in the cells are the probabilities of transitioning from the source tag to the target tag.
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
### Emission Probability Matrix (B)
The emission probability matrix B represents how often we see a specific word with a particular POS tag. The rows represent the POS tags, and the columns represent the words. The values in the cells are the probabilities of observing a word given a specific POS tag.

In this report, we only consider 7 words, including 'science', 'all', 'well', 'like', 'but', 'red', and 'dog'.

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

### Summary
We have calculated the transition and emission probability matrices for an HMM using the news category of the Brown corpus with the universal tagset. The transition matrix (A) represents the probabilities of moving from one POS tag to another, while the emission matrix (B) represents the probabilities of observing a specific word with a particular POS tag.

## Question 2

In this problem, a Recurrent Neural Network (RNN) is implemented to perform Part-of-speech (POS) tagging. The following features were tested: capitalization, word position in the sentence (first or last), word contains numbers and letters, word has a hyphen, entire word is capitalized, word is a number, first four characters of suffixes and prefixes (e.g., -ed, -ing, -ous), Glove embeddings, etc. At least 5 of these features were incorporated.

The best model architecture is a Keras simpleRNN with an embedding layer in the beginning and a dense layer in the end. To run the code, execute:

```
python liujia45_nn.py
```

The model was tested under various hyperparameters, including embedding dimension, hidden layer size, batch size, and epochs. The results of these experiments are shown in the tables below.

### Experiment 1: Varying the number of features
Fixing embedding_dim = 128, hidden_size = 64, batch_size = 32, and epochs = 1:

| Features Included                  | Test Accuracy |
|------------------------------------|---------------|
| no features                        | 93.74%        |
| 1 feature (+is_capitalize)         | 93.08%        |
| 2 features (+contains_digit)       | 93.84%        |
| 3 features (+contains_hyphen)      | 93.45%        |
| 4 features (+is_title)             | 93.17%        |
| 5 features (+word_length)          | 95.81%        |

### Experiment 2: Varying embedding dimension
Having all 5 features, hidden_size = 64, batch_size = 32, and epochs = 1:

| embedding_dim | Test Accuracy |
|---------------|---------------|
| 32            | 94.41%        |
| 64            | 95.59%        |
| 128           | 95.81%        |

### Experiment 3: Varying hidden layer size
Having all 5 features, embedding_dim = 128, batch_size = 32, and epochs = 1:

| hidden_size | Test Accuracy |
|-------------|---------------|
| 16          | 95.15%        |
| 32          | 96.14%        |
| 64          | 95.81%        |

### Experiment 4: Varying batch size
Having all 5 features, embedding_dim = 128, hidden_size = 32, and epochs = 1:

| batch_size | Test Accuracy |
|------------|---------------|
| 4          | 99.24%        |
| 8          | 98.47%        |
| 16         | 97.56%        |
| 32         | 96.14%        |

### Experiment 5: Varying the number of epochs
Having all 5 features, embedding_dim = 128, hidden_size = 32, batch_size = 4:

| Epoch | Validation Accuracy |
|-------|---------------------|
| 1     | 99.30%              |
| 2     | 99.57%              |
| 3     | 99.60%              |
| 4     | 99.63%              |
| 5     | 99.64%              |
| 6     | 99.62%              |
| 7     | 99.62%              |
| 8     | 99.60%              |
| 9     | 99.59%              |
| 10    | 99.58%              |
| Final Test Accuracy | 99.59%  |

From the experiments, it can be observed that the model's performance improves with the inclusion of more features. Additionally, the optimal hyperparameters are found to be an embedding dimension of 128, hidden layer size of 32, batch size of 4, and training for multiple epochs. The model achieves a final test accuracy of 99.59% under these conditions.

