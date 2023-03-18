from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, SimpleRNN, Dense
from keras.models import Sequential, Model
from keras.layers import Input, Concatenate
import numpy as np
import nltk

# load the data
nltk.download('universal')

# from IPython import embed


tagged_sents = nltk.corpus.treebank.tagged_sents(tagset='universal')

idx_words = {}
idx_tags = {}

for sentence in tagged_sents:
    for word, tag in sentence:
        if word not in idx_words:
            idx_words[word] = len(idx_words) + 1
        if tag not in idx_tags:
            idx_tags[tag] = len(idx_tags) + 1

vocab_size = len(idx_words) + 1
tag_size = len(idx_tags) + 1

X = []
y = []


def is_capitalized(word):
    return int(word[0].isupper())


def cnt_digits(word):
    return int(any(char.isdigit() for char in word))


def cnt_hyphen(word):
    return int('-' in word)


def is_title(word):
    return int(word in ['Mr.', 'Mrs.', 'Miss', 'Ms.', 'Dr.', 'Prof.'])


def get_feature_vec(word_index, capitalized, length, have_digit, have_hyphen, title):
    return np.array([word_index, capitalized, length, have_digit, have_hyphen, title])


for sentence in tagged_sents:
    X_sentence = []
    for word, tag in sentence:
        word_index = idx_words[word]
        capitalized = is_capitalized(word)
        length = len(word)
        have_digit = cnt_digits(word)
        have_hyphen = cnt_hyphen(word)
        title = is_title(word)
        feature_vector = get_feature_vec(
            word_index, capitalized, length, have_digit, have_hyphen, title)
        X_sentence.append(feature_vector)
    X.append(X_sentence)
    y.append([idx_tags[tag] for _, tag in sentence])


max_sequence_length = max(len(seq) for seq in X)
X_padded = pad_sequences(X, maxlen=max_sequence_length,
                         padding='post', dtype='float32')
y_padded = pad_sequences(y, maxlen=max_sequence_length, padding='post')


y_categorical = to_categorical(y_padded, num_classes=tag_size)
X_train, X_val, y_train, y_val = train_test_split(
    X_padded, y_categorical, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_val, y_val, test_size=0.5, random_state=42)

embedding_dim = 128
hidden_size = 64
feature_size = 5  # Number of additional features

input_layer = Input(shape=(max_sequence_length, feature_size + 1))
word_indices = input_layer[:, :, 0]
additional_features = input_layer[:, :, 1:]
word_embeddings = Embedding(vocab_size, embedding_dim)(word_indices)
expanded_features = Dense(
    embedding_dim, activation='linear')(additional_features)
combined_features = Concatenate()([word_embeddings, expanded_features])
rnn_output = SimpleRNN(hidden_size, return_sequences=True)(combined_features)
tag_probabilities = Dense(tag_size, activation='softmax')(rnn_output)
model = Model(inputs=input_layer, outputs=tag_probabilities)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])


batch_size = 32
epochs = 1
model.fit(X_train, y_train, batch_size=batch_size,
          epochs=epochs, validation_data=(X_val, y_val))

_, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy}!")
