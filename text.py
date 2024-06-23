# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint
import random
import sys

# Load the dataset
with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create character to index and index to character mappings
chars = sorted(list(set(text)))
char_indices = {char: i for i, char in enumerate(chars)}
indices_char = {i: char for i, char in enumerate(chars)}

# Prepare the data for training
maxlen = 40  # Length of the sequence for training
step = 3     # Step to move over the text data for creating sequences

sentences = []  # List to store input sequences
next_chars = []  # List to store the next characters for each input sequence

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i + maxlen])
    next_chars.append(text[i + maxlen])

# Vectorization: One-hot encode the characters into binary arrays
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# Build the LSTM model
model = Sequential([
    LSTM(128, input_shape=(maxlen, len(chars))),
    Dense(len(chars), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X, y, batch_size=128, epochs=30)

# Save the trained model
model.save('text_generation_model.h5')

# Function to generate text
def generate_text(length, diversity):
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated = ''
    sentence = text[start_index:start_index + maxlen]
    generated += sentence
    for i in range(length):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    return generated

# Helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Generate text using the trained model
generated_text = generate_text(length=500, diversity=0.5)
print(generated_text)