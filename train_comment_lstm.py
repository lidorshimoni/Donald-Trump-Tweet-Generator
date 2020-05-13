import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import os
from utils import *

SEQ_LENGTH = 24
MAX_EPOCHS = 100
BATCH_SIZE = 256
DATA_DIR = 'training_data'
MODEL_DIR = 'models'
EMB_SIZE = 200

from tensorflow.keras.callbacks import ModelCheckpoint

CALLBACKS = [ModelCheckpoint(filepath=MODEL_DIR + '/best_model.h5')]
# CALLBACKS = [ModelCheckpoint(filepath=MODEL_DIR + '/model.{epoch:02d}.h5')]

if not os.path.exists(DATA_DIR + '/sequences.txt'):
    print("- No sequences found. generating new sequences file...")
    comments = load_text(DATA_DIR + '/tweets.txt')
    comments = comments.replace('\n', ' endoftweet ')
    tokens = comments.split()

    print('Total Tokens: ', len(tokens))
    print('Unique Tokens: ', len(set(tokens)))

    length = SEQ_LENGTH + 1
    sequences = []
    for i in range(length, len(tokens)):
        seq = tokens[i - length:i]
        line = ' '.join(seq)
        sequences.append(line)

    # print(sequences[0])

    save_text(sequences, DATA_DIR + '/sequences.txt')

    print('Total Sequences: ', len(sequences))

# ========TOKENIZE SEQUENCES========

doc = load_text(DATA_DIR + '/sequences.txt')
lines = doc.split('\n')

from tensorflow.keras.preprocessing.text import Tokenizer
from pickle import dump, load

if not os.path.exists( DATA_DIR + '/tokenizer.pkl'):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    dump(tokenizer, open( DATA_DIR + '/tokenizer.pkl', 'wb'))

tokenizer = load(open( DATA_DIR + '/tokenizer.pkl', 'rb'))
sequences = tokenizer.texts_to_sequences(lines)

vocab_size = len(tokenizer.word_index) + 1

# ========CREATE MODEL========

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, GRU, Embedding, BatchNormalization

model = Sequential()
# model.add(Input(shape=(SEQ_LENGTH,)))
model.add(Embedding(vocab_size, EMB_SIZE, input_length=SEQ_LENGTH))
model.add(GRU(128))
model.add(Dropout(0.2))
model.add(BatchNormalization(momentum=0.99))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_crossentropy'])

model.summary()

# ========ASSEMBLING TRAINING DATA========
# from tensorflow.keras.utils import to_categorical

sequences = np.array(sequences)
print(sequences.shape)
X, y = sequences[:, :-1], sequences[:, -1]
# y = to_categorical(y, num_classes=vocab_size)

print(X.shape)
print(y.shape)

if os.path.exists(MODEL_DIR + '/DT_model.h5'):
    from tensorflow.keras.models import load_model

    model = load_model(MODEL_DIR + '/DT_model.h5')
    print("Model loaded!!!")

model.fit(X, y, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=CALLBACKS)
model.save(MODEL_DIR + '/DT_model.h5')
