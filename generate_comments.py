import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from utils import *
import random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_DIR = 'training_data'
SEQ_LENGTH = 24
BATCH_SIZE = 200
CONF_THRESH = 0.6
MODEL_DIR = 'models'

doc = load_text(DATA_DIR + '/sequences.txt')
lines = doc.split('\n')

from pickle import load

tokenizer = load(open(DATA_DIR + '/tokenizer.pkl', 'rb'))
sequences = tokenizer.texts_to_sequences(lines)
print("- Tokenizer file loaded.")

vocab_size = len(tokenizer.word_index) + 1

model = load_model(MODEL_DIR + '/best_model.h5')
# model = load_model('DT_model.h5')
print("- weights file loaded.")


output = ""

for i in range(50):
    # result = []
    in_text = lines[random.randint(0, len(lines)-1)].split()
    in_text[len(in_text) - 1] = 'endoftweet'
    in_text = ' '.join(in_text)
    output += '\n--------SAMPLE {}-------\n'.format(i + 1)
    output += '----------Seed---------\n' + in_text
    output += '\n-------Generated-------\n'
    for _ in range(10):
        new_tweeet = ''
        while True:
            if len(new_tweeet.split()) >= SEQ_LENGTH:
                in_text += ' endoftweet'
                break
            encoded = tokenizer.texts_to_sequences([in_text])[0]
            encoded = pad_sequences([encoded], maxlen=SEQ_LENGTH, truncating='pre')
            yhat_probs = model.predict(encoded, verbose=0)[0]
            yhat = np.random.choice(len(yhat_probs), 1, p=yhat_probs)
            # yhat = model.predict_classes(encoded, verbose=0)
            out_word = ''
            for word, index in tokenizer.word_index.items():
                if index == yhat:
                    out_word = word
                    break
            in_text += ' ' + out_word
            if out_word == 'endoftweet':
                break
            else:
                new_tweeet += ' ' + out_word
        output += ('-' + new_tweeet + '\n')
        # result.append(new_comment)
    output += '----------Done---------'
# print(output)


f = open("results/" + "results.txt", "a")
f.write(output)
f.close()

print("results written to file.")
