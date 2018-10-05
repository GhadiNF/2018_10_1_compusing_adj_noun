#
# import pip
# from pip._internal import main
# main(["install","tensorflow"])
#
#




# load libraries
import numpy as np
from gensim import models


# Load word embeddings
# you can try to add more pretrained word embeddings in our collection
# but just loading each file into memory is a time consuming process.
# Loading them all together is not recommended.
embeddings = {
    'w2v-gnews': models.KeyedVectors.load_word2vec_format
    ('GoogleNews-vectors-negative300.bin.gz', binary=True),
}

# read the file and filter those who are not in the embeddings
phrase_annotate = []
with open('AN-phrase-annotations.csv') as f_csv:
    for i, line in enumerate(f_csv):
        if i == 0:
            continue

        adj, noun, is_meta, count = line.strip().split(',')

        is_oov = False
        for title, emb in embeddings.items():
            if noun[:-2] not in emb or adj[:-2] not in emb:
                is_oov = True

        if is_oov:
            continue

        phrase_annotate.append((
            adj[:-2],
            noun[:-2],
            1 if is_meta == 'y' else 0,
            0 if count == '#N/A' else int(count))
        )

adjectives = set(adj for adj, _, _, _ in phrase_annotate)
nouns = set(n for _, n, _, _ in phrase_annotate)

print("""
{0:10} {nadj}
{1:10} {nn}
""".format('adjectives', 'nouns', nadj=len(adjectives), nn=len(nouns)))

import os
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, TimeDistributed
from keras.layers import Input, Flatten, Reshape, Lambda, merge


import keras.backend as K
def recall(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# shuffel the training data:
phrase_annotate_org = phrase_annotate[:]
np.random.shuffle(phrase_annotate)

# First choose an embedding for this part
# embeding {title, total-score, per-adjective-scores}
report = []
for title, emb in embeddings.items():

    ### Prepare the dataset
    # Create the training and testing dataset based on the given embedding:
    X_all = []
    y_all = []

    for adj, noun, is_met, _ in phrase_annotate:
        X_all.append([emb[adj], emb[noun]])
        y_all.append(is_met)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    # split in half for train and test:
    test_split = 500  # int(len(phrase_annotate)/2)
    X_train, y_train = X_all[:test_split], y_all[:test_split]
    X_test, y_test = X_all[test_split:], y_all[test_split:]

    ### Define the network layers
    # Compose two vectors (W)
    model_composer = Sequential()
    model_composer.add(Dense(300, activation='linear', input_shape=(600,)))

    # Map it to one measure (find a vector which maximized
    # the prediction of metaphor) (q)
    model_decoder = Sequential()
    model_decoder.add(Dense(1, activation='sigmoid', input_shape=(300,)))

    # Connecting models
    input_adj = Input(shape=(300,))
    input_noun = Input(shape=(300,))
    input_seq = keras.layers.Concatenate(axis=-1)([input_adj, input_noun])

    out_binary = model_decoder(
        model_composer(input_seq)
    )

    # final model specifications (loss, optimizer, and etc.)
    final_model = Model(input=[input_adj, input_noun], output=out_binary)
    final_model.compile(optimizer='adam',
                        loss='binary_crossentropy',  # good
                        # loss='mse', #good
                        # loss='msle', #mehhh
                        # loss='cosine_proximity', #nope
                        metrics=['accuracy', recall, precision])

    ### Train the network
    final_model.fit([X_train[:, 0], X_train[:, 1]],
    y_train, nb_epoch=20, batch_size=100, validation_split=0.0)

    ### Evaluate the trained network based on the test data
    score = final_model.evaluate([X_test[:, 0], X_test[:, 1]],
     y_test, batch_size=len(X_test))

    # print and save the report
    print("\n")
    print("Embedding:", title)
    for key, value in dict(zip(final_model.metrics_names, score)).items():
        print("{0:10} {1:0.4}".format(key, value))
