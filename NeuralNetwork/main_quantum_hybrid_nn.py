import tensorflow as tf
import pandas as pd
import json
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from keras.layers import Input, Lambda, Reshape, Dense, Dropout, LSTM, Embedding, Convolution2D, MaxPooling2D, Flatten, Concatenate, Bidirectional, GRU, Multiply, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.models import Model
import keras.layers
import sys
import argparse
import os


def prepare_x_y(df, unique_actions, number_of_actions):
    # recover all the actions in order
    #print(df)
    actions = df['activity'].values
    
    timestamps = df.index.values
    #timestamps = pd.to_datetime(timestamps, format='%Y%m%d')
    dates = timestamps.astype('M8[D]')
    df['dates'] = dates
    df.set_index('dates')

    print(('total actions', len(actions)))
    # use tokenizer to generate indices for every action
    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(actions.tolist())
    action_index = tokenizer.word_index  
    
    # translate actions to indexes
    actions_by_index = []
    for action in actions:
        actions_by_index.append(action_index[action])

    df['actions_by_index'] = actions_by_index

    dates_dict = {}
    for index, row in df.iterrows():
        #Remove the 'None' that are tokenized as 'None'
        if str(row['actions_by_index']) != '2':
            activity_list = dates_dict.get(str(index).split(" ")[0], [])
            activity_list.append(row['actions_by_index'])
            dates_dict[str(index).split(" ")[0]] = activity_list
    
    to_remove = []
    for key in dates_dict:
        if len(dates_dict[key]) <= number_of_actions:
            to_remove.append(key)

    for remove in to_remove:
        dates_dict.pop(remove)

    X_actions = []
    y = []

    

    for key in dates_dict:
        last_action = len(dates_dict[key]) - 1

        for i in range(last_action-number_of_actions):
            X_actions.append(actions_by_index[i:i+number_of_actions])
            target_action = ''.join(i for i in actions[i+number_of_actions] if not i.isdigit()) # remove the period if it exists
            y.append(target_action)


    return X_actions, y

def quantumLSTM(total_actions, qnode_a, weight_shapes, n_qubits):
    
    qlayer   = qml.qnn.KerasLayer(qnode_a, weight_shapes, output_dim=n_qubits)
    reshape  = Reshape((n_qubits, 1))
    clayer_3 = keras.layers.LSTM(512, return_sequences=False, name='lstm')
    clayer_4 = Dense(1024, activation='relu', name='dense_1')
    clayer_5 = Dropout(0.8, name='drop_1')
    clayer_6 = Dense(1024, activation='relu', name='dense_2')
    clayer_7 = Dropout(0.8, name='drop_2')
    clayer_8 = Dense(total_actions, activation='softmax', name='output_action')
    model    = tf.keras.models.Sequential([qlayer, reshape, clayer_3,  clayer_4, clayer_5, clayer_6, clayer_7, clayer_8])
    
    return model


def LSTM(total_actions):
    
    clayer_1 = Input(shape=(5,1), name='actions')
    clayer_2 = keras.layers.LSTM(512, return_sequences=False, name='lstm')
    clayer_3 = Dense(1024, activation='relu', name='dense_1')
    clayer_4 = Dropout(0, name='drop_1')
    clayer_5 = Dense(1024, activation='relu', name='dense_2')
    clayer_6 = Dropout(0, name='drop_2')
    clayer_7 = Dense(total_actions, activation='softmax', name='output_action')
    model    = tf.keras.models.Sequential([clayer_1, clayer_2, clayer_3, clayer_4, clayer_5, clayer_6, clayer_7])
    
    return model

def multiQuantumCNN(total_actions, qnode_a, weight_shapes, n_qubits):
    NUMBER_OF_ACTIONS = 5
    actions = Input(shape=(5, 1), name='actions')

    qlayer   = qml.qnn.KerasLayer(qnode_a, weight_shapes, output_dim=(n_qubits,5))(actions)
    reshape = Reshape((n_qubits, 5, 1), name='reshape')(qlayer)

    # convolutional layer
    ngram_2 = Convolution2D(200, (2, 1), padding='valid', activation='relu', name='conv_2')(reshape)
    maxpool_2 = MaxPooling2D(pool_size=(n_qubits-2+1,1), name='pooling_2')(ngram_2)
    ngram_3 = Convolution2D(200, (3, 1), padding='valid', activation='relu', name='conv_3')(reshape)
    maxpool_3 = MaxPooling2D(pool_size=(n_qubits-3+1,1), name='pooling_3')(ngram_3)
    ngram_4 = Convolution2D(200, (4, 1), padding='valid', activation='relu', name='conv_4')(reshape)
    maxpool_4 = MaxPooling2D(pool_size=(n_qubits-4+1,1), name='pooling_4')(ngram_4)
    ngram_5 = Convolution2D(200, (5, 1), padding='valid', activation='relu', name='conv_5')(reshape)
    maxpool_5 = MaxPooling2D(pool_size=(n_qubits-5+1,1), name='pooling_5')(ngram_5)
    merged = Concatenate(axis=2)([maxpool_2, maxpool_3, maxpool_4, maxpool_5])
    flatten = Flatten(name='flatten')(merged)

    # classification layer fully connected with dropout
    dense_1 = Dense(512, activation='relu', name='dense_1')(flatten)
    drop_1 = Dropout(0.8, name='drop_1')(dense_1)
    dense_2 = Dense(512, activation='relu', name='dense_2')(drop_1)
    drop_2 = Dropout(0.8, name='drop_2')(dense_2)
    output_action = Dense(total_actions, activation='softmax', name='output_action')(drop_2)

    model = Model(actions, output_action)
    
    return model


def multiCNN(total_actions):
    
    NUMBER_OF_ACTIONS = 5
    actions = Input(shape=(5, 1), name='actions')

    reshape = Reshape((NUMBER_OF_ACTIONS, 1, 1), name='reshape')(actions)

    # convolutional layer
    ngram_2 = Convolution2D(200, (2, 1), padding='valid', activation='relu', name='conv_2')(reshape)
    maxpool_2 = MaxPooling2D(pool_size=(NUMBER_OF_ACTIONS-2+1,1), name='pooling_2')(ngram_2)
    ngram_3 = Convolution2D(200, (3, 1), padding='valid', activation='relu', name='conv_3')(reshape)
    maxpool_3 = MaxPooling2D(pool_size=(NUMBER_OF_ACTIONS-3+1,1), name='pooling_3')(ngram_3)
    ngram_4 = Convolution2D(200, (4, 1), padding='valid', activation='relu', name='conv_4')(reshape)
    maxpool_4 = MaxPooling2D(pool_size=(NUMBER_OF_ACTIONS-4+1,1), name='pooling_4')(ngram_4)
    ngram_5 = Convolution2D(200, (5, 1), padding='valid', activation='relu', name='conv_5')(reshape)
    maxpool_5 = MaxPooling2D(pool_size=(NUMBER_OF_ACTIONS-5+1,1), name='pooling_5')(ngram_5)
    merged = Concatenate(axis=2)([maxpool_2, maxpool_3, maxpool_4, maxpool_5])
    flatten = Flatten(name='flatten')(merged)

    # classification layer fully connected with dropout
    dense_1 = Dense(512, activation='relu', name='dense_1')(flatten)
    drop_1 = Dropout(0.8, name='drop_1')(dense_1)
    dense_2 = Dense(512, activation='relu', name='dense_2')(drop_1)
    drop_2 = Dropout(0.8, name='drop_2')(dense_2)
    output_action = Dense(total_actions, activation='softmax', name='output_action')(drop_2)

    model = Model(actions, output_action)
    
    return model



def main(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--datasetFileName",
                        type=str,
                        default="base_kasterenA_reduced",
                        nargs="?",
                        help="Dataset file name")
    parser.add_argument("--neuralNetworkType",
                        type=str,
                        default="quantumLSTM",
                        nargs="?",
                        help="Dataset file name")
    
    args = parser.parse_args()
    
    
    directory = os.getcwd()
    
    num_inputs = 5
    num_samples = 20

    print('Loading DATASET...')
    DATASET = directory + "/data/" + args.datasetFileName + ".csv"
    print(DATASET)
    df_har = pd.read_csv(DATASET, parse_dates=[[0]], index_col=0, sep=' ', header=None)
    df_har.columns = ['date','sensor', 'action', 'event', 'activity']
    df_har.index.names = ["timestamp"]
    print('DATASET loaded')
    
    UNIQUE_ACTIONS = directory + "/activities/" + args.datasetFileName + ".json"
    unique_actions = json.load(open(UNIQUE_ACTIONS, 'r'))
    total_actions = len(unique_actions) + 1
    print(total_actions)

    #X_actions, y = prepare_x_y_corrected(df_har, unique_actions, num_inputs) 
    X_actions, y = prepare_x_y(df_har, unique_actions, num_inputs) 

    total_examples = len(X_actions)
    test_per = 0.2
    limit = int(test_per * total_examples)
    X_actions_train = X_actions[limit:]
    X_actions_test = X_actions[:limit]
    y_train = y[limit:]
    y_test = y[:limit]
    #print(('Different actions:'< total_actions))
    print(('Total examples:', total_examples))
    print(('Train examples:', len(X_actions_train), len(y_train))) 
    print(('Test examples:', len(X_actions_test), len(y_test)))
    X_actions_train = np.array(X_actions_train)
    #y_train = np.array(y_train)
    X_actions_test = np.array(X_actions_test)
    #y_test = np.array(y_test)
    print('Shape (X,y):')
    print((X_actions_train.shape))

    n_qubits = 8
    dev = qml.device("default.qubit", wires=n_qubits)
    #dev = qml.device("qiskit.aer", wires=n_qubits)

    @qml.qnode(dev)
    def qnode_a(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    n_layers = 6
    weight_shapes = {"weights": (n_layers, n_qubits)}
    
    if args.neuralNetworkType == "quantumLSTM": 
        model = quantumLSTM(total_actions, qnode_a, weight_shapes, n_qubits)

    elif args.neuralNetworkType == "LSTM": 
        model = LSTM(total_actions)

    elif args.neuralNetworkType == "multiQuantumCNN": 
        model = multiQuantumCNN(total_actions, qnode_a, weight_shapes, n_qubits)
    
    elif args.neuralNetworkType == "multiCNN": 
        model = multiCNN(total_actions)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(unique_actions)
    action_index = tokenizer.word_index

    y_train_final = []
    y_test_final = []

    print(action_index)

    for value in y_train:
        y_train_final.append(action_index[value])

    for value in y_test:
        y_test_final.append(action_index[value])


    fitting = model.fit(X_actions_train.tolist(), 
                        y_train_final, 
                        epochs=100, 
                        batch_size=16, 
                        validation_data=(X_actions_test.tolist(), y_test_final), 
                        verbose=2)


if __name__ == "__main__":
    main(sys.argv)
