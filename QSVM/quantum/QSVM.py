from __future__ import print_function
import json
from string import punctuation
import sys
import argparse
import time
from matplotlib import style
import numpy as np
import pandas as pd
import qiskit
import sklearn
from sklearn import naive_bayes
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

#import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

from qiskit_machine_learning.algorithms import QSVC, VQC
from qiskit_machine_learning.kernels import QuantumKernel

from qiskit import BasicAer, Aer, IBMQ
from qiskit.providers.aer import AerSimulator, noise
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit.utils import QuantumInstance

from pylab import cm
import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import circuit_drawer
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.datasets import ad_hoc_data


class QKTCallback:
    """Callback wrapper class."""

    def __init__(self) -> None:
        self._data = [[] for i in range(5)]

    def callback(self, x0, x1=None, x2=None, x3=None, x4=None):
        """
        Args:
            x0: number of function evaluations
            x1: the parameters
            x2: the function value
            x3: the stepsize
            x4: whether the step was accepted
        """
        self._data[0].append(x0)
        self._data[1].append(x1)
        self._data[2].append(x2)
        self._data[3].append(x3)
        self._data[4].append(x4)

    def get_callback_data(self):
        return self._data

    def clear_callback_data(self):
        self._data = [[] for i in range(5)]

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
            # represent the target action as a one-hot for the softmax
            target_action = ''.join(i for i in actions[i+number_of_actions] if not i.isdigit()) # remove the period if it exists
            #target_action_onehot = np.zeros(len(unique_actions))
            #target_action_onehot[unique_actions.index(target_action)] = 1.0
            y.append(target_action)

    #last_action = len(actions) - 1
    #
    #for i in range(last_action-number_of_actions):
    #    X_actions.append(actions_by_index[i:i+number_of_actions])
    #    # represent the target action as a one-hot for the softmax
    #    target_action = ''.join(i for i in actions[i+number_of_actions] if not i.isdigit()) # remove the period if it exists
    #    #target_action_onehot = np.zeros(len(unique_actions))
    #    #target_action_onehot[unique_actions.index(target_action)] = 1.0
    #    y.append(target_action)
    #
    return X_actions, y

def create_action_embedding_matrix_from_file(tokenizer, vector_file, embedding_size):
    data = pd.read_csv(vector_file, sep=",", header=None)
    data.columns = ["action", "vector"]
    action_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(action_index) + 1, embedding_size))
    unknown_actions = {}    
    for action, i in list(action_index.items()):
        try:
            embedding_vector = np.fromstring(data[data['action'] == action]['vector'].values[0], dtype=float, sep=' ')
            embedding_matrix[i] = embedding_vector        
        except:
            if action in unknown_actions:
                unknown_actions[action] += 1
            else:
                unknown_actions[action] = 1
    
    return embedding_matrix, unknown_actions

def main(argv):
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress=True)
    pd.options.mode.chained_assignment = None
    #parse args
    parser = argparse.ArgumentParser()
    #general args
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="kasteren_house_a/reduced",
                        nargs="?",
                        help="Dataset dir")
    parser.add_argument("--dataset_file",
                        type=str,
                        default="base_kasteren_reduced.csv",
                        nargs="?",
                        help="Dataset file")
    parser.add_argument("--number_of_actions",
                        type=int,
                        default=5,
                        nargs="?",
                        help="Number of actions to input")
    args = parser.parse_args()
    
    sys.stdout.flush()

    print('Loading DATASET...')
    DATASET = args.dataset_dir + "/" + args.dataset_file
    df_har = pd.read_csv(DATASET, parse_dates=[[0]], index_col=0, sep=' ', header=None)
    df_har.columns = ['date','sensor', 'action', 'event', 'activity']
    df_har.index.names = ["timestamp"]
    print('DATASET loaded')


    UNIQUE_ACTIONS = args.dataset_dir + "/" + 'unique_activities.json'
    unique_actions = json.load(open(UNIQUE_ACTIONS, 'r'))
    total_actions = len(unique_actions)

    print("-----------------")
    print("Total actions: "+ str(unique_actions))

    print(('*' * 20))
    print('Preparing dataset...')
    sys.stdout.flush()
    # prepare sequences using action indices
    # each action will be an index which will point to an action vector
    # in the weights matrix of the embedding layer of the network input
    X_actions, y = prepare_x_y(df_har, unique_actions, args.number_of_actions) 

    total_examples = len(X_actions)
    test_per = 0.20
    limit = int(test_per * total_examples)
    X_actions_train = X_actions[limit:]
    X_actions_test = X_actions[:limit]
    y_train = y[limit:]
    y_test = y[:limit]
    print(('Different actions:', total_actions))
    print(('Total examples:', total_examples))
    print(('Train examples:', len(X_actions_train), len(y_train))) 
    print(('Test examples:', len(X_actions_test), len(y_test)))
    sys.stdout.flush()  
    X_actions_train = np.array(X_actions_train)
    y_train = np.array(y_train)
    X_actions_test = np.array(X_actions_test)
    y_test = np.array(y_test)
    print('Shape (X,y):')
    print((X_actions_train.shape))
    print((y_train.shape))

    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]



    classifiers = [
        KNeighborsClassifier(7),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]


    feature_map = ZZFeatureMap(feature_dimension=5, reps=1, entanglement='circular')
 
    file = open("noise_results.txt", "a")
    noise_probs = [0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]

    for noise_prob in noise_probs:
        iterations = 20
        noise_sum = 0
        for i in range(0, iterations):
            #Synthetic noise model for specific 
            #prob_1 = 0.001 # 1-qubit gate
            error_1 = noise.depolarizing_error(noise_prob, 1)
            noise_model = noise.NoiseModel()
            #Apply noise error to all the gates used un ZZFeatureMap
            noise_model.add_all_qubit_quantum_error(error_1, ['x', 'u1', 'h'])

    
            simulator_gpu = AerSimulator(method="statevector", noise_model=noise_model)
            #simulator_gpu = AerSimulator(method="statevector")
            #simulator_gpu.set_options(device='GPU')
            backend = QuantumInstance(simulator_gpu, shots=1024)
            kernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)
    
    
    
            qsvc = QSVC(quantum_kernel=kernel)
            qsvc.fit(X_actions_train, y_train)
            qsvc_score = qsvc.score(X_actions_test, y_test)
            noise_sum = noise_sum + qsvc_score
            noise_sum = 0
            print(f'Iteration {i} Quantum Test score with error of {noise_prob}: {qsvc_score}')
        result = noise_sum / iterations
        print(f'Quantum Test score with error of {noise_prob} for {iterations} iterations: {result}')
        file.write(f'Quantum Test score with error of {noise_prob} for {iterations} iterations: {result}\n')
    file.close()


if __name__ == "__main__":
    main(sys.argv)