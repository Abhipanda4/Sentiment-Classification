import os
import argparse
import numpy as np
import pickle
from copy import deepcopy

from data_loader import DataLoader
from representations import *;
from algorithms import *

if os.path.exists("stemmed_train.pkl") and False:
    with open("stemmed_train.pkl", 'rb') as f:
        train_corpus = pickle.load(f)
else:
    train_base_dir = "./data/aclImdb/train/"
    loader = DataLoader(train_base_dir)
    pos, neg = loader.gen_data()
    train_corpus = pos + neg
    with open('stemmed_train.pkl', 'wb') as f:
        pickle.dump(train_corpus, f)

if os.path.exists("stemmed_test.pkl") and False:
    with open("stemmed_test.pkl", 'rb') as f:
        test_corpus = pickle.load(f)
else:
    test_base_dir = "./data/aclImdb/test/"
    loader = DataLoader(test_base_dir)
    pos, neg = loader.gen_data()
    test_corpus = pos + neg
    with open('stemmed_test.pkl', 'wb') as f:
        pickle.dump(test_corpus, f)


n_samples = len(pos)
Y_train = np.array([0] * n_samples + [1] * n_samples)
Y_test = deepcopy(Y_train)

algo = "NB"
rep = "BBoW"
if algo == "NB" and rep == "BBoW":
    print(len(train_corpus))
    print(len(test_corpus))
    print(train_corpus[0])
    X = bag_of_words(train_corpus + test_corpus).toarray()
    X_train = X[ :2 * n_samples]
    X_test = X[2 * n_samples: ]
    test_accuracy = naive_bayes(X_train, Y_train, X_test, Y_test)
    print(test_accuracy)
