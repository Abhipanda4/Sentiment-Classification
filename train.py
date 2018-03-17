import os
import argparse
import numpy as np
import pickle
from copy import deepcopy

from data_loader import DataLoader
from representations import *;
from algorithms import *

if os.path.exists("train.pkl") and False:
    with open("train.pkl", 'rb') as f:
        train_corpus = pickle.load(f)
else:
    train_base_dir = "./data/aclImdb/train/"
    loader = DataLoader(train_base_dir)
    pos, neg = loader.gen_data()
    train_corpus = pos + neg
    with open('train.pkl', 'wb') as f:
        pickle.dump(train_corpus, f)

if os.path.exists("test.pkl") and False:
    with open("test.pkl", 'rb') as f:
        test_corpus = pickle.load(f)
else:
    test_base_dir = "./data/aclImdb/test/"
    loader = DataLoader(test_base_dir)
    pos, neg = loader.gen_data()
    test_corpus = pos + neg
    with open('test.pkl', 'wb') as f:
        pickle.dump(test_corpus, f)


n_samples = len(train_corpus) // 2
Y_train = np.array([0] * n_samples + [1] * n_samples)
Y_test = deepcopy(Y_train)

algo = "LR"
rep = "BBoW"

def test_train_split(X):
    X_train = X[ :2 * n_samples]
    X_test = X[2 * n_samples: ]
    return X_train, X_test

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default="LR",
        choices=("NB", "LR", "MLP", "SVM", "LSTM"))
parser.add_argument("--rep", type=str, default="tfidf",
        choices=("BBoW", "NTF", "tfidf", "avg_W2V", "avg_GLoVE",
            "sentence_vector", "par_vector"))

args = parser.parse_args()
algo = args.algo
rep = args.rep

# Naive Bayes Classifier
if algo == "NB" and rep == "BBoW":
    X = bag_of_words(train_corpus + test_corpus)
    X_train, X_test = test_train_split(X)
    print(naive_bayes(X_train, Y_train, X_test, Y_test))

elif algo == "NB" and rep == "tfidf":
    X = tfidf(train_corpus + test_corpus)
    X_train, X_test = test_train_split(X)
    print(naive_bayes(X_train, Y_train, X_test, Y_test))

elif algo == "NB" and rep == "NTF":
    X = normalized_tf(train_corpus + test_corpus)
    X_train, X_test = test_train_split(X)
    print(naive_bayes(X_train, Y_train, X_test, Y_test))

# Logistic Regression Classifier
elif algo == "LR" and rep == "BBoW":
    X = bag_of_words(train_corpus + test_corpus)
    X_train, X_test = test_train_split(X)
    print(logistic_regression(X_train, Y_train, X_test, Y_test))

elif algo == "LR" and rep == "tfidf":
    X = tfidf(train_corpus + test_corpus)
    X_train, X_test = test_train_split(X)
    print(logistic_regression(X_train, Y_train, X_test, Y_test))

# SVM Classifier
elif algo == "SVM" and rep == "BBoW":
    X = bag_of_words(train_corpus + test_corpus)
    X_train, X_test = test_train_split(X)
    print(SVM_classification(X_train, Y_train, X_test, Y_test))

elif algo == "SVM" and rep == "tfidf":
    X = tfidf(train_corpus + test_corpus)
    X_train, X_test = test_train_split(X)
    print(SVM_classification(X_train, Y_train, X_test, Y_test))

# Feed Forward Neural Network CLassifier
elif algo == "MLP" and rep == "BBoW":
    X = bag_of_words(train_corpus + test_corpus)
    X_train, X_test = test_train_split(X)
    print(feed_forward_NN(X_train, Y_train, X_test, Y_test))

elif algo == "MLP" and rep == "tfidf":
    X = tfidf(train_corpus + test_corpus)
    X_train, X_test = test_train_split(X)
    print(feed_forward_NN(X_train, Y_train, X_test, Y_test))
