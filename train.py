import os
import argparse
import numpy as np
import pickle
from copy import deepcopy

from data_loader import DataLoader
from representations import *;
from algorithms import *

if os.path.exists("train.pkl"):
    with open("train.pkl", 'rb') as f:
        train_corpus = pickle.load(f)
else:
    train_base_dir = "./data/aclImdb/train/"
    loader = DataLoader(train_base_dir)
    pos, neg = loader.gen_data()
    train_corpus = pos + neg
    with open('train.pkl', 'wb') as f:
        pickle.dump(train_corpus, f)

if os.path.exists("test.pkl"):
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

def test_train_split(X):
    X_train = X[ :2 * n_samples]
    X_test = X[2 * n_samples: ]
    return X_train, X_test

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default="LR",
        choices=("NB", "LR", "MLP", "SVM", "LSTM"))
parser.add_argument("--rep", type=str, default="BBoW",
        choices=("BBoW", "NTF", "tfidf", "avg_W2V", "avg_GLoVE",
            "sen_vec", "doc_vec"))
parser.add_argument("--use_weights", type=bool, default=False)

args = parser.parse_args()
algo = args.algo
rep = args.rep
use_weights = args.use_weights

print("Representation: %s, algo: %s" %(rep, algo), end=" -- ")

total_corpus = train_corpus + test_corpus
if rep == "BBoW":
    X = bag_of_words(total_corpus)
elif rep == "NTF":
    X = normalized_tf(total_corpus)
elif rep == "tfidf":
    X = tfidf(total_corpus)
elif rep == "avg_W2V" and not use_weights:
    X = avg_word2vec(total_corpus)
elif rep == "avg_W2V" and use_weights:
    print("Using weights: ")
    X = avg_word2vec(total_corpus, True)
elif rep == "avg_GLoVE" and not use_weights:
    print("Using weights: ")
    X = avg_GLoVE(total_corpus)
elif rep == "avg_GLoVE" and use_weights:
    X = avg_GLoVE(total_corpus, True)
elif rep == "doc_vec":
    X = doc_vector(total_corpus)
elif rep == "sen_vec":
    X = sentence_vector(total_corpus)


X_train, X_test = test_train_split(X)

if algo == "NB":
    print(naive_bayes(X_train, Y_train, X_test, Y_test))
elif algo == "LR":
    print(logistic_regression(X_train, Y_train, X_test, Y_test))
elif algo == "SVM":
    print(SVM_classification(X_train, Y_train, X_test, Y_test))
elif algo == "MLP":
    print(feed_forward_NN(X_train, Y_train, X_test, Y_test))
elif algo == "LSTM":
    print(LSTM_classification(X_train, Y_train, X_test, Y_test))
