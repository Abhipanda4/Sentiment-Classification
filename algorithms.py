from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import sklearn
import numpy as np

def naive_bayes(X_train, Y_train, X_test, Y_test):
    clf = MultinomialNB()
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    return metrics.accuracy_score(Y_test, predictions)


def logistic_regression(X_train, Y_train, X_test, Y_test):
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    return metrics.accuracy_score(Y_test, predictions)


def SVM_classification(X_train, Y_train, X_test, Y_test):
    clf = svm.LinearSVC(
            # max_iter=400,
            verbose=1
    )
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    return metrics.accuracy_score(Y_test, predictions)


def feed_forward_NN(X_train, Y_train, X_test, Y_test):
    clf = MLPClassifier(
            hidden_layer_sizes=(200, 64),
            max_iter=10
    )
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    return metrics.accuracy_score(Y_test, predictions)
