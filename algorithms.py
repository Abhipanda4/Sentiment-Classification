from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np

def naive_bayes(X_train, Y_train, X_test, Y_test):
    clf = GaussianNB()
    P = np.random.permutation(len(X_train))
    X_train = X_train[P]
    Y_train = Y_train[P]

    batch_size = 500
    num_batches = int(len(X_train) / batch_size)
    for i in range(num_batches):
        print("Training batch %3d" %(i + 1))
        minibatch_X = X_train[i * batch_size: (i + 1) * batch_size]
        minibatch_Y = Y_train[i * batch_size: (i + 1) * batch_size]
        clf.partial_fit(minibatch_X, minibatch_Y, [0, 1])

    accuracies = []
    num_batches = int(len(X_test) / batch_size)
    for i in range(num_batches):
        minibatch_X = X_test[i * batch_size: (i + 1) * batch_size]
        minibatch_Y = Y_test[i * batch_size: (i + 1) * batch_size]
        predictions = clf.predict(minibatch_X)
        accuracies.append(metrics.accuracy_score(minibatch_Y, predictions))

    return np.mean(accuracies)


def logistic_regression(X_train, Y_train, X_test, Y_test):
    clf = LogisticRegression()
    P = np.random.permutation(len(X_train))
    X_train = X_train[P]
    Y_train = Y_train[P]

    batch_size = 500
    num_batches = int(len(X_train) / batch_size)
    for i in range(num_batches):
        print("Training batch %3d" %(i + 1))
        minibatch_X = X_train[i * batch_size: (i + 1) * batch_size]
        minibatch_Y = Y_train[i * batch_size: (i + 1) * batch_size]
        clf.partial_fit(minibatch_X, minibatch_Y, [0, 1])

    accuracies = []
    num_batches = int(len(X_test) / batch_size)
    for i in range(num_batches):
        minibatch_X = X_test[i * batch_size: (i + 1) * batch_size]
        minibatch_Y = Y_test[i * batch_size: (i + 1) * batch_size]
        predictions = clf.predict(minibatch_X)
        accuracies.append(metrics.accuracy_score(minibatch_Y, predictions))

    return np.mean(accuracies)
