from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import sklearn
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM

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
    clf = svm.LinearSVC()
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

def LSTM_classify(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    s1 = len(X_train[0])
    s2 = len(X_train[0][0])
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(s1, s2)))
    model.add(Dense(1, activation='sigmoid'))

    # print model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=32, epochs=30,
    validation_data=(X_test, Y_test))
    scores = model.evaluate(X_test, Y_test)
    return scores
