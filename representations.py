import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def bag_of_words(dataset):
    count_vec = CountVectorizer()
    doc_term_matrix = count_vec.fit_transform(dataset)
    doc_term_matrix = doc_term_matrix > 0

    return doc_term_matrix


def normalized_tf(dataset):
    count_vec = CountVectorizer()
    doc_term_matrix = count_vec.fit_transform(dataset)
    num_terms = np.sum(doc_term_matrix, axis=0, keepdims=True)
    doc_term_matrix /= num_terms

    return doc_term_matrix


def tfidf(dataset):
    count_vec = CountVectorizer()
    doc_term_matrix = count_vec.fit_transform(dataset)
    doc_term_tfidf = TfidfTransformer(use_idf=True).fit_transform(doc_term_matrix)

    return doc_term_tfidf


def avg_word2vec(dataset, use_weights=False):
    pass

def avg_GLoVE(dataset, use_weights=False):
    pass

def avg_sentence_vec(dataset):
    pass

def par_vector(dataset):
    pass
