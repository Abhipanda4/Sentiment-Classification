import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors

def bag_of_words(dataset):
    count_vec = CountVectorizer()
    doc_term_matrix = count_vec.fit_transform(dataset)
    doc_term_matrix = doc_term_matrix > 0
    return doc_term_matrix


def normalized_tf(dataset):
    count_vec = CountVectorizer()
    doc_term_matrix = count_vec.fit_transform(dataset)
    num_terms = sparse.diags(1/doc_term_matrix.sum(axis=1).A.ravel())
    doc_term_matrix = num_terms.dot(doc_term_matrix)
    return doc_term_matrix


def tfidf(dataset):
    count_vec = CountVectorizer()
    doc_term_matrix = count_vec.fit_transform(dataset)
    doc_term_tfidf = TfidfTransformer(use_idf=True).fit_transform(doc_term_matrix)
    return doc_term_tfidf


def avg_word2vec(dataset, use_weights=False):
    # model = Word2Vec.load_word2vec_format('./W2V/GoogleNews-vectors-negative300.bin', binary=True)
    # model = KeyedVectors.load_word2vec_format('./W2V/GoogleNews-vectors-negative300.bin', binary=True)
    # model.save_word2vec_format('./W2V/GoogleNews-vectors-negative300.txt', binary=False)
    with open("./W2V/GoogleNews-vectors-negative300.txt") as vocab:
        word2vec_rep = {line.split()[0] : np.array(list(map(float, line.split()[1:])))
            for line in vocab}
        dim = len(word2vec_rep.items())

        if not use_weights:
            doc_term_matrix = np.array([
                np.mean([word2vec_rep[w] for w in sentence if w in word2vec_rep]
                    or [np.zeros(dim)], axis=0)
                for sentence in dataset])
            return sparse.csr_matrix(doc_term_matrix)
        else:
            tfidf = TfidfVectorizer(lambda x: x).fit(dataset)
            max_idf = max(tfidf.idf_)
            word_weights = defaultdict(lambda: max_idf,
                    [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
            doc_term_matrix = np.array([
                np.mean([word2vec_rep[w] * word_weights[w] for w in sentence if w in word2vec_rep]
                    or [np.zeros(dim)], axis=0)
                for sentence in dataset])
            return sparse.csr_matrix(doc_term_matrix)


def avg_GLoVE(dataset, use_weights=False):
    with open("./W2V/glove.6B.300d.txt") as vocab:
        glove_rep = {line.split()[0] : np.array(list(map(float, line.split()[1:])))
            for line in vocab}

        dim = len(glove_rep.items())

        if not use_weights:
            doc_term_matrix = np.array([
                np.mean([glove_rep[w] for w in sentence if w in glove_rep]
                    or [np.zeros(dim)], axis=0)
                for sentence in dataset])
            return sparse.csr_matrix(doc_term_matrix)
        else:
            tfidf = TfidfVectorizer(lambda x: x).fit(dataset)
            max_idf = max(tfidf.idf_)
            word_weights = defaultdict(lambda: max_idf,
                    [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
            doc_term_matrix = np.array([
                np.mean([glove_rep[w] * word_weights[w] for w in sentence if w in glove_rep]
                    or [np.zeros(dim)], axis=0)
                for sentence in dataset])
            return sparse.csr_matrix(doc_term_matrix)


def avg_sentence_vec(dataset):
    pass

def par_vector(dataset):
    pass
