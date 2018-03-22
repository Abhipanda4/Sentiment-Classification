import numpy as np
import os
import pickle
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse

from gensim.models import Doc2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.doc2vec import LabeledSentence

from nltk.tokenize import sent_tokenize

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
        dim = 300

        if not use_weights:
            if os.path.exists("w2v_unweighted.pkl"):
                with open("w2v_unweighted.pkl", "rb") as f:
                    t = pickle.load(f)
                    return sparse.csr_matrix(t)
            else:
                doc_term_matrix = np.array([
                    np.mean([word2vec_rep[w] for w in sentence.split() if w in word2vec_rep]
                        or [np.zeros(dim)], axis=0)
                    for sentence in dataset])
                with open("w2v_unweighted.pkl", "wb") as f:
                    pickle.dump(doc_term_matrix, f)
                return sparse.csr_matrix(doc_term_matrix)
        else:
            if os.path.exists("w2v_weighted.pkl"):
                with open("w2v_weighted.pkl", "rb") as f:
                    t = pickle.load(f)
                    return sparse.csr_matrix(t)
            else:
                tfidf = TfidfVectorizer(lambda x: x).fit(dataset)
                max_idf = max(tfidf.idf_)
                word_weights = defaultdict(lambda: max_idf,
                        [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
                doc_term_matrix = np.array([
                    np.mean([word2vec_rep[w] * word_weights[w] for w in sentence.split() if w in word2vec_rep]
                        or [np.zeros(dim)], axis=0)
                    for sentence in dataset])
                with open("w2v_weighted.pkl", "wb") as f:
                    pickle.dump(doc_term_matrix, f)
                return sparse.csr_matrix(doc_term_matrix)


def avg_GLoVE(dataset, use_weights=False):
    with open("./W2V/glove.6B.300d.txt") as vocab:
        glove_rep = {line.split()[0] : np.array(list(map(float, line.split()[1:])))
            for line in vocab}

        # dim = len(glove_rep.items())
        dim = 300

        if not use_weights:
            if os.path.exists("glove_unweighted.pkl"):
                with open("glove_unweighted.pkl", "rb") as f:
                    t = pickle.load(f)
                    return sparse.csr_matrix(t)
            else:
                doc_term_matrix = np.array([
                    np.mean([glove_rep[w] for w in sentence.split() if w in glove_rep]
                        or [np.zeros(dim)], axis=0)
                    for sentence in dataset])
                with open("glove_unweighted.pkl", "wb") as f:
                    pickle.dump(doc_term_matrix, f)
                return sparse.csr_matrix(doc_term_matrix)
        else:
            if os.path.exists("glove_weighted.pkl"):
                with open("glove_weighted.pkl", "rb") as f:
                    t = pickle.load(f)
                    return sparse.csr_matrix(t)
            else:
                tfidf = TfidfVectorizer(lambda x: x).fit(dataset)
                max_idf = max(tfidf.idf_)
                word_weights = defaultdict(lambda: max_idf,
                        [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
                doc_term_matrix = np.array([
                    np.mean([glove_rep[w] * word_weights[w] for w in sentence.split() if w in glove_rep]
                        or [np.zeros(dim)], axis=0)
                    for sentence in dataset])
                with open("glove_weighted.pkl", "wb") as f:
                    pickle.dump(doc_term_matrix, f)
                return sparse.csr_matrix(doc_term_matrix)


## Document vector code
## code adopted from https://medium.com/@klintcho/doc2vec-tutorial-using-gensim-ab3ac03d3a1
class LabeledLineSentence:
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(words=doc.split(), tags=[self.labels_list[idx]])

def doc_vector(dataset):
    saved_doc_model = "./W2V/doc_vectors.model"
    if os.path.exists(saved_doc_model):
        model = Doc2Vec.load(saved_doc_model)
    else:
        model = Doc2Vec(vector_size=100, window=10, workers=11,alpha=0.025, min_alpha=0.025, dbow_words=0)
        labels_list = list(range(len(dataset)))
        it = LabeledLineSentence(dataset, labels_list)
        model.build_vocab(it)
        model.train(it, total_examples=len(dataset), epochs=10, end_alpha = 0.01)
        model.save(saved_doc_model)

    vectors = []
    for x in range(len(dataset)):
        vectors.append(model.docvecs[x])
    return vectors

def sentence_vector(dataset):
    saved_doc_model = "./W2V/sentence_vectors.model"
    if os.path.exists(saved_doc_model):
        model = Doc2Vec.load(saved_doc_model)
    else:
        model = Doc2Vec(vector_size=100, window=10, workers=11,alpha=0.025, min_alpha=0.025, dbow_words=0)
        # build labels and sentences
        sentences = []
        labels_list = []
        for i, review in enumerate(dataset):
            sentence = sent_tokenize(review)
            labels_list.extend([str(i) + "_" + str(j) for j in range(len(sentence))])
            sentences.extend(sentence)

        it = LabeledLineSentence(sentences, labels_list)
        model.build_vocab(it)
        model.train(it, total_examples=len(dataset), epochs=10, end_alpha = 0.01)
        model.save(saved_doc_model)

    vectors = []
    for i, r in enumerate(dataset):
        review = [model.docvecs[str(i) + "_" + str(j)] for j in range(len(sent_tokenize(r)))]
        # print(np.shape(np.asarray(review)))
        vectors.append(np.mean(review, axis=0))

    return vectors
