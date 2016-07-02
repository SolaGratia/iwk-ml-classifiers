'''Machine Learning, assignment 2
Isac Waern Kyrck'''


# for Python 2 compatibility
from __future__ import print_function
from io import open

import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

import ml_lecture3

def read_corpus(corpus_file):
    X, Y = [], []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            Y.append(tokens[1])
            X.append(tokens[3:])
    return X, Y


def assignment2_experiment():
    X_all, Y_all = read_corpus("all_sentiment_shuffled.txt")
    X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all,
                                                        train_size=0.8,
                                                        random_state=0)

    vec = CountVectorizer(preprocessor=lambda x: x,
                          tokenizer=lambda x: x,
                          binary=True)

    # using one of the perceptron implementations from the lecture
    #cls = ml_lecture3.SparsePerceptron()
    #cls = ml_lecture3.DensePerceptron()
    #cls = ml_lecture3.DensePerceptron2()
    #cls = ml_lecture3.AveragedSparsePerceptron()
    #cls = ml_lecture3.Pegasos(n_steps=10*len(X_train))
    #cls = ml_lecture3.SparsePegasos(n_steps=10*len(X_train))
    #cls = ml_lecture3.SparsePegasosWithScalingTrick(n_steps=10*len(X_train))
    cls = ml_lecture3.PegasosLogisticRegression(n_steps=10*len(X_train))
    

    # using classifiers from scikit-learn
    #cls = Perceptron()
    #cls = LinearSVC()
    #cls = LogisticRegression()

    # if you have memory problems when using the dense perceptrons,
    # change from 'all' to a number, e.g. 1000
    number_of_features = 'all'

    pipeline = Pipeline([('vec', vec),
                         ('prune', SelectKBest(k=number_of_features)),
                         ('cls', cls)])

    t1 = time.time()
    pipeline.fit(X_train, Y_train)
    t2 = time.time()

    print('Training time: {0:.3f} sec.'.format(t2-t1))

    t3 = time.time()
    Y_guesses = pipeline.predict(X_test)
    t4 = time.time()

    print('Classification time: {0:.3f} sec.'.format(t4-t3))

    acc = accuracy_score(Y_test, Y_guesses)
    print('Accuracy on the test set: {0:.3f}'.format(acc))

if __name__ == '__main__':
    assignment2_experiment()

