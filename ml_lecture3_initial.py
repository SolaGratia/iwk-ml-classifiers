
from __future__ import print_function

# BaseEstimator and ClassifierMixin are base classes for classifiers in
# scikit-learn.
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy

class LinearClassifier(BaseEstimator, ClassifierMixin):
    """Base class for linear classifiers, i.e. classifiers with a linear
    prediction function."""

    def find_classes(self, Y):
        """Find the set of output classes in a training set.

        The attributes positive_class and negative_class will be assigned
        after calling this method. An exception is raised if the number of
        classes in the training set is not equal to 2."""

        classes = list(set(Y))
        if len(classes) != 2:
            raise Exception("this does not seem to be a 2-class problem")
        self.positive_class = classes[0]
        self.negative_class = classes[1]

    def predict(self, X):        
        """Apply the linear scoring function and outputs self.positive_class
        for instances where the scoring function is positive, otherwise
        self.negative_class."""
        
        # To speed up, we apply the scoring function to all the instances
        # at the same time.
        scores = X.dot(self.w)
        
        # Create the output array.
        # At the positions where the score is positive, this will contain
        # self.positive class, otherwise self.negative_class.
        out = numpy.select([scores>=0.0, scores<0.0], [self.positive_class, 
                                                       self.negative_class])
        return out


class DensePerceptron(LinearClassifier):
    """A perceptron implementation, see slide 20 in the lecture."""

    def __init__(self, n_iter=10):
        self.n_iter = n_iter

    def fit(self, X, Y):
        Y = list(Y)
        self.find_classes(Y)

        # The feature vectors returned by DictVectorizer/CountVectorizer
        # are sparse vectors of the type scipy.sparse.csc_matrix. We convert
        # them to dense vectors of the type numpy.array.
        X = X.toarray()

        # The shape attribute holds the dimensions of the feature matrix. 
        # This is a tuple where
        # X.shape[0] = number of instances, 
        # X.shape[1] = number of features
        n_features = X.shape[1]

        # Initialize the weight vector to all zero
        self.w = numpy.zeros( n_features )
        
        for i in range(self.n_iter):            

            for x, y in zip(X, Y):
                
                # Compute the linear scoring function
                score = self.w.dot(x)                

                # If a positive instance was misclassified...
                if score < 0 and y == self.positive_class:
                    # then add its features to the weight vector
                    self.w += x

                # on the other hand if a negative instance was misclassified...
                elif score >= 0 and y == self.negative_class:
                    # then subtract its features from the weight vector
                    self.w -= x

def sign(y, pos):
    if y == pos:
        return 1.0
    else:
        return -1.0

class DensePerceptron2(LinearClassifier):
    """Reformulation of the perceptron where we encode the output classes as
    +1 and -1, see slide 21 in the lecture."""

    def __init__(self, n_iter=10):
        self.n_iter = n_iter

    def fit(self, X, Y):
        Y = list(Y)
        self.find_classes(Y)

        # convert all output values to +1 or -1
        Yn = [sign(y, self.positive_class) for y in Y]

        X = X.toarray()
        n_features = X.shape[1]
        self.w = numpy.zeros( n_features )
        
        for i in range(self.n_iter):            
            for x, y in zip(X, Yn):
                score = self.w.dot(x) * y
                if score <= 0:
                    self.w += y*x

# Two helper functions for processing sparse and dense vectors.
# I haven't been able to do this efficiently in a more "civilised" manner: 
# these functions rely on the internal details of scipy.sparse.csr_matrix.
def add_sparse_to_dense(x, w, xw):
    w[x.indices] += xw*x.data
def sparse_dense_dot(x, w):
    return numpy.dot(w[x.indices], x.data)

class SparsePerceptron(LinearClassifier):
    """A perceptron implementation using sparse feature vectors, see 
    slide 23 in the lecture."""

    def __init__(self, n_iter=10):
        self.n_iter = n_iter

    def fit(self, X, Y):
        Y = list(Y)
        self.find_classes(Y)

        Yn = [sign(y, self.positive_class) for y in Y]
        n_features = X.shape[1]
        self.w = numpy.zeros( n_features )
        X = list(X) 
        for i in range(self.n_iter):            
            for x, y in zip(X, Yn):
                score = sparse_dense_dot(x, self.w) * y
                if score <= 0:
                    add_sparse_to_dense(x, self.w, y)
        

class AveragedSparsePerceptron(LinearClassifier):
    """Averaged perceptron, see slide 11 in the bonus lecture."""

    def __init__(self, n_iter=10):
        self.n_iter = n_iter

    def fit(self, X, Y):
        Y = list(Y)
        self.find_classes(Y)

        Yn = [sign(y, self.positive_class) for y in Y]
        n_features = X.shape[1]
        X = list(X) 

        w = numpy.zeros( n_features )
        a = numpy.zeros( n_features )

        NT = self.n_iter * len(Y)
        step = NT

        for i in range(self.n_iter):            
            for x, y in zip(X, Yn):
                score = sparse_dense_dot(x, w) * y
                if score <= 0:
                    add_sparse_to_dense(x, w, y)
                    add_sparse_to_dense(x, a, step * y / NT)
                step -= 1

        self.w = a
