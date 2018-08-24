#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    # Note that collections.Counter is a subclass of dict
    return collections.Counter(x.split())
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    def init_predictor(w):
        '''
        return a predictor function with weights w
        '''
        return lambda x: 1 if dotProduct(featureExtractor(x), w) >= 0 else -1

    # maybe here we should call numIters "numEpochs"
    # I've spent so long debugging and finally found out I misunderstood the word here
    for iter in range(numIters):
        random.shuffle(trainExamples)
        for x, y in trainExamples:
            featX = featureExtractor(x)
            if 1 - y * dotProduct(weights, featX) > 0:
                increment(weights, eta * y, featX)

        # debug
        # print 'Iter {}/{} Train Error Rate: {}'.format(iter+1, numIters, evaluatePredictor(trainExamples, init_predictor(weights)))
        # print 'Iter {}/{} Test Error Rate: {}'.format(iter+1, numIters, evaluatePredictor(testExamples, init_predictor(weights)))

    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = {k: random.uniform(-1, 1) for k in random.sample(weights, random.randint(1, len(weights)))}
        y =  1 if dotProduct(phi, weights) >= 0 else -1
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        shrinkX = ''.join(x.split())
        counter = collections.Counter([shrinkX[start: start+n] for start in range(len(shrinkX)-n+1)])
        return dict(counter)
        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    def euclideanDistance(v1, v2, v1_square, v2_square):
        return v1_square + v2_square - 2 * dotProduct(v1, v2)

    def assign(v, centroids, e_square, c_squares):
        '''
        assign each vector to a centroid with minimum reconstruction loss
        '''
        min_dist = 1000000
        for c in range(len(centroids)):
            dist = euclideanDistance(v, centroids[c], e_square, c_squares[c])
            if dist < min_dist:
                min_dist = dist
                min_c = c
        return min_c, min_dist

    def avgV(vs):
        size = float(len(vs))
        result = collections.defaultdict(float)
        for vec in vs:
            for k, v in vec.items():
                result[k] += v / size
        return result

    centroids = random.sample(examples, K)
    ele_squares = [dotProduct(ele, ele) for ele in examples]
    example_centroid_map = [0 for _ in examples]
    it = 0
    while True:
        loss = 0.0
        it += 1

        c_squares = [dotProduct(c, c) for c in centroids]

        # assign each ele to entroids and get reconstruction loss
        centroid_example_map = {x: [] for x in range(len(centroids))}
        new_example_centroid_map = []

        for i, (ele, e_square) in enumerate(zip(examples, ele_squares)):
            c, l = assign(ele, centroids, e_square, c_squares)
            centroid_example_map[c].append(ele)
            new_example_centroid_map.append(c)
            loss += l

        # import pdb;pdb.set_trace()
        if example_centroid_map == new_example_centroid_map or it >= maxIters:
            return centroids, example_centroid_map, loss

        example_centroid_map = new_example_centroid_map
        # compute new centroids
        centroids = [avgV(eles) for eles in centroid_example_map.values()]
    # END_YOUR_CODE
