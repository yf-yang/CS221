from submission import extractCharacterFeatures, learnPredictor, extractWordFeatures, readExamples, evaluatePredictor, dotProduct
import sys

maxN = 10
test_error = []

trainExamples = readExamples('polarity.train')
devExamples = readExamples('polarity.dev')

for n in range(maxN):
    sys.stdout.write('\r=== Learning Character Features #{} Predictor ==='.format(n+1))
    sys.stdout.flush()
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples, devExamples, featureExtractor, numIters=20, eta=0.01)
    devError = evaluatePredictor(devExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    test_error.append(devError)

print 'Test Error from #1 to #{}:'.format(maxN)
for e in test_error:
    print '{:.3f}'.format(e),
print

print '=== Learning Reference Word Feature Predictor ==='
featureExtractor = extractWordFeatures
weights = learnPredictor(trainExamples, devExamples, featureExtractor, numIters=20, eta=0.01)
devError = evaluatePredictor(devExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
print 'Reference Test Error: {:.3f}'.format(devError)

from matplotlib import pyplot as plt
plt.plot([i+1 for i in range(maxN)], test_error, 'b')
plt.plot([i+1 for i in range(maxN)], [devError] * maxN, 'r')
plt.show()