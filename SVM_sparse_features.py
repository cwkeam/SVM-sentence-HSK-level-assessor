from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics


main_corpus = []
main_corpus_target = []


my_categories = ["1", "2", "3", "4", "5"]

# feeding corpus the testing data

print("Loading system call database for categories:")
print(my_categories if my_categories else "all")


with open('hsk/1.txt') as f:
    for line in f:
        main_corpus.append(line)
        main_corpus_target.append(1)

with open('hsk/2.txt') as f:
    for line in f:
        main_corpus.append(line)
        main_corpus_target.append(2)

with open('hsk/3.txt') as f:
    for line in f:
        main_corpus.append(line)
        main_corpus_target.append(3)

with open('hsk/4.txt') as f:
    for line in f:
        main_corpus.append(line)
        main_corpus_target.append(4)

with open('hsk/5.txt') as f:
    for line in f:
        main_corpus.append(line)
        main_corpus_target.append(5)


from sklearn.utils import shuffle
main_corpus_target, main_corpus = shuffle(main_corpus_target, main_corpus, random_state=0)


print("Data loaded.")
print(len(main_corpus_target))


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

all = False
if all:
    ratio = 25  # training to test set
    train_corpus = main_corpus[:(ratio*len(main_corpus)//(ratio+1))]
    train_corpus_target = main_corpus_target[:(ratio*len(main_corpus)//(ratio+1))]
    test_corpus = main_corpus[(len(main_corpus)-(len(main_corpus)//(ratio+1))):]
    test_corpus_target = main_corpus_target[(len(main_corpus)-len(main_corpus)//(ratio+1)):]
else:
    train_corpus = main_corpus
    train_corpus_target = main_corpus_target
    test_corpus = ["有一次，某个动物园里有一只大猩猩被铁笼子里的铁支架压着了，看样子， 压得真不轻，因为大猩猩的表情显得很痛苦。", "我真好"]
    test_corpus_target = [5, 1]

# size of datasets
train_corpus_size_mb = size_mb(train_corpus)
test_corpus_size_mb = size_mb(test_corpus)


print("%d documents - %0.3fMB (training set)" % (
    len(train_corpus_target), train_corpus_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(test_corpus_target), test_corpus_size_mb))
print("%d categories" % len(my_categories))
print()


print("Extracting features from the training data using a sparse vectorizer...")
t0 = time()


import jieba

def tokenize(text):
    tokens = jieba.cut(text, cut_all=False)
    return list(tokens)

vectorizer = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, max_df=0.5, tokenizer=tokenize, use_idf=True, smooth_idf=True)
analyze = vectorizer.build_analyzer()
print(analyze("我真好"))
X_train = vectorizer.fit_transform(train_corpus)



duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, train_corpus_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer...")
t0 = time()
X_test = vectorizer.transform(test_corpus)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, test_corpus_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()


def benchmark(clf):
    print('_'*60)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, train_corpus_target)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time: %0.3fs" % test_time)

    score = metrics.accuracy_score(test_corpus_target, pred)
    print("accuracy: %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        print()
    print(metrics.classification_report(test_corpus_target, pred,target_names=my_categories))
    print()
    clf_descr = str(clf).split('(')[0]

    print("Predicted hsk levels: ")
    print(pred.tolist());
    print()
    print("Real hsk levels:")
    print(test_corpus_target)
    print()
    errorArr = []
    for x in range(0, len(test_corpus_target)-1):
        a = test_corpus_target[x]
        b = pred.tolist()[x]
        errorArr.append((b-a))

    print("Margin of error: ")
    print(errorArr)

    return clf_descr, score, train_time, test_time

results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))



for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3))),
  ('classification', LinearSVC(penalty="l2"))])))


# plotting results

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
