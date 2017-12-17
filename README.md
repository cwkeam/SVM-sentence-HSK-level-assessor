# SVM-sentence-HSK-level-assessor
Machine learning program using sparse features to determine the HSK level of a given sentence.

Sample Log:

Loading system call database for categories:
['1', '2', '3', '4', '5']
Data loaded.
1252
1203 documents - 0.039MB (training set)
48 documents - 0.002MB (test set)
5 categories

Extracting features from the training data using a sparse vectorizer...
Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/5y/9y0n51zs30s0xbfnq4tk441m0000gn/T/jieba.cache
['你', '可以', '坐船去', '上海']
Loading model cost 1.039 seconds.
Prefix dict has been built succesfully.
done in 1.314996s at 0.030MB/s
n_samples: 1203, n_features: 1854

Extracting features from the test data using the same vectorizer...
/Users/chanwookim/Library/Python/3.6/lib/python/site-packages/sklearn/linear_model/ridge.py:311: UserWarning: In Ridge, only 'sag' solver can currently fit the intercept when X is sparse. Solver has been automatically changed into 'sag'.
  warnings.warn("In Ridge, only 'sag' solver can currently fit the "
done in 0.006041s at 0.286MB/s
n_samples: 48, n_features: 1854

================================================================================
Ridge Classifier
____________________________________________________________
Training: 
RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
        max_iter=None, normalize=False, random_state=None, solver='lsqr',
        tol=0.01)
train time: 0.019s
test time: 0.000s
accuracy: 0.625
dimensionality: 1854
density: 1.000000

             precision    recall  f1-score   support

          1       0.88      0.88      0.88         8
          2       0.21      0.60      0.32         5
          3       0.82      0.45      0.58        20
          4       0.60      0.67      0.63         9
          5       1.00      0.83      0.91         6

avg / total       0.75      0.62      0.65        48


Predicted hsk levels: 
[5, 1, 3, 1, 5, 4, 4, 2, 1, 2, 3, 3, 3, 2, 3, 2, 2, 5, 3, 2, 5, 5, 1, 4, 1, 2, 4, 1, 3, 1, 4, 4, 3, 1, 2, 2, 4, 2, 4, 4, 2, 3, 2, 2, 4, 2, 3, 3]

Real hsk levels:
[5, 1, 3, 1, 5, 4, 3, 3, 1, 2, 3, 3, 3, 3, 3, 2, 4, 5, 3, 3, 5, 5, 1, 4, 2, 3, 4, 1, 3, 1, 4, 3, 4, 1, 1, 3, 3, 2, 3, 4, 3, 2, 4, 3, 4, 5, 3, 3]

Margin of error: 
[0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, -1, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 1, -1, 0, 1, -1, 1, 0, 1, 0, -1, 1, -2, -1, 0, -3, 0]

================================================================================
Perceptron
____________________________________________________________
Training: 
Perceptron(alpha=0.0001, class_weight=None, eta0=1.0, fit_intercept=True,
      n_iter=50, n_jobs=1, penalty=None, random_state=0, shuffle=True,
      verbose=0, warm_start=False)
train time: 0.016s
test time: 0.000s
accuracy: 0.500
dimensionality: 1854
density: 0.568932

             precision    recall  f1-score   support

          1       0.88      0.88      0.88         8
          2       0.22      0.80      0.35         5
          3       0.67      0.30      0.41        20
          4       0.33      0.22      0.27         9
          5       0.71      0.83      0.77         6

avg / total       0.60      0.50      0.50        48


Predicted hsk levels: 
[5, 1, 2, 1, 5, 4, 2, 5, 1, 2, 3, 3, 2, 2, 4, 2, 2, 5, 3, 2, 5, 5, 1, 2, 1, 2, 3, 1, 2, 1, 3, 4, 3, 1, 4, 2, 4, 2, 3, 5, 2, 2, 2, 2, 4, 2, 3, 3]

Real hsk levels:
[5, 1, 3, 1, 5, 4, 3, 3, 1, 2, 3, 3, 3, 3, 3, 2, 4, 5, 3, 3, 5, 5, 1, 4, 2, 3, 4, 1, 3, 1, 4, 3, 4, 1, 1, 3, 3, 2, 3, 4, 3, 2, 4, 3, 4, 5, 3, 3]

Margin of error: 
[0, 0, -1, 0, 0, 0, -1, 2, 0, 0, 0, 0, -1, -1, 1, 0, -2, 0, 0, -1, 0, 0, 0, -2, -1, -1, -1, 0, -1, 0, -1, 1, -1, 0, 3, -1, 1, 0, 0, 1, -1, 0, -2, -1, 0, -3, 0]

================================================================================
L2 penalty
____________________________________________________________
Training: 
LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.001,
     verbose=0)
train time: 0.010s
test time: 0.000s
accuracy: 0.667
dimensionality: 1854
density: 1.000000

             precision    recall  f1-score   support

          1       0.88      0.88      0.88         8
          2       0.29      0.80      0.42         5
          3       0.91      0.50      0.65        20
          4       0.67      0.67      0.67         9
          5       0.83      0.83      0.83         6

avg / total       0.78      0.67      0.69        48


Predicted hsk levels: 
[5, 1, 3, 1, 5, 4, 4, 2, 1, 2, 3, 3, 3, 2, 3, 2, 2, 5, 3, 2, 5, 5, 1, 4, 1, 2, 4, 1, 3, 1, 4, 4, 3, 1, 2, 2, 4, 2, 3, 4, 2, 2, 2, 5, 4, 2, 3, 3]

Real hsk levels:
[5, 1, 3, 1, 5, 4, 3, 3, 1, 2, 3, 3, 3, 3, 3, 2, 4, 5, 3, 3, 5, 5, 1, 4, 2, 3, 4, 1, 3, 1, 4, 3, 4, 1, 1, 3, 3, 2, 3, 4, 3, 2, 4, 3, 4, 5, 3, 3]

Margin of error: 
[0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, -1, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 1, -1, 0, 1, -1, 1, 0, 0, 0, -1, 0, -2, 2, 0, -3, 0]

____________________________________________________________
Training: 
SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', n_iter=50, n_jobs=1,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False)
train time: 0.021s
test time: 0.001s
accuracy: 0.667
dimensionality: 1854
density: 0.753722

             precision    recall  f1-score   support

          1       1.00      1.00      1.00         8
          2       0.36      1.00      0.53         5
          3       0.82      0.45      0.58        20
          4       0.56      0.56      0.56         9
          5       0.83      0.83      0.83         6

avg / total       0.75      0.67      0.67        48


Predicted hsk levels: 
[5, 1, 3, 1, 5, 4, 4, 2, 1, 2, 3, 3, 2, 2, 3, 2, 2, 5, 3, 2, 5, 5, 1, 4, 2, 2, 4, 1, 3, 1, 3, 4, 3, 1, 1, 2, 4, 2, 3, 4, 2, 2, 2, 5, 4, 4, 3, 3]

Real hsk levels:
[5, 1, 3, 1, 5, 4, 3, 3, 1, 2, 3, 3, 3, 3, 3, 2, 4, 5, 3, 3, 5, 5, 1, 4, 2, 3, 4, 1, 3, 1, 4, 3, 4, 1, 1, 3, 3, 2, 3, 4, 3, 2, 4, 3, 4, 5, 3, 3]

Margin of error: 
[0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, -1, -1, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 1, -1, 0, 0, -1, 1, 0, 0, 0, -1, 0, -2, 2, 0, -1, 0]
================================================================================
L1 penalty
____________________________________________________________
Training: 
LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l1', random_state=None, tol=0.001,
     verbose=0)
train time: 0.024s
test time: 0.000s
accuracy: 0.521
dimensionality: 1854
density: 0.179396

             precision    recall  f1-score   support

          1       0.88      0.88      0.88         8
          2       0.25      0.60      0.35         5
          3       0.71      0.25      0.37        20
          4       0.33      0.56      0.42         9
          5       0.83      0.83      0.83         6

avg / total       0.64      0.52      0.52        48


Predicted hsk levels: 
[5, 1, 2, 1, 5, 4, 4, 5, 1, 2, 4, 3, 4, 4, 3, 4, 2, 5, 3, 2, 4, 5, 1, 4, 2, 2, 4, 1, 3, 1, 4, 4, 3, 1, 2, 2, 4, 2, 4, 1, 2, 3, 2, 2, 4, 5, 4, 3]

Real hsk levels:
[5, 1, 3, 1, 5, 4, 3, 3, 1, 2, 3, 3, 3, 3, 3, 2, 4, 5, 3, 3, 5, 5, 1, 4, 2, 3, 4, 1, 3, 1, 4, 3, 4, 1, 1, 3, 3, 2, 3, 4, 3, 2, 4, 3, 4, 5, 3, 3]

Margin of error: 
[0, 0, -1, 0, 0, 0, 1, 2, 0, 0, 1, 0, 1, 1, 0, 2, -2, 0, 0, -1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, -1, 0, 1, -1, 1, 0, 1, -3, -1, 1, -2, -1, 0, 0, 1]
____________________________________________________________
Training: 
SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', n_iter=50, n_jobs=1,
       penalty='l1', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False)
train time: 0.042s
test time: 0.000s
accuracy: 0.521
dimensionality: 1854
density: 0.359115

             precision    recall  f1-score   support

          1       0.70      0.88      0.78         8
          2       0.29      0.80      0.42         5
          3       0.67      0.30      0.41        20
          4       0.38      0.33      0.35         9
          5       0.71      0.83      0.77         6

avg / total       0.58      0.52      0.51        48


Predicted hsk levels: 
[5, 1, 2, 1, 5, 3, 4, 5, 1, 2, 1, 3, 2, 2, 4, 2, 2, 5, 3, 2, 5, 5, 1, 4, 1, 2, 4, 1, 3, 1, 3, 4, 3, 1, 2, 2, 4, 2, 3, 1, 2, 2, 2, 5, 4, 4, 3, 3]

Real hsk levels:
[5, 1, 3, 1, 5, 4, 3, 3, 1, 2, 3, 3, 3, 3, 3, 2, 4, 5, 3, 3, 5, 5, 1, 4, 2, 3, 4, 1, 3, 1, 4, 3, 4, 1, 1, 3, 3, 2, 3, 4, 3, 2, 4, 3, 4, 5, 3, 3]

Margin of error: 
[0, 0, -1, 0, 0, -1, 1, 2, 0, 0, -2, 0, -1, -1, 1, 0, -2, 0, 0, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, 1, -1, 0, 1, -1, 1, 0, 0, -3, -1, 0, -2, 2, 0, -1, 0]

================================================================================
Elastic-Net penalty
____________________________________________________________
Training: 
SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', n_iter=50, n_jobs=1,
       penalty='elasticnet', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False)
train time: 0.037s
test time: 0.000s
accuracy: 0.646
dimensionality: 1854
density: 0.641424

             precision    recall  f1-score   support

          1       1.00      1.00      1.00         8
          2       0.33      1.00      0.50         5
          3       0.75      0.45      0.56        20
          4       0.57      0.44      0.50         9
          5       0.83      0.83      0.83         6

avg / total       0.73      0.65      0.65        48


Predicted hsk levels: 
[5, 1, 3, 1, 5, 3, 2, 2, 1, 2, 3, 3, 2, 2, 3, 2, 2, 5, 3, 2, 5, 5, 1, 4, 2, 2, 4, 1, 3, 1, 3, 4, 3, 1, 1, 2, 4, 2, 3, 4, 2, 2, 2, 5, 4, 4, 3, 3]

Real hsk levels:
[5, 1, 3, 1, 5, 4, 3, 3, 1, 2, 3, 3, 3, 3, 3, 2, 4, 5, 3, 3, 5, 5, 1, 4, 2, 3, 4, 1, 3, 1, 4, 3, 4, 1, 1, 3, 3, 2, 3, 4, 3, 2, 4, 3, 4, 5, 3, 3]

Margin of error: 
[0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 1, -1, 0, 0, -1, 1, 0, 0, 0, -1, 0, -2, 2, 0, -1, 0]

================================================================================
NearestCentroid (aka Rocchio classifier)
____________________________________________________________
Training: 
NearestCentroid(metric='euclidean', shrink_threshold=None)
train time: 0.004s
test time: 0.000s
accuracy: 0.667
             precision    recall  f1-score   support

          1       0.64      0.88      0.74         8
          2       0.40      0.80      0.53         5
          3       0.91      0.50      0.65        20
          4       0.60      0.67      0.63         9
          5       0.83      0.83      0.83         6

avg / total       0.74      0.67      0.67        48


Predicted hsk levels: 
[5, 1, 4, 1, 5, 4, 3, 5, 1, 2, 1, 3, 3, 4, 3, 2, 2, 5, 3, 1, 4, 5, 1, 4, 2, 3, 4, 1, 3, 1, 4, 3, 4, 1, 2, 2, 4, 2, 3, 1, 2, 1, 3, 2, 4, 5, 3, 2]

Real hsk levels:
[5, 1, 3, 1, 5, 4, 3, 3, 1, 2, 3, 3, 3, 3, 3, 2, 4, 5, 3, 3, 5, 5, 1, 4, 2, 3, 4, 1, 3, 1, 4, 3, 4, 1, 1, 3, 3, 2, 3, 4, 3, 2, 4, 3, 4, 5, 3, 3]

Margin of error: 
[0, 0, 1, 0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 1, 0, 0, -2, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0, -3, -1, -1, -1, -1, 0, 0, 0]

================================================================================
Naive Bayes
____________________________________________________________
Training: 
MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)
train time: 0.001s
test time: 0.000s
accuracy: 0.604
dimensionality: 1854
density: 1.000000

             precision    recall  f1-score   support

          1       0.88      0.88      0.88         8
          2       0.25      0.60      0.35         5
          3       0.83      0.50      0.62        20
          4       0.50      0.44      0.47         9
          5       0.62      0.83      0.71         6

avg / total       0.69      0.60      0.62        48


Predicted hsk levels: 
[5, 1, 3, 1, 5, 3, 3, 4, 1, 2, 3, 3, 3, 2, 3, 2, 2, 5, 3, 2, 5, 5, 1, 5, 1, 2, 4, 1, 3, 1, 4, 4, 4, 1, 2, 2, 4, 2, 4, 5, 2, 3, 5, 2, 4, 2, 3, 3]

Real hsk levels:
[5, 1, 3, 1, 5, 4, 3, 3, 1, 2, 3, 3, 3, 3, 3, 2, 4, 5, 3, 3, 5, 5, 1, 4, 2, 3, 4, 1, 3, 1, 4, 3, 4, 1, 1, 3, 3, 2, 3, 4, 3, 2, 4, 3, 4, 5, 3, 3]

Margin of error: 
[0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -2, 0, 0, -1, 0, 0, 0, 1, -1, -1, 0, 0, 0, 0, 0, 1, 0, 0, 1, -1, 1, 0, 1, 1, -1, 1, 1, -1, 0, -3, 0]
____________________________________________________________
Training: 
BernoulliNB(alpha=0.01, binarize=0.0, class_prior=None, fit_prior=True)
train time: 0.001s
test time: 0.000s
accuracy: 0.667
dimensionality: 1854
density: 1.000000

             precision    recall  f1-score   support

          1       0.70      0.88      0.78         8
          2       0.25      0.60      0.35         5
          3       0.91      0.50      0.65        20
          4       0.70      0.78      0.74         9
          5       1.00      0.83      0.91         6

avg / total       0.78      0.67      0.69        48


Predicted hsk levels: 
[5, 1, 3, 1, 5, 4, 2, 4, 1, 2, 3, 3, 3, 2, 3, 2, 1, 5, 3, 2, 5, 1, 1, 4, 1, 2, 4, 1, 3, 1, 4, 3, 4, 1, 2, 2, 4, 2, 4, 4, 2, 3, 2, 2, 4, 5, 3, 3]

Real hsk levels:
[5, 1, 3, 1, 5, 4, 3, 3, 1, 2, 3, 3, 3, 3, 3, 2, 4, 5, 3, 3, 5, 5, 1, 4, 2, 3, 4, 1, 3, 1, 4, 3, 4, 1, 1, 3, 3, 2, 3, 4, 3, 2, 4, 3, 4, 5, 3, 3]

Margin of error: 
[0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, -1, 0, 0, -3, 0, 0, -1, 0, -4, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, 0, 1, 0, -1, 1, -2, -1, 0, 0, 0]

================================================================================
LinearSVC with L1-based feature selection
____________________________________________________________
Training: 
Pipeline(steps=[('feature_selection', SelectFromModel(estimator=LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l1', random_state=None, tol=0.001,
     verbose=0),
        prefit=False, thresho...ax_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0))])
train time: 0.030s
test time: 0.001s
accuracy: 0.583
             precision    recall  f1-score   support

          1       0.89      1.00      0.94         8
          2       0.25      0.60      0.35         5
          3       0.78      0.35      0.48        20
          4       0.50      0.67      0.57         9
          5       0.67      0.67      0.67         6
avg / total       0.68      0.58      0.59        48


Predicted hsk levels: 
[5, 1, 2, 1, 5, 4, 4, 2, 1, 2, 4, 3, 3, 2, 3, 2, 2, 5, 3, 2, 4, 5, 1, 4, 1, 2, 4, 1, 3, 1, 4, 4, 3, 1, 1, 2, 4, 2, 4, 4, 2, 3, 5, 5, 4, 2, 3, 3]

Real hsk levels:
[5, 1, 3, 1, 5, 4, 3, 3, 1, 2, 3, 3, 3, 3, 3, 2, 4, 5, 3, 3, 5, 5, 1, 4, 2, 3, 4, 1, 3, 1, 4, 3, 4, 1, 1, 3, 3, 2, 3, 4, 3, 2, 4, 3, 4, 5, 3, 3]

Margin of error: 
[0, 0, -1, 0, 0, 0, 1, -1, 0, 0, 1, 0, 0, -1, 0, 0, -2, 0, 0, -1, -1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 1, -1, 0, 0, -1, 1, 0, 1, 0, -1, 1, 1, 2, 0, -3, 0]

