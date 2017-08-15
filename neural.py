import sys
import os
import time
from random import shuffle

from sklearn.feature_extraction.text import TfidfVectorizer
from  sklearn.metrics import classification_report
from sklearn import svm
import numpy as np
classes = ['pos', 'neg']
data = []
input_data = []
train_data = []
train_labels = []
test_data = []
test_labels = []
for fname in os.listdir('F:\ML\ML\sentiment labelled sentences',):
    if fname.endswith('.txt'):
        with open(os.path.join('F:\ML\ML\sentiment labelled sentences',fname), 'r') as input_file:
            input_data = input_file.read().split('\n')
            print(fname)
            for i in range(0, len(input_data) - 1):
                data.append(input_data[i])


shuffle(data)
print(data[1])

for i in range(0, len(data) - 600):
        x=data[i]
        train_data.append(x[:-2])
        if data[i].endswith('0'):
            train_labels.append('neg')
        else:
            train_labels.append('pos')



for i in range(601, len(data) - 1):
        x=data[i]
        test_data.append(x[:-2])
        if data[i].endswith('0'):
            test_labels.append('neg')
        else:
            test_labels.append('pos')



            # Create feature vectors
vectorizer = TfidfVectorizer(min_df=5,
                             max_df=0.8,
                             sublinear_tf=True,
                             use_idf=True)
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

# Perform classification with SVM, kernel=rbf
# classifier_rbf = svm.SVC(kernel= 'rbf')
# t0 = time.time()
# classifier_rbf.fit(train_vectors, train_labels)
# t1 = time.time()
# prediction_rbf = classifier_rbf.predict(test_vectors)
# t2 = time.time()
# time_rbf_train = t1 - t0
# time_rbf_predict = t2 - t1

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1 - t0
time_linear_predict = t2 - t1

# Perform classification with SVM, kernel=linear
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(test_vectors)
t2 = time.time()
time_liblinear_train = t1 - t0
time_liblinear_predict = t2 - t1

# Print results in a nice table
# print("Results for SVC(kernel=rbf)")
# print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
# print(classification_report(test_labels, prediction_rbf))
print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(test_labels, prediction_linear))
print("Results for LinearSVC()")
print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
print(classification_report(test_labels, prediction_liblinear))


# Neural Network

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(max_iter=300, alpha=0.00001)
mlp.fit(train_vectors, train_labels)

predictions = mlp.predict(test_vectors)

from sklearn.metrics import classification_report

print('Neural Network')
print(classification_report(test_labels, predictions))

from sklearn import linear_model

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(train_vectors, train_labels)
prediction_logreg = logreg.predict(test_vectors)
print('Logistic Regression')
print(classification_report(test_labels, prediction_logreg))