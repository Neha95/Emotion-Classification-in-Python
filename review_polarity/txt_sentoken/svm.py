import sys
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
def usage():
    print("Couldn't find folders:")
    print("python %s <data_dir>" % sys.argv[0])

if __name__ == '__main__':

    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    data_dir = sys.argv[1]
    classes = ['pos', 'neg']

    #Reading the Data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
#For pos and neg classes dividing 90% into test
    for curr_class in classes:
        dirname = os.path.join(data_dir, curr_class)
        for fname in os.listdir(dirname):
            with open(os.path.join(dirname, fname), 'r') as f:
                content = f.read()
                if fname.startswith('cv9'):
                    test_data.append(content)
                    test_labels.append(curr_class)
                else:
                    train_data.append(content)
                    train_labels.append(curr_class)

    # Create feature vectors, decode error ignore because of Python 3 to 2.7
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 use_idf=True,decode_error='ignore')
#fit to train data
    train_vectors = vectorizer.fit_transform(train_data)
#transform test data
    test_vectors = vectorizer.transform(test_data)

    # classification with SVC, kernel=rbf
    classifier_rbf = svm.SVC(C=50.0,gamma=0.3)
#initial time
    t0 = time.time()
    classifier_rbf.fit(train_vectors, train_labels)
#after fitting
    t1 = time.time()
    prediction_rbf = classifier_rbf.predict(test_vectors)
#after prediction
    t2 = time.time()
    time_rbf_train = t1-t0
    time_rbf_predict = t2-t1

    # classification with SVM, kernel=linear using svmlib
    classifier_linear = svm.SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1-t0
    time_linear_predict = t2-t1

    # classification with LinearSVC, kernel=linear uses liblinear
    classifier_liblinear = svm.LinearSVC()
    t0 = time.time()
    classifier_liblinear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_liblinear = classifier_liblinear.predict(test_vectors)
    t2 = time.time()
    time_liblinear_train = t1-t0
    time_liblinear_predict = t2-t1

    # Print results in a nice table
# support is number of occurence of each class in correct target values, basically how many pos and neg
    print("Results for SVC(kernel=rbf)")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(test_labels, prediction_rbf))
#separately getting accuracy
    print("Accuracy is : ")
    print(accuracy_score(test_labels,prediction_rbf))

    print("Results for SVC(kernel=linear)")
    print("Training time: %fs; Prediction time: %fs" %(time_linear_train, time_linear_predict))
    print(classification_report(test_labels, prediction_linear))
    print("Accuracy is : ")
    print(accuracy_score(test_labels,prediction_linear))

    print("Results for LinearSVC()")
    print("Training time: %fs; Prediction time: %fs"% (time_liblinear_train, time_liblinear_predict))
    print(classification_report(test_labels, prediction_liblinear))
    print("Accuracy is : ")
    print(accuracy_score(test_labels,prediction_liblinear))