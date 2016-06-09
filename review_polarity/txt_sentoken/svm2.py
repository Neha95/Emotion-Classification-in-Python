

import sys
import os
import time
import pandas as pd
import pylab as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
def usage():
    print("Usage:")
    print("python %s <data_dir>" % sys.argv[0])
#for graph
if __name__ == '__main__':

    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    data_dir = sys.argv[1]
    classes = ['pos', 'neg']

    # Read the data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
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

    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 use_idf=True,decode_error='ignore')
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)
#Test
    # Perform classification with SVM, kernel=rbf
#   classifier_rbf = svm.SVC(C=50.0,gamma=0.3)
#   t0 = time.time()
#   classifier_rbf.fit(train_vectors, train_labels)
#   t1 = time.time()
#   prediction_rbf = classifier_rbf.predict(test_vectors)
#   t2 = time.time()
#   time_rbf_train = t1-t0
#   time_rbf_predict = t2-t1


    results = []
#n can only be int!
    for n in range(1,401,50):
	x=401-n*1.0
	y=n*0.01
    	clf = svm.SVC(C=x,gamma=y)
    	clf.fit(train_vectors, train_labels)
    	preds = clf.predict(test_vectors)
   	accuracy=accuracy_score(test_labels, preds)
    	print "C= %f,Gamma: %f, Accuracy: %3f" % (x,y, accuracy)

    	results.append([x, accuracy])

results = pd.DataFrame(results, columns=["x", "accuracy"])

pl.plot(results.x, results.accuracy)
pl.title("Accuracy with Increasing C and Increasing Gamma")
pl.show()

 