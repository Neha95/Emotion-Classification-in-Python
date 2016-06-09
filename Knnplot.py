

import sys
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def usage():
    print("Usage:")
    print("python %s <data_dir>" % sys.argv[0])

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

    # Perform classification with SVM, kernel=rbf

    t0 = time.time()
   
    t1 = time.time()
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.xlim(0,50)
    plt.ylim(0,1)
    for k in range(0,50):
        classifier_rbf = KNeighborsClassifier(n_neighbors=k).fit(train_vectors, train_labels)
            
        prediction_rbf = classifier_rbf.predict(test_vectors)
        prec=accuracy_score(test_labels,prediction_rbf)
        #rec=recall_score(test_labels, prediction_rbf, average='macro')
        print(prec)
        plt.plot(k,prec)

    plt.show()
    t2 = time.time()
    time_rbf_train = t1-t0
    time_rbf_predict = t2-t1


    # Print results in a nice table

    