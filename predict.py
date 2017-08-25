import pandas as pd
#import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

data = pd.read_csv('N1.csv')
display(data.head())

#Data Exploration

#Total number of matches (1549 matches)
n_matches = data.shape[0]

#Calculate the number of features.
n_features = data.shape[1] - 1

#Calculate the number of home wins
n_homewins = len(data[data.FTR == 'H'])

n_awaywins = len(data[data.FTR == 'A'])

n_draws = len(data[data.FTR == 'D'])

#Calculate the winrate
win_rate = (float(n_homewins) / (n_matches)) * 100

# Print the results
print "Total number of matches: {}".format(n_matches)
print "Number of features: {}".format(n_features)
print "Number of matches won by home team: {}".format(n_homewins)
print "Number of matches won by away team: {}".format(n_awaywins)
print "Number of matches draws by home team: {}".format(n_draws)
print "Win rate of home team: {:.2f}%".format(win_rate)
X_all = data.drop(['FTR'],1)
Y_all = data['FTR']

from sklearn.preprocessing import scale

#Center to the mean and component wise scale to unit variance.
#Full Time Result (H=Home Win, D=Draw, A=Away Win)
#HTGD - Home team goal difference
#ATGD - away team goal difference
#HTP - Home team points
#ATP - Away team points
#DiffFormPts Diff in points
#DiffLP - Differnece in last years prediction
cols = [['FTHG','FTAG']]

for col in cols:
    X_all[col] = scale(X_all[col])

def preproces_features(X):

    output = pd.DataFrame(index = X.index)

    for col, col_data in X.iteritems():
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data,prefix=col)

        if np.any(np.isnan(col_data)):
            col_data = pd.get_dummies(0,prefix=col)

        output = output.join(col_data)
    return output

X_all = preproces_features(X_all)

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=50, random_state=2, stratify=Y_all)

from time import time

from sklearn.metrics import f1_score

def train_classifier(clf, X_train, Y_train):

    start = time()
    clf.fit(X_train, Y_train)
    end = time()

    print "Trained model in {:.4f} seconds".format(end - start)

def predict_labels(clf, features, target):

    start = time()
    y_pred = clf.predict(features)
    end = time()

    print "Made predictions in {:.4f} seconds.".format(end - start)

    return f1_score(target, y_pred, pos_label='H'), sum(target == y_pred) / float(len(y_pred))

def train_predict(clf, X_train, Y_train, X_test, Y_test):
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))

    train_classifier(clf, X_train, Y_train)

    f1, acc = predict_labels(clf, X_train, Y_train)
    print f1, acc
    print "F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc)

    f1, acc = predict_labels(clf, X_test, y_test)
    print "F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc)

clf = SVC(random_state = 912, kernel='rbf')

train_predict(clf, X_train, Y_train, X_test, Y_test)
print ''
