import pandas as pd
#import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
#from IPython.display import display

data = pd.read_csv('N1.csv')
#display(data.head())

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
cols = [['HTGD','ATGD','HTP','ATP','DiffLP']]
print X_all.head()
for col in cols:
    X_all[col] = scale(X_all[col])

#last 3 wins for bot sides
X_all.HM1 = X_all.HM1.astype('str')
X_all.HM2 = X_all.HM2.astype('str')
X_all.HM3 = X_all.HM3.astype('str')
X_all.AM1 = X_all.AM1.astype('str')
X_all.AM2 = X_all.AM2.astype('str')
X_all.AM3 = X_all.AM3.astype('str')

def preproces_features(X):

    output = pd.DataFrame(index = X.index)

    for col, col_data in X.iteritems():
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data,prefix=col)

        output = output.join(col_data)
    return output

X_all = preproces_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))
