#Using LazyPredict on titanic model

import pyforest
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.metrics import accuracy_score

# importing .csv files using Pandas
train = pd.read_csv(‘train.csv’)
test = pd.read_csv(‘test.csv’)

train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 2)
train.drop(columns=[‘Name’,’Ticket’,’Cabin’, ‘PassengerId’, ‘Parch’, ‘Embarked’], inplace=True)
X = train.drop([‘Survived’], axis=1)
y = train.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

import lazypredict
from lazypredict.Supervised import LazyClassifier

clf = LazyClassifier(verbose=0,ignore_warnings=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
models

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

rf = LogisticRegression()
rf.fit(X_train, y_train)
y_pred_lr = rf.predict(X_test)