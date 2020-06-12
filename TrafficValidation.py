import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
import sklearn

pip install requests
pip install tabulate
pip install "colorama>=0.3.8"
pip install future
pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o

import h2o
h2o.init()
h2o.demo("glm")

dir(h2o.estimators)
model = h2o.estimators.H2ORandomForestEstimator(ntrees=50, max_depth=20, nfolds=10)



studies = pd.read_csv('./CleanedData/traffic_studies_with_features')
studies = studies[studies['setyear'] < 2019]
len(studies)
#means_by_year = studies.groupby('setyear').mean()['aadb']
#means_by_year[2010]
#studies['adjusted aadb to 2018'] = studies.apply(lambda x: x['aadb']/means_by_year.loc[x['setyear']]*means_by_year.loc[2018],axis=1)

studies = studies.drop(['Unnamed: 0','setyear', 'u', 'v', 'key', 'osmid', 'geometry'],axis=1)

studies_no_encoding = pd.read_csv('./CleanedData/traffic_studies_with_features_no_encoding')

studies_no_encoding = studies_no_encoding.drop(['Unnamed: 0','setyear', 'u', 'v', 'key', 'osmid', 'geometry'],axis=1)

response='aadb'
predictors = studies_no_encoding.drop('aadb',axis=1)
rf = h2o.estimators.H2ORandomForestEstimator(ntrees=50, max_depth=20, nfolds=10)
rf.train(x=predictors, y=response, training_frame=h2o.H2OFrame(studies_no_encoding))

h2o.H2OFrame(studies_no_encoding)

X = studies.drop('aadb',axis=1)
y = studies['aadb']

scaler = StandardScalar()
X_scaled = preprocessing.StandardScaler(X)


pipeline = make_pipeline([
    ('lasso', linear_model.Lasso()),
    ('ada', AdaBoostRegressor()),
])
parameters = [
    {
        'clf': (linear_model.Lasso(),)
    }, {
        'clf': (AdaBoostRegressor(),),
        'clf__n_estimators': (1, 5, 25, 100)
    }
]
grid_search = GridSearchCV(pipeline, parameters, cv=10, scoring = 'neg_mean_squared_error')


len(studies)

## Don't expect linear models to do well
clf = make_pipeline(preprocessing.StandardScaler(),linear_model.Lasso())
cv_results = cross_validate(clf, X, y, scoring = 'neg_mean_squared_error',cv=479)
cv_results['test_score'].mean()

## AdaBoost
clf = make_pipeline(preprocessing.StandardScaler(),AdaBoostRegressor(loss = 'square',n_estimators=10,random_state=0))
cv_results = cross_validate(clf, X, y, scoring = 'neg_mean_squared_error',cv=479)
cv_results['test_score'].mean()


## RF Regression
clf = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(max_depth=10, random_state=0))
cv_results = cross_validate(clf, X, y, scoring = 'neg_mean_absolute_error',cv=479)
pd.Series(-cv_results['test_score']).mean()
(pd.Series(cv_results['test_score'])**2).mean()

clf = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(max_depth=10, random_state=0))
cv_results = cross_validate(clf, X, y, scoring = 'neg_mean_squared_error',cv=479)
cv_results['test_score'].mean()


## KNN Regression
clf = make_pipeline(preprocessing.StandardScaler(),KNeighborsRegressor(n_neighbors=4))
cv_results = cross_validate(clf, X, y, scoring = 'neg_mean_squared_error',cv=479)
cv_results['test_score'].mean()


model = RandomForestRegressor(max_depth=5, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/479, random_state=1)

model.fit(X_train,y_train)
(model.predict(X_test) - y_test).hist()
y_test.hist()

np.sqrt(163458.8413393822)

((studies['adjusted aadb to 2018']-studies['adjusted aadb to 2018'].mean())**2).mean()
234**2

np.sqrt(((studies['adjusted aadb to 2018']-studies['adjusted aadb to 2018'].mean())**2).sum())
np.abs(studies['aadb']-studies['aadb'].mean()).mean()
studies['aadb'].mean()
y_train.describe()


pd.Series(-cv_results['test_score'])

np.sqrt(((y_train-y_train.mean())**2)).mean()
