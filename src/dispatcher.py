from sklearn import ensemble
from sklearn import linear_model

MODELS = {
    'random_forest': ensemble.RandomForestClassifier(n_estimators = 150, n_jobs = -1),
    'extra_trees': ensemble.ExtraTreesClassifier(n_estimators = 150, n_jobs = -1),
    'linear_regression': linear_model.LinearRegression(),
    'logistic_regression': linear_model.LogisticRegression()
}
