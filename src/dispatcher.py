from sklearn import ensemble

MODELS = {
    'randomforest': ensemble.RandomForestClassifier(n_estimators = 150, n_jobs = -1),
    'extratrees': ensemble.ExtraTreesClassifier(n_estimators = 150, n_jobs = -1)
}
