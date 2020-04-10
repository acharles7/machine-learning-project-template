import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

FOLD_MAPPING = {
    0:[1, 2, 3, 4],
    1:[0, 2, 3, 4],
    2:[0, 1, 3, 4],
    3:[0, 1, 2, 4],
    4:[0, 1, 2, 3]
}


if __name__ == "__main__":

    print('###### Data Reading #######')

    df = pd.read_csv(TRAINING_DATA)
    # test_df = pd.read_csv(TEST_DATA)

    print('###### Data Preprocessing #######')
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold == FOLD]

    y_train = train_df.target.values
    y_valid = valid_df.target.values

    train_df = train_df.drop(['index', 'id', 'target', 'kfold'], axis=1)
    valid_df = valid_df.drop(['index', 'id', 'target', 'kfold'], axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoder = {}

    for col in train_df.columns:
        lbl_encoder = preprocessing.LabelEncoder()
        train_df.loc[:, col] = train_df.loc[:, col].astype(str).fillna("NONE")
        valid_df.loc[:, col] = valid_df.loc[:, col].astype(str).fillna("NONE")

        lbl_encoder.fit(train_df[col].values.tolist() + valid_df[col].values.tolist())

        train_df.loc[:, col] = lbl_encoder.transform(train_df[col].values.tolist())
        valid_df.loc[:, col] = lbl_encoder.transform(valid_df[col].values.tolist())

        label_encoder[col] = lbl_encoder

    print('###### Data Training #######')
    classifier = dispatcher.MODELS[MODEL]
    classifier.fit(train_df, y_train)
    print('###### Data Testing #######')
    preds = classifier.predict_proba(valid_df)[:, 1]
    print('###### Performance Checking #######')
    print('ROC Score {}'.format(metrics.roc_auc_score(y_valid, preds)))

    # joblib.dump(label_encoder, f'models/{MODEL}_label_encoder.pkl')
    # joblib.dump(classifier, f'models/{MODEL}_classifier.pkl')
