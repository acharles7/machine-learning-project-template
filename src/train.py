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
MODEL_PATH = os.environ.get("MODEL_PATH")

FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    print('Data Reading...')
    df = pd.read_csv(TRAINING_DATA)
    test_df = pd.read_csv(TEST_DATA)

    train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold==FOLD].reset_index(drop=True)

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["id", "target", "kfold"], axis=1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    print('Data Preprocessing...')
    label_encoders = {}
    for col in train_df.columns:
        label_encoder = preprocessing.LabelEncoder()
        train_df.loc[:, col] = train_df.loc[:, col].astype(str).fillna("NONE")
        valid_df.loc[:, col] = valid_df.loc[:, col].astype(str).fillna("NONE")
        test_df.loc[:, col] = test_df.loc[:, col].astype(str).fillna("NONE")

        label_encoder.fit(train_df[col].values.tolist() +
                            valid_df[col].values.tolist() +
                            test_df[col].values.tolist())
        train_df.loc[:, col] = label_encoder.transform(train_df[col].values.tolist())
        valid_df.loc[:, col] = label_encoder.transform(valid_df[col].values.tolist())
        label_encoders[col] = label_encoder

    print('Data Training...')
    classifier = dispatcher.MODELS[MODEL]
    classifier.fit(train_df, ytrain)
    preds = classifier.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(yvalid, preds))

    print('Data Dumping...')
    joblib.dump(label_encoders, os.path.join(MODEL_PATH, f"{MODEL}_labelencoder_{FOLD}.pkl"))
    joblib.dump(clf, os.path.join(MODEL_PATH, f"{MODEL}_{FOLD}.pkl"))
    joblib.dump(train_df.columns, os.path.join(MODEL_PATH, f"{MODEL}_columns_{FOLD}.pkl"))

    print('Training Done')
