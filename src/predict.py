import os
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
from . import dispatcher

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")
MODEL_PATH = os.environ.get("MODEL_PATH")


def predict(test_data_path, model_type, model_path):

    print('Data Reading...')
    data = pd.read_csv(test_data_path)
    test_idx = data["id"].values
    predictions = None

    print('Folds...')
    for FOLD in range(1):
        print('Fold_{}'.format(FOLD))
        data = pd.read_csv(test_data_path)
        encoders = joblib.load(os.path.join(model_path, f"{model_type}_labelencoder_{FOLD}.pkl"))
        columns = joblib.load(os.path.join(model_path, f"{model_type}_columns_{FOLD}.pkl"))
        for col in encoders:
            label_encoder = encoders[col]
            data.loc[:, col] = data.loc[:, col].astype(str).fillna("NONE")
            data.loc[:, col] = label_encoder.transform(data[col].values.tolist())

        classifier = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}.pkl"))

        data = data[columns]
        preds = classifier.predict_proba(data)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])
    return sub


if __name__ == "__main__":
    submission = predict(test_data_path=TEST_DATA,
                         model_type=MODEL,
                         model_path=MODEL_PATH)
    submission.loc[:, "id"] = submission.loc[:, "id"].astype(int)
    submission.to_csv(f"{MODEL_PATH}/rf_submission.csv", index=False)
    print('Predictions saved.')
