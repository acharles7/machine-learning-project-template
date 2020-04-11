import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":

    data = pd.read_csv("input/train.csv")
    data['kfold'] = -1

    data = data.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(data, data.target.values)):
        data.loc[val_idx, 'kfold'] = fold

    data.to_csv("input/train_folds.csv", index=False)
