export TRAINING_DATA=input/train_folds.csv
export TEST_DATA=input/test.csv
export MODEL_PATH=/Users/lomesh/Downloads/models/
export MODEL=$1

# python -m src.predict
FOLD=0 python -m src.train
# FOLD=1 python -m src.train
# FOLD=2 python -m src.train
# FOLD=3 python -m src.train
# FOLD=4 python -m src.train
