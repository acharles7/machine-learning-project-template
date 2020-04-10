export TRAINING_DATA=input/train_folds.csv
export FOLD=0
export MODEL=$1
export TEST_DATA=input/test.csv

python -m src.train
