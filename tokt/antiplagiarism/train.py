from __future__ import print_function

import argparse
import os
import pandas as pd
import joblib
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def model_fn(model_dir):

    print("Loading model.")

    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-dir', type=str, default='model')
    parser.add_argument('--data-dir', type=str, default='plagiarism_data')

    args = parser.parse_args()

    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    train_y = train_data.iloc[:, 0]
    train_x = train_data.iloc[:, 1:]

    svm = LinearSVC()

    model = CalibratedClassifierCV(svm)
    model.fit(train_x, train_y)

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
