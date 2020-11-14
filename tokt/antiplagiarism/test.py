import pandas as pd
import os
import train
import numpy as np

data_dir = 'plagiarism_data'


predictor = train.model_fn('model')
test_data = pd.read_csv(os.path.join(data_dir, "test.csv"), header=None, names=None)

test_y = test_data.iloc[:,0]
test_x = test_data.iloc[:,1:]

test_y_preds = predictor.predict_proba(test_x)
print(test_x)

print('\nPredicted class labels: ')
print(test_y_preds)
print('\nTrue class labels: ')
print(test_y.values)

df = pd.concat([pd.DataFrame(test_x), pd.DataFrame(test_y), pd.DataFrame(
    test_y_preds), pd.DataFrame(map(max,test_y_preds)), pd.DataFrame(np.argmax(test_y_preds, axis=1))], axis=1)
df.columns=['c_1', 'c_3', 'c_8', 'lcs_word', 'class', 'predicted0', 'predicted1', 'predicted2', 'predicted3', 'max_predicted', 'pred_class']
print(df)
