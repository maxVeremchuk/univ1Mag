import pandas as pd
#import boto3
#import sagemaker
from sagemaker.sklearn.estimator import SKLearn
import os
import train

# sagemaker_session = sagemaker.Session(boto3.session.Session())
# role = sagemaker.get_execution_role()

# # create an S3 bucket
# bucket = sagemaker_session.default_bucket()

data_dir = 'plagiarism_data'

# set prefix, a descriptive name for a directory  
prefix = 'plagiarism-detection'

# upload all data to S3
# input_data = sagemaker_session.upload_data(path=data_dir, bucket=bucket, key_prefix=prefix)
# print(input_data)

# estimator = SKLearn(entry_point="train.py",
#                     source_dir="source_sklearn",
#                     role=role,
#                     train_instance_count=1,
#                     train_instance_type='ml.c4.xlarge')

# estimator.fit({'train': input_data})

# predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.t2.medium')

# read in test data, assuming it is stored locally
predictor = train.model_fn('model')
test_data = pd.read_csv(os.path.join(data_dir, "test.csv"), header=None, names=None)

test_y = test_data.iloc[:,0]
test_x = test_data.iloc[:,1:]

test_y_preds = predictor.predict(test_x)

print('\nPredicted class labels: ')
print(test_y_preds)
print('\nTrue class labels: ')
print(test_y.values)

df = pd.concat([pd.DataFrame(test_x), pd.DataFrame(test_y), pd.DataFrame(test_y_preds)], axis=1)
df.columns=['c_1', 'c_5', 'c_10', 'lcs_word', 'class', 'predicted']
print(df)
