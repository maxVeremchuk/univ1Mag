import pandas as pd
import numpy as np
import os
import helpers 
from sklearn.feature_extraction.text import CountVectorizer

csv_file = 'data/file_information.csv'
plagiarism_df = pd.read_csv(csv_file)

def numerical_dataframe(csv_file='data/file_information.csv'):
    '''Reads in a csv file which is assumed to have `File`, `Category` and `Task` columns.
       This function does two things: 
       1) converts `Category` column values to numerical values 
       2) Adds a new, numerical `Class` label column.
       The `Class` column will label plagiarized answers as 1 and non-plagiarized as 0.
       Source texts have a special label, -1.
       :param csv_file: The directory for the file_information.csv file
       :return: A dataframe with numerical categories and a new `Class` label column'''

    df = pd.read_csv(csv_file)
    df.loc[:,'Class'] =  df.loc[:,'Category'].map({'non': 0, 'heavy': 1, 'light': 1, 'cut': 1, 'orig': -1})
    df.loc[:,'Category'] =  df.loc[:,'Category'].map({'non': 0, 'heavy': 1, 'light': 2, 'cut': 3, 'orig': -1})
    
    return df

transformed_df = numerical_dataframe(csv_file ='data/file_information.csv')

text_df = helpers.create_text_column(transformed_df)

# sample_text = text_df.iloc[0]['Text']
# print('Sample processed text:\n\n', sample_text)

random_seed = 1
complete_df = helpers.train_test_dataframe(text_df, random_seed=random_seed)

def calculate_containment(df, n, answer_filename):
    '''Calculates the containment between a given answer text and its associated source text.
       This function creates a count of ngrams (of a size, n) for each text file in our data.
       Then calculates the containment by finding the ngram count for a given answer text, 
       and its associated source text, and calculating the normalized intersection of those counts.
       :param df: A dataframe with columns,
           'File', 'Task', 'Category', 'Class', 'Text', and 'Datatype'
       :param n: An integer that defines the ngram size
       :param answer_filename: A filename for an answer text in the df, ex. 'g0pB_taskd.txt'
       :return: A single containment value that represents the similarity
           between an answer text and its source text.
    '''
    
    answer_text, answer_task  = df[df.File == answer_filename][['Text', 'Task']].iloc[0]
    source_text = df[(df.Task == answer_task) & (df.Class == -1)]['Text'].iloc[0]
    
    counter = CountVectorizer(analyzer='word', ngram_range=(n,n))
    ngrams_array = counter.fit_transform([answer_text, source_text]).toarray()
    
    count_common_ngrams = sum(min(a, s) for a, s in zip(*ngrams_array))
    count_ngrams_a = ngrams_array[0].sum()
    
    return count_common_ngrams / count_ngrams_a

def lcs_norm_word(answer_text, source_text):
    '''Computes the longest common subsequence of words in two texts; returns a normalized value.
       :param answer_text: The pre-processed text for an answer text
       :param source_text: The pre-processed text for an answer's associated source text
       :return: A normalized LCS value'''
    
    # your code here
    a_words = answer_text.split()
    s_words = source_text.split()
    
    a_word_count = len(a_words)
    s_word_count = len(s_words)
        
    lcs_matrix = np.zeros((s_word_count + 1, a_word_count + 1), dtype=int)

    for s, s_word in enumerate(s_words, 1):
        for a, a_word in enumerate(a_words, 1):
            if s_word == a_word:
                lcs_matrix[s][a] = lcs_matrix[s-1][a-1] + 1
            else:
                lcs_matrix[s][a] = max(lcs_matrix[s-1][a], lcs_matrix[s][a-1])
    
    lcs = lcs_matrix[s_word_count][a_word_count]
    
    return lcs / a_word_count

def create_containment_features(df, n, column_name=None):
    containment_values = []
    
    if(column_name==None):
        column_name = 'c_'+str(n) # c_1, c_2, .. c_n
    
    # iterates through dataframe rows
    for i in df.index:
        file = df.loc[i, 'File']
        # Computes features using calculate_containment function
        if df.loc[i,'Category'] > -1:
            c = calculate_containment(df, n, file)
            containment_values.append(c)
        # Sets value to -1 for original tasks 
        else:
            containment_values.append(-1)
    
    print(str(n)+'-gram containment features created!')
    return containment_values

def create_lcs_features(df, column_name='lcs_word'):
    
    lcs_values = []
    
    # iterate through files in dataframe
    for i in df.index:
        # Computes LCS_norm words feature using function above for answer tasks
        if df.loc[i,'Category'] > -1:
            # get texts to compare
            answer_text = df.loc[i, 'Text'] 
            task = df.loc[i, 'Task']
            # we know that source texts have Class = -1
            orig_rows = df[(df['Class'] == -1)]
            orig_row = orig_rows[(orig_rows['Task'] == task)]
            source_text = orig_row['Text'].values[0]

            # calculate lcs
            lcs = lcs_norm_word(answer_text, source_text)
            lcs_values.append(lcs)
        # Sets to -1 for original tasks 
        else:
            lcs_values.append(-1)

    print('LCS features created!')
    return lcs_values

def train_test_data(complete_df, features_df, selected_features):
    '''Gets selected training and test features from given dataframes, and 
       returns tuples for training and test features and their corresponding class labels.
       :param complete_df: A dataframe with all of our processed text data, datatypes, and labels
       :param features_df: A dataframe of all computed, similarity features
       :param selected_features: An array of selected features that correspond to certain columns in `features_df`
       :return: training and test features and labels: (train_x, train_y), (test_x, test_y)'''
    
    df = pd.concat([complete_df, features_df[selected_features]], axis=1)    
    df_train = df[df.Datatype == 'train']
    df_test = df[df.Datatype == 'test']

    # get the training features
    train_x = df_train[selected_features].values
    # And training class labels (0 or 1)
    train_y = df_train['Class'].values
    print("----------------------")
    print(train_x)
    print(train_y)
    print(df_train['Class'])
    
    
    # get the test features and labels
    test_x = df_test[selected_features].values
    test_y = df_test['Class'].values
    
    return (train_x, train_y), (test_x, test_y)

def make_csv(x, y, filename, data_dir):
    '''Merges features and labels and converts them into one csv file with labels in the first column.
       :param x: Data features
       :param y: Data labels
       :param file_name: Name of csv file, ex. 'train.csv'
       :param data_dir: The directory where files will be saved
       '''
    # make data dir, if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # your code here
    # first column is the labels and rest is features 
    pd.concat([pd.DataFrame(y), pd.DataFrame(x)], axis=1).to_csv(os.path.join(data_dir, filename), header=False, index=False)
    
    # nothing is returned, but a print statement indicates that the function has run
    print('Path created: '+str(data_dir)+'/'+str(filename))

def generate_features(ngram_range):
    features_list = []
    all_features = np.zeros((len(ngram_range)+1, len(complete_df)))

    i=0
    for n in ngram_range:
        column_name = 'c_'+str(n)
        features_list.append(column_name)
        all_features[i]=np.squeeze(create_containment_features(complete_df, n))
        i+=1
    features_list.append('lcs_word')
    all_features[i]= np.squeeze(create_lcs_features(complete_df))

    features_df = pd.DataFrame(np.transpose(all_features), columns=features_list)
    return features_df

def show_corr(features_df):
    corr_matrix = features_df.corr().abs().round(2)
    print(corr_matrix)


#show_corr(generate_features([1, 5, 10]))

data_dir = 'plagiarism_data'
selected_features = ['c_1', 'c_5', 'c_10']
features_df = generate_features([1, 5, 10])

(train_x, train_y), (test_x, test_y) = train_test_data(complete_df, features_df, selected_features)

make_csv(train_x, train_y, filename='train.csv', data_dir=data_dir)
make_csv(test_x, test_y, filename='test.csv', data_dir=data_dir)