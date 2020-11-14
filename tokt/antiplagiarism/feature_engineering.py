import pandas as pd
import numpy as np
import os
import helpers
from sklearn.feature_extraction.text import CountVectorizer

def numerical_dataframe(csv_file='data/file_information.csv'):
    df = pd.read_csv(csv_file)
    df.loc[:,'Class'] =  df.loc[:,'Category'].map({'non': 0, 'heavy': 1, 'light': 2, 'cut': 3, 'orig': -1})
    df.loc[:,'Category'] =  df.loc[:,'Category'].map({'non': 0, 'heavy': 1, 'light': 2, 'cut': 3, 'orig': -1})

    return df

def calculate_containment(df, n, answer_filename):
    answer_text, answer_task  = df[df.File == answer_filename][['Text', 'Task']].iloc[0]
    source_text = df[(df.Task == answer_task) & (df.Class == -1)]['Text'].iloc[0]

    counter = CountVectorizer(analyzer='word', ngram_range=(n,n))
    ngrams_array = counter.fit_transform([answer_text, source_text]).toarray()

    count_common_ngrams = sum(min(a, s) for a, s in zip(*ngrams_array))
    count_ngrams_a = ngrams_array[0].sum()

    if(np.isnan(count_common_ngrams) or count_ngrams_a==0):
        return 0
    return count_common_ngrams / count_ngrams_a

def lcs_norm_word(answer_text, source_text):
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

    if(np.isnan(lcs) or a_word_count==0):
        return 0
    return lcs / a_word_count

def create_containment_features(df, n, column_name=None):
    containment_values = []

    if(column_name==None):
        column_name = 'c_'+str(n) # c_1, c_2, .. c_n

    for i in df.index:
        file = df.loc[i, 'File']

        if df.loc[i,'Category'] > -1:
            c = calculate_containment(df, n, file)
            containment_values.append(c)
        else:
            containment_values.append(-1)

    print(str(n)+'-gram containment features created!')
    return containment_values

def create_lcs_features(df, column_name='lcs_word'):

    lcs_values = []

    for i in df.index:
        if df.loc[i,'Category'] > -1:
            answer_text = df.loc[i, 'Text']
            task = df.loc[i, 'Task']
            orig_rows = df[(df['Class'] == -1)]
            orig_row = orig_rows[(orig_rows['Task'] == task)]
            source_text = orig_row['Text'].values[0]

            lcs = lcs_norm_word(answer_text, source_text)
            lcs_values.append(lcs)
        else:
            lcs_values.append(-1)

    print('LCS features created!')
    return lcs_values

def train_test_data(complete_df, features_df, selected_features):
    df = pd.concat([complete_df, features_df[selected_features]], axis=1)
    df_train = df[df.Datatype == 'train']
    df_test = df[df.Datatype == 'test']

    train_x = df_train[selected_features].values
    train_y = df_train['Class'].values

    test_x = df_test[selected_features].values
    test_y = df_test['Class'].values

    return (train_x, train_y), (test_x, test_y)

def make_csv(x, y, filename, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    pd.concat([pd.DataFrame(y), pd.DataFrame(x)], axis=1).to_csv(os.path.join(data_dir, filename), header=False, index=False)

    print('Path created: '+str(data_dir)+'/'+str(filename))

def generate_features(df, ngram_range):
    features_list = []
    all_features = np.zeros((len(ngram_range)+1, len(df)))

    i=0
    for n in ngram_range:
        column_name = 'c_'+str(n)
        features_list.append(column_name)
        all_features[i]=np.squeeze(create_containment_features(df, n))
        i+=1
    features_list.append('lcs_word')
    all_features[i]= np.squeeze(create_lcs_features(df))

    features_df = pd.DataFrame(np.transpose(all_features), columns=features_list)
    return features_df

def show_corr(features_df):
    corr_matrix = features_df.corr().abs().round(2)
    print(corr_matrix)

if __name__ == "__main__":
    transformed_df = numerical_dataframe(csv_file ='data/file_information.csv')
    text_df = helpers.create_text_column(transformed_df)
    print(text_df)

    # sample_text = text_df.iloc[0]['Text']
    # print('Sample processed text:\n\n', sample_text)

    random_seed = 1
    complete_df = helpers.train_test_dataframe(text_df, random_seed=random_seed)

    #show_corr(generate_features(complete_df, [1, 5, 10]))

    data_dir = 'plagiarism_data'
    selected_features = ['c_1', 'c_3', 'c_8', 'lcs_word']
    features_df = generate_features(complete_df, [1, 3, 8])

    (train_x, train_y), (test_x, test_y) = train_test_data(complete_df, features_df, selected_features)

    make_csv(train_x, train_y, filename='train.csv', data_dir=data_dir)
    make_csv(test_x, test_y, filename='test.csv', data_dir=data_dir)
