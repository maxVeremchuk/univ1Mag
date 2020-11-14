import feature_engineering
import helpers
import train
import pandas as pd
import numpy as np
from itertools import islice
import generate_html

if __name__ == "__main__":

    paragraph_num = []

    f = open("text_to_check.txt", "r")
    data = f.read()
    for paragrah in data.split('\n\n'):
        paragraph_num.append(paragrah)

    origin_df = feature_engineering.numerical_dataframe(csv_file ='data/file_information_orig.csv')
    origin_text_df = helpers.create_text_column(origin_df)

    tasks = ['a', 'b', 'c', 'd', 'e']
    class_0 = [0]*len(tasks)*len(paragraph_num)
    category_0 = [0]*len(tasks)*len(paragraph_num)

    file_names = ["filename"]*len(tasks)*len(paragraph_num)
    file_names = [x + "_" + str(i) for i, x in enumerate(file_names)]
    print(file_names)

    index = pd.MultiIndex.from_product([paragraph_num, tasks])

    df = pd.DataFrame(index=index).reset_index()

    end_df = pd.concat([pd.DataFrame(df), pd.DataFrame(
        class_0), pd.DataFrame(category_0), pd.DataFrame(file_names)], axis=1)
    end_df.columns = ['Text', 'Task', 'Class', 'Category', 'File']
    end_df = end_df.append(origin_text_df, ignore_index=True)
    print(end_df)

    features_df = feature_engineering.generate_features(end_df, [1, 3, 8])
    print(features_df)

    predictor = train.model_fn('model')
    predicted = predictor.predict_proba(features_df.values[:-5])
    print(predicted)

    cycle = 0
    non_cycle = 1
    heavy_cycle = 1
    light_cycle = 1
    cut_cycle = 1
    paragrah_plagiarized = []
    tasks_len = len(tasks)
    predict_cases = [predicted[x:x+tasks_len] for x in range(0, len(predicted), tasks_len)]

    for idx, predict_case in enumerate(predict_cases):
        most_plagiarized = np.argmin((predict_case[:, 0]))
        class_idx = np.argmax(predict_case[most_plagiarized])
        paragtaph_dict = {}
        paragtaph_dict['class'] = class_idx
        if (class_idx == 0):
            paragtaph_dict['task'] = "-"
        else:
            paragtaph_dict['task'] = tasks[most_plagiarized]
        paragtaph_dict['prob'] = predict_case[most_plagiarized]
        paragtaph_dict['text'] = paragraph_num[idx]
        paragrah_plagiarized.append(paragtaph_dict)

    generate_html.generate_html(paragrah_plagiarized)




