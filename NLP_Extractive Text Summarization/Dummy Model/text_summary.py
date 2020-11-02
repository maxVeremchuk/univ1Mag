import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import *
import re


dir_path = "../dataset/preprocessed-input-directory/cnn-dailymail.test.label.singleoracle"
text_path = "../dataset/CNN-DM-Filtered-TokenizedSegmented/test/mainbody/"



def create_freq_table(text_string):
    stopwords_list = set(stopwords.words('english'))

    words = word_tokenize(text_string)

    ps = PorterStemmer()

    freq_table = {}

    for word in words:
        #stem word
        word = ps.stem(word)

        #remove stopwords
        if word in stopwords_list:
            continue
        elif word in freq_table:
            freq_table[word] += 1
        else:
            freq_table[word] = 1

    return freq_table



def score_sentences(sentences, freq_table):

    sentence_value = {}

    for sentence in sentences:
        word_count_in_sentence = len(word_tokenize(sentence))

        for wordValue in freq_table:

            if wordValue.lower() in sentence.lower():
                if sentence in sentence_value:
                    sentence_value[sentence] += freq_table[wordValue]
                else:
                    sentence_value[sentence] = freq_table[wordValue]

        sentence_value[sentence] = sentence_value[sentence] // word_count_in_sentence
    return sentence_value

def find_average_score(sentence_value):
    sum_values = 0

    for entry in sentence_value:
        sum_values += sentence_value[entry]

    average = int(sum_values/len(sentence_value))

    return average

def generate_summary(sentences, sentence_value, threshold):
    sentence_count = 0

    summary = ''

    for sentence in sentences:
        if sentence in sentence_value and sentence_value[sentence] > threshold:
            summary += " " + sentence
            sentence_count += 1

    return summary


def get_dataset_dict():
    gold_summary_file = open(dir_path, 'r')
    lines = gold_summary_file.readlines()

    dataset_dict = {}
    filenames = []
    file_name = ""
    gold_summary = []
    for line in lines:
        if line == "\n":
            if file_name == "dailymail":
                file_name = ""
                gold_summary = []
            else:
                dataset_dict[file_name.strip()] = gold_summary
                filenames.append(file_name.strip())
                gold_summary = []
        elif line.startswith("cnn"):
            file_name = line
        elif line.startswith("dailymail"):
            file_name = "dailymail"
        else:
            gold_summary.append(int(line))
    return dataset_dict, filenames


if __name__ == "__main__":

    dataset_dict, filenames = get_dataset_dict()
    for filename in filenames:
        text = open(text_path + filename[4:] + ".mainbody", 'r').readlines()

        sentences = [line.strip() for line in text]
        freq_table = create_freq_table(" ".join(sentences))
        sentence_scores = score_sentences(sentences, freq_table)
        threshold = find_average_score(sentence_scores)
        summary = generate_summary(sentences, sentence_scores, 1.0 * threshold)
        print(summary)
        break
