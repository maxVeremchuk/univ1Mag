# -*- coding: utf-8 -*-
from dialogue_manager import *
import os
import en_core_web_lg
import spacy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils import *
import numpy as np
import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_features(X_train, X_test, vectorizer_path):
    """Performs TF-IDF transformation and dumps the model."""

    tfidf_vectorizer = TfidfVectorizer(
        min_df=5, max_df=0.9, token_pattern=re.compile('\S+'))
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)
    pickle.dump(tfidf_vectorizer, open(vectorizer_path, "wb"))
    return X_train, X_test


sample_size = 200000

dialogue_df = pd.read_csv('data/dialogues.tsv',
                          sep='\t').sample(sample_size, random_state=0)
stackoverflow_df = pd.read_csv(
    'data/tagged_posts.tsv', sep='\t').sample(sample_size, random_state=0)

dialogue_df['text'] = [text_prepare(text) for text in dialogue_df['text']]
stackoverflow_df['title'] = [text_prepare(
    title) for title in stackoverflow_df['title']]


X = np.concatenate([dialogue_df['text'].values,
                    stackoverflow_df['title'].values])
y = ['dialogue'] * dialogue_df.shape[0] + \
    ['stackoverflow'] * stackoverflow_df.shape[0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=0)
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))

X_train_tfidf, X_test_tfidf = tfidf_features(
    X_train, X_test, RESOURCE_PATH['TFIDF_VECTORIZER'])


intent_recognizer = LogisticRegression(
    penalty='l2', C=10, random_state=0).fit(X_train_tfidf, y_train)

y_test_pred = intent_recognizer.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test accuracy = {}'.format(test_accuracy))

pickle.dump(intent_recognizer, open(RESOURCE_PATH['INTENT_RECOGNIZER'], 'wb'))

X = stackoverflow_df['title'].values
y = stackoverflow_df['tag'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))

vectorizer = pickle.load(open(RESOURCE_PATH['TFIDF_VECTORIZER'], 'rb'))

X_train_tfidf, X_test_tfidf = vectorizer.transform(
    X_train), vectorizer.transform(X_test)


tag_classifier = OneVsRestClassifier(
    estimator=LogisticRegression(penalty='l2', C=5, random_state=0))
tag_classifier.fit(X_train_tfidf, y_train)

y_test_pred = tag_classifier.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test accuracy = {}'.format(test_accuracy))

pickle.dump(tag_classifier, open(RESOURCE_PATH['TAG_CLASSIFIER'], 'wb'))

embeddings = en_core_web_lg.load()

posts_df = pd.read_csv('data/tagged_posts.tsv', sep='\t')

counts_by_tag = posts_df.groupby('tag')['tag'].count()

os.makedirs(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], exist_ok=True)

embeddings_dim = len(embeddings.vocab['dimension'].vector)
for tag, count in counts_by_tag.items():
    tag_posts = posts_df[posts_df['tag'] == tag]

    tag_post_ids = tag_posts['post_id'].tolist()

    tag_vectors = np.zeros((count, embeddings_dim), dtype=np.float32)
    for i, title in enumerate(tag_posts['title']):
        tag_vectors[i, :] = question_to_vec(
            question=title, embeddings=embeddings, dim=embeddings_dim)

    filename = os.path.join(
        RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], os.path.normpath('%s.pkl' % tag))
    pickle.dump((tag_post_ids, tag_vectors), open(filename, 'wb'))


# dialogue_manager = DialogueManager(RESOURCE_PATH)
# dialogue_manager.generate_answer("age in c#")
# print(response)
# dialogue_manager.generate_answer("how are you")
# print(response)
# dialogue_manager.generate_answer("how old are you?")
# print(response)
# dialogue_manager.generate_answer("quick sort for array ")
# print(response)
# dialogue_manager.generate_answer("run cpp code in python")
# print(response)

