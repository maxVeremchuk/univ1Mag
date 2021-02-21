import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam

from Pyro4 import expose


class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers
        print("Inited")

    def solve(self):
        print("Job Started")
        print("Workers %d" % len(self.workers))

        nltk.download('treebank')
        tagged_sentences = nltk.corpus.treebank.tagged_sents()

        sentences, sentence_tags = [], []
        for tagged_sentence in tagged_sentences:
            sentence, tags = zip(*tagged_sentence)
            sentences.append(np.array(sentence))
            sentence_tags.append(np.array(tags))

        (train_sentences,
         test_sentences,
         train_tags,
         test_tags) = train_test_split(sentences, sentence_tags, test_size=0.2)

        words, tags = set([]), set([])

        for s in train_sentences:
            for w in s:
                words.add(w.lower())

        for ts in train_tags:
            for t in ts:
                tags.add(t)

        self.word2index = {w: i + 2 for i, w in enumerate(list(words))}
        self.word2index['-PAD-'] = 0  # The special value used for padding
        self.word2index['-OOV-'] = 1  # The special value used for OOVs

        self.tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
        self.tag2index['-PAD-'] = 0

        train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []

        for s in train_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(self.word2index[w.lower()])
                except KeyError:
                    s_int.append(self.word2index['-OOV-'])

            train_sentences_X.append(s_int)

        for s in test_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(self.word2index[w.lower()])
                except KeyError:
                    s_int.append(self.word2index['-OOV-'])

            test_sentences_X.append(s_int)

        for s in train_tags:
            train_tags_y.append([self.tag2index[t] for t in s])

        for s in test_tags:
            try:
                test_tags_y.append([self.tag2index[t] for t in s])
            except KeyError:
                test_tags_y.append(self.word2index['-PAD-'])

        MAX_LENGTH = len(max(train_sentences_X, key=len))
        train_sentences_X = pad_sequences(
            train_sentences_X, maxlen=MAX_LENGTH, padding='post')
        test_sentences_X = pad_sequences(
            test_sentences_X, maxlen=MAX_LENGTH, padding='post')
        train_tags_y = pad_sequences(
            train_tags_y, maxlen=MAX_LENGTH, padding='post')
        test_tags_y = pad_sequences(
            test_tags_y, maxlen=MAX_LENGTH, padding='post')

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(MAX_LENGTH, )))
        self.model.add(Embedding(len(self.word2index), 128))
        self.model.add(Bidirectional(LSTM(256, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(len(self.tag2index))))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(0.001),
                           metrics=['accuracy'])

        n = self.read_input()
        step = n / len(self.workers)

        # map
        mapped = []
        for i in xrange(0, len(self.workers)):
            mapped.append(self.workers[i].mymap(i * step, i * step + step))

        print("Map finished: " + str(mapped))

        # reduce
        reduced = self.myreduce(mapped)
        print("Reduce finished: " + str(reduced))

        # output
        self.write_output(reduced)

        print("Job Finished")

    @staticmethod
    @expose
    def mymap(a, b):
        print(a, b)

        test_samples = []
        for i in xrange(a, b):
            test_samples.append(i.split())

        test_samples_X = []
        for s in test_samples:
            s_int = []
            for w in s:
                try:
                    s_int.append(self.word2index[w.lower()])
                except KeyError:
                    s_int.append(self.word2index['-OOV-'])
            test_samples_X.append(s_int)

        test_samples_X = pad_sequences(
            test_samples_X, maxlen=MAX_LENGTH, padding='post')

        predictions = model.predict(test_samples_X)
        print(self.logits_to_tokens(predictions, {
              i: t for t, i in self.tag2index.items()}))

        return self.logits_to_tokens(predictions, {i: t for t, i in self.tag2index.items()})

    @staticmethod
    @expose
    def myreduce(mapped):
        output = []
        for x in mapped:
            output.append(x.value)
        return output

    def read_input(self):
        f = open(self.input_file_name, 'r')
        line = f.readline()
        f.close()
        return int(line)

    def write_output(self, output):
        f = open(self.output_file_name, 'w')
        f.write(str(output))
        f.write('\n')
        f.close()

    def to_categorical(self, sequences, categories):
        cat_sequences = []
        for s in sequences:
            cats = []
            for item in s:
                cats.append(np.zeros(categories))
                cats[-1][item] = 1.0
            cat_sequences.append(cats)
        return np.array(cat_sequences)

    def logits_to_tokens(self, sequences, index):
        token_sequences = []
        for categorical_sequence in sequences:
            token_sequence = []
            for categorical in categorical_sequence:
                token_sequence.append(index[np.argmax(categorical)])

            token_sequences.append(token_sequence)

        return token_sequences
