from config import *


class Dictionary:
    def __init__(self):
        self.word2index = {}
        self.index2word = {UNK_token_id: 'UNK', PAD_token_id: "PAD",
                           SOS_token_id: "SOS", EOS_token_id: "EOS", }
        self.n_words = len(self.index2word)
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    def index_words(self, words):
        if isinstance(words, str):
            for word in words.split():
                self.index_word(word)
        elif isinstance(words, list):
            for word in words:
                self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
