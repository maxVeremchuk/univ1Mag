from Pyro4 import expose
import time
#import math

class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers

    def solve(self):
        corpus = self.read_input()
        size = len(corpus)
        step = size / len(self.workers)
        start_time = time.time()
        # map
        mapped = []
        for i in xrange(0, len(self.workers)):
            mapped.append(self.workers[i].mymap(i * step, i * step + step, corpus))
            #mapped.append(self.mymap(i * step, i * step + step, corpus))

        tf_idf_list = self.myreduce(self, mapped, corpus)
        final_time = (time.time() - start_time)

        self.write_output(tf_idf_list, final_time)

    @staticmethod
    @expose
    def mymap(a, b, corpus):
        corpus_list = []
        if (b + 1 == len(corpus)):
            b += 1
        for text in corpus[a:b]:
            term_dict = {}
            words = text.decode('utf-8').strip().lower().split(' ')
            for word in words:
                #print(word)
                if (word not in term_dict.keys()):
                    term_dict[word] = 1.0 / len(list(dict.fromkeys(words)))
                else:
                    term_dict[word] += 1.0 / len(list(dict.fromkeys(words)))
            corpus_list.append(term_dict)
        return corpus_list

    @staticmethod
    @expose
    def myreduce(self, mapped, corpus):
        tf_idf_list = []
        for x in mapped:
            corpus_list = x.value
            for term_dict in corpus_list:
                tf_idf_text_list = []
                for word in term_dict.keys():
                    tf_idf = term_dict[word] * self.compute_idf(word, corpus)
                    tf_idf_text_list.append(str(word) + ":" + str(tf_idf))
                tf_idf_list.append(tf_idf_text_list)
        print(tf_idf_list)
        return tf_idf_list

    def compute_idf(self, word, corpus):
        return (len(corpus)/sum([1.0 for i in corpus if word in i.lower()]))

    def read_input(self):
        f = open(self.input_file_name, 'r')
        lines = f.readlines()
        f.close()
        return lines

    def write_output(self, output, final_time):
        f = open(self.output_file_name, 'w')
        for out in output:
            f.write(str(out))
            f.write('\n')
        f.write('\n')
        f.write(str(final_time))
        f.write('\n')
        f.close()
        print("output done")

# s = Solver(["1", "2"], "input.txt", "output.txt")
# s.solve()

















#local time: 0.138884067535
#parcs time 3 workers: 0.0205411911011
