import numpy as np
from collections import Counter
import math
 
class BM25(object):
    def __init__(self, docs, k1=1.3, k2=1.5, b=0.75,topk=10):
        self.docs = docs
        self.Numdocs = len(docs)
        self.avg_doclen = sum([len(doc) for doc in docs.values()]) / self.Numdocs
        self.tf = self.calculate_tf()
        self.idf = self.calculate_idf()
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.topk=topk
 
    def calculate_idf(self):
        idf = {}
        for doc in self.docs.values():
            for word in set(doc):
                idf[word] = idf.get(word, 0) + 1
        for word, freq in idf.items():
            idf[word] = math.log((self.Numdocs - freq + 0.5) / (freq + 0.5))
        return idf

    def calculate_tf(self):
        tf = {}
        for id,doc in self.docs.items():
            temp = {}
            for word in doc:
                temp[word] = temp.get(word, 0) + 1
            tf[id]=temp
        return tf

    def get_score(self, index, query):
        score = 0.0
        doclen = len(self.docs[index])
        qf = Counter(query)
        for q in query:
            if q not in self.tf[index]:
                continue
            score += self.idf[q] * (self.tf[index][q] * (self.k1 + 1) / (
                        self.tf[index][q] + self.k1 * (1 - self.b + self.b * doclen / self.avg_doclen))) * (
                                 qf[q] * (self.k2 + 1) / (qf[q] + self.k2))
        return score
 
    def query(self, query):
        score_list = []
        for index in self.docs.keys():
            score_list.append((index,self.get_score(index, query)))
        score_list.sort(key=lambda x:(-x[1]))

        return [item[0] for item in score_list[:self.topk]]