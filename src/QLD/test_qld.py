import jieba
import numpy as np
import json
from tqdm import tqdm 
corpus_prob_dict ={}


def calcDocP(tokens):
    doc_len = len(tokens)
    freq = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1
    prob = {}
    for word in freq.keys():
        prob[word] = freq[word] / doc_len
    return prob

def calcCorpusP(corpus):
    tokens = []
    for doc in corpus:
        tokens.extend(doc["content"])
    corpus_len = len(tokens)
    freq = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1
    prob = {}
    for word in freq.keys():
        prob[word] = freq[word] / corpus_len
    return prob

def querylikelihood(doc, query_tokens,lam):
    doc_prob_dict = calcDocP(doc["content"])
    log_ql = 1
    for word in query_tokens:
        doc_prob = doc_prob_dict.get(word, 0)
        corpus_prob = corpus_prob_dict.get(word, 0)
        # Jelinek-Mercer Smoothing
        smooth_prob = lam * doc_prob + (1 - lam) * corpus_prob
        log_prob = np.log(smooth_prob+1e-100)
        log_ql += log_prob
        
    return log_ql

def topk_docs(docs, query, lam, k):
    ql_list = []
    
    for doc in docs:
        ql = querylikelihood(doc, query, lam)
        ql_list.append((doc["id"], ql))
    ql_list.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in ql_list[:k]]


stopwords = ['\n','\\','n',' ','.','（','）']

with open('../../data/example/cn_stopwords.txt',encoding='UTF-8') as f:
    lines=f.readlines()
    for line in lines:
        stopwords.append(line.strip())

testids=[]

with open('../../data/example/dev.query.txt', "r", encoding="utf-8") as file:
    for line in file:
        qid,q = line.split('\t')
        testids.append(int(qid))


with open('../../data/queries.json', "r", encoding="utf-8") as file:
    data = json.load(file)

docs=[]    

with open('../../data/corpus.jsonl',encoding='UTF-8') as f:
    for line in f:
        doc = json.loads(line)
        words = list(jieba.cut(doc['content']))
        words = [word for word in words if word not in stopwords]
        
        new_doc = {}
        new_doc['id']=doc['id']
        new_doc['content']=words
        docs.append(new_doc)

corpus_prob_dict = calcCorpusP(docs)

lam = 0.9

tot = 0.0
one_recall_hits = 0.0
recall_hits = 0.0
mrr_sum = 0.0

topk = 3

for obj in tqdm(data):
    if(type(obj['问题'])==float):
        continue
    if(obj['query_id'] not in testids):
            continue
    tot += 1
    q = list(jieba.cut(obj['问题']))
    q = [word for word in q if word not in stopwords]
    ans = topk_docs(docs, q, lam, topk)
    
    for rank,match in enumerate(ans, 1):
        if match in obj['match_id']:
            one_recall_hits += 1
            mrr_sum += 1.0/rank
            break
    cnt = 0
    for match in obj['match_id']:
        if match in ans:
            cnt += 1
    recall_hits += cnt/len(obj['match_id'])

print(f"topk:{topk}")
print(f'one_recall:{one_recall_hits/tot}')
print(f'Recall:{recall_hits/tot}')

print(f'mrr:{mrr_sum/tot}')
        