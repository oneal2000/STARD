import jieba
import json
from BM25 import BM25
from tqdm import tqdm
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

docs={}    

with open('../../data/corpus.jsonl',encoding='UTF-8') as f:
    for line in f:
        doc = json.loads(line)
        words = list(jieba.cut(doc['content']))
        words = [word for word in words if word not in stopwords]
        docs[doc['name']]=words
    
# Give your topk number here    
topk=3

model = BM25(docs,topk=topk)


tot = 0.0
recall_hits = 0.0
one_recall_hits = 0.0
mrr_sum = 0.0
for obj in tqdm(data):
    if(type(obj['问题'])==float):
        continue
    if(obj['query_id'] not in testids):
            continue
    tot += 1
    q = list(jieba.cut(obj['问题']))
    q = [word for word in q if word not in stopwords]
    ans = model.query(q)
    for rank,match in enumerate(ans, 1):
        if match in obj['match_name']:
            one_recall_hits += 1.0
            mrr_sum += 1.0/rank
            break
    cnt = 0
    for match in obj['match_name']:
        if match in ans:
            cnt += 1
    recall_hits += cnt/len(obj['match_name'])
print(f"topk:{topk}")
print(f'one_recall:{one_recall_hits/tot}')
print(f'Recall:{recall_hits/tot}')

print(f'mrr:{mrr_sum/tot}')
        