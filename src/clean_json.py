import json 

docs = []

dir = f'roberta/corpus/split09.json'

with open(dir,encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        doc = json.loads(line)
        if doc['text']!=[]:
            docs.append(doc)

with open(dir,'w',encoding='utf-8') as f:
    for doc in docs:
        json.dump(doc, f, ensure_ascii=False)
        f.write('\n')