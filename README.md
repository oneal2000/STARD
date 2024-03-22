# STARD
Welcome to StaRD(the STAtute Retrieval Dataset).This is an innovative dataset derived from real-world legal consultation inquiries made by the general public. STARD addresses a notable void in the current landscape of law retrieval datasets, which primarily focus on professional legal queries, thereby neglecting the complexity and variety inherent in layperson inquiries.

## Overview

StaRD is a dataset derived from real-world legal consultation inquiries.The whole dataset consists of 1543 queries focus on professional legal queries.

In the dataset,we provide the origin corpus and queries files,you could find them in

```
data/corpus.jsonl
data/queries.json
```

The corpus consists of all the latest laws of China,with a total number of over 50 thousands statutes.In `data/corpus.jsonl`,each line containing a dict describes a statute,the dict have three fields:`id` , `name` and `content` 

An example is given as below

```json
{"id": 22648, "name": "农村土地承包经营纠纷调解仲裁法第十八条", "content": "农村土地承包经营纠纷申请仲裁的时效期间为二年，自当事人知道或者应当知道其权利被侵害之日起计算。\\n"}
```

A query example  in `queries.json` is as below

```json
{
        "query_id": 1542,
        "问题": "企业是否必须要为从事危险作业的职工缴纳工伤保险费？企业为职工投保意外伤害险，能否免除缴纳工伤保险的义务？",
        "相关法规": {
            "建筑法第四十八条": "建筑施工企业应当依法为职工参加工伤保险缴纳工伤保险费。鼓励企业为从事危险作业的职工办理意外伤害保险，支付保险费。\n",
            "保险法第三十九条": "人身保险的受益人由被保险人或者投保人指定。\n\n投保人指定受益人时须经被保险人同意。投保人为与其有劳动关系的劳动者投保人身保险，不得指定被保险人及其近亲属以外的人为受益人。\n\n被保险人为无民事行为能力人或者限制民事行为能力人的，可以由其监护人指定受益人。\n"
        },
        "match_id": [
            11543,
            16411
        ],
        "match_name": [
            "建筑法第四十八条",
            "保险法第三十九条"
        ]
    }
```

Each query contains the above five fields, `问题` field describes the question itself . `match_id` and `match_name` give the `id`  and `name` fields refer to  `corpus.jsonl` .They are relative statutes.

## Install environment

### Requirements

```
dense==0.0.1

jieba==0.42.1
numpy==1.23.3
pandas==1.5.0
torch==1.8.0
tqdm==4.49.0
transformers==4.2.0
```

 Except **Dense**,the other packages will be installed through the following instructions. In order to ensure that the version of `torch` is correct, you may want to follow the instructions of [official documentation](https://pytorch.org/get-started/locally/) for installation.

```bash
conda create -n stard python=3.9
conda activate stard
pip install torch==1.8.0
pip install -r requirements.txt
```

The package **Dense** is a dense retriever toolkit,which should be downloaded seperately. [Click here](https://github.com/luyug/Dense) for the repo of Dense, follow the repo to install **Dense**.

## Evaluation

We use several different retrieval algorithms to test our dataset,including `BM25`, `Query Likelihood` and `Dense Retriever`. To give a consistent test, we process the origin queries and split the dataset into two parts ,`train` and `dev` ,all relative files are in `data/example` 

```
dev.query.txt         # Query content in dev
qrels.dev.tsv         # Relateive statute ids in dev
qrels.train.tsv       # Relateive statute ids in train
train.negatives.tsv   # Negative statutes corresponding to train
train.query.txt       # Query content in train
```

The `dev` and `train` are randomly split with a 1:4 ratio and the negatives are generated randomly from the whole corpus.

### BM25

We implement `BM25` and the source code is in `src/BM25/BM25.py`,we also give an example to test the above `dev` set.

```bash
cd src
cd BM25
python test.py
```

 ### Query Likelihood

We also implement `Query Likelihood` algorithm and the source code is in `src/QLD/test_qld.py`.

```bash
cd src
cd QLD
python test_qld.py
```

### Dense Retriever 

We use the  **Dense** toolkit to implement dense retrieval. We use `Chinese-Roberta-wwm` as the base model and use `train` set with negatives to do contrastive learning to fine-tune the model.

Begin with the raw data in `data/example` , to run the example, you may want to first download the model from  huggingface and run the following bash scripts.

You need to modify the relative path to be your model path in the bash files before you run it.  

```bash
cd src
bash getData.sh
bash train.sh
bash test.sh
```

After running,the result is stored in `data/example/$TOKENZIER_ID/ranking/rank.txt.macro` ,each line in the file corresponds to a result,that is 

```
[query_id] [match_id] [rank]
```

## RAG in LLM

Our dataset provides professional legal questions. Fine tuning with it will enhance the ability of retrieval model.

In our experiments, we use Chinese-Roberta-wwm as bese model,do MLM training with our corpus and successively fine tune the model with GPT generated data and our dataset.

As for GPT generated queries,we ask GPT to make quesitons toward certain statute and then we could generate `Query-Statute` pairs to fine tune our model.

Then use the fine-tuned model to do RAG. Take Jec-QA as an example,first do retrieval with the question and its options,then we find the relative statutes. Involve the statutes in the prompt then we finish the RAG.

 One of prompt format is as below

```
请根据以下法条回答问题，下面给出法条:
   【法条名1：法条内容】
  ......
   【法条名10：法条内容】
请根据上面的法条和你的知识回答问题,优先参考法条，不要解释原因。你的答案只需包含选项的序号,请注意，下面给出问题
   【问题题面】
A. 【选项内容】
  ...
  ...   
 请给出答案（只包含选项字母，不要有汉字）：
```

