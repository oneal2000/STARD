# STARD: A Statute Retrieval Dataset for Layperson Queries


## Overview

Welcome to the official GitHub repository for STARD (STAtute Retrieval Dataset). STARD is derived from real-world legal consultation questions made by the general public. 

Unlike existing statute retrieval datasets that focus predominantly on professional legal queries, STARD captures the complexity and diversity of layperson queries. 

Through a comprehensive evaluation of various retrieval baselines, including conventional methods and those employing advanced techniques such as GPT-4, we reveal that existing retrieval approaches all fall short of achieving optimal results. 

Additionally, we show that employing STARD as a Retrieval-Augmented Generation (RAG) dataset markedly improves LLM's performance on legal tasks, which indicates that STARD is a pivotal resource for developing more accessible and effective legal systems.



## Installation Instructions

### Requirements

Before you begin, make sure you have the following packages installed in your environment:

```plaintext
jieba==0.42.1
numpy==1.23.3
pandas==1.5.0
torch==1.8.0
tqdm==4.49.0
transformers==4.2.0
```

### Setting Up Your Environment

To create a new environment and install the required packages except for **Dense**, follow these steps:

```bash
conda create -n stard python=3.9
conda activate stard
pip install -r requirements.txt
```

**Note:** The `requirements.txt` file should exclude`torch`. Install PyTorch specifically according to your system setup by following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

```bash
pip install torch==1.8.0
```

### Installing Dense

**Dense** is a specialized dense retriever toolkit that must be installed separately. Visit the [Dense repository](https://github.com/luyug/Dense) and follow the installation instructions provided there.





## Dataset Structure

### Directory Overview

The root directory for the STARD dataset is located at `/STARD/data`. The dataset comprises a total of 1,543 queries and a corresponding large-scale corpus of 55,348 candidate statutory articles.

### Data Files and Structure

**Queries:**
The queries and their relevant statutory articles are stored in the JSON file:
```
data/queries.json
```

**Example Query:**
Below is a sample entry from `queries.json`, showcasing the structure and data fields:
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

Translated:
{
    "query_id": 1542,
    "question": "Must enterprises pay work-related injury insurance for employees engaged in hazardous work? Can the obligation to pay work injury insurance be waived if the enterprise insures employees against accidental injuries?",
    "relevant_laws": {
        "Construction Law Article 48": "Construction enterprises must, in accordance with the law, pay work injury insurance for their employees. Enterprises are encouraged to handle accidental injury insurance for employees engaged in hazardous work and pay the insurance premiums.",
        "Insurance Law Article 39": "The beneficiary of personal insurance is designated by the insured or the policyholder. When the policyholder designates the beneficiary, the consent of the insured is required. If the policyholder insures a worker with whom they have a labor relationship, they may not designate anyone other than the insured and their immediate family as beneficiaries. If the insured is a person without civil conduct capacity or with limited civil conduct capacity, their guardian may designate the beneficiary."
    },
    "match_id": [11543, 16411],
    "match_name": ["Construction Law Article 48", "Insurance Law Article 39"]
}
```

**Corpus:**
The comprehensive corpus containing all candidate statutory articles is available in:
```
data/corpus.jsonl
```

**Example Article:**
Here is an example from `corpus.jsonl`, illustrating the format and content:

```json
{
	"id": 22648,
	"name": "农村土地承包经营纠纷调解仲裁法第十八条",
	"content": "农村土地承包经营纠纷申请仲裁的时效期间为二年，自当事人知道或者应当知道其权利被侵害之日起计算。\\n"
}

Translated:
{
    "id": 22648,
    "name": "Article 18 of the Law on Mediation and Arbitration of Disputes over Rural Land Contracting Management",
    "content": "The limitation period for applying for arbitration of disputes over rural land contracting management is two years, calculated from the date when the party becomes aware or should have become aware of the infringement of their rights."
}
```

### Data Collection Methodology

The corpus consists of national-level laws, regulations, and judicial interpretations from China. Our legal team manually curated and downloaded the latest versions from official government sources. Each document is divided into the smallest searchable units based on articles, facilitating detailed legal research and application.





## Evaluation

Our evaluation framework tests the dataset using multiple retrieval algorithms, including BM25, Query Likelihood, and Dense Retriever. We standardize our evaluation process by pre-processing the original queries and partitioning the dataset into two subsets: `train` and `dev`. All related files are located in `data/example`.

### Dataset Files

Here are the files associated with the development and training datasets:

```
dev.query.txt         # Queries for the development set
qrels.dev.tsv         # Relevant statute IDs for the development set
qrels.train.tsv       # Relevant statute IDs for the training set
train.negatives.tsv   # Randomly sampled non-relevant IDs for contrastive learning
train.query.txt       # Queries for the training set
```

The datasets `dev` and `train` are randomly split in a 1:4 ratio. Negative samples are also randomly generated from the entire corpus.

### BM25

We have implemented the BM25 algorithm. The source code is available at `src/BM25/BM25.py`. An example script to test the development set is provided:

```bash
cd src/BM25
python test.py
```

### Query Likelihood

The Query Likelihood algorithm is also implemented, with the source code located at `src/QLD/test_qld.py`.

```bash
cd src/QLD
python test_qld.py
```



### Dense Retriever

For dense retrieval, we use the Dense toolkit and the `Chinese-Roberta-wwm` model as the back-bone model. The model is fine-tuned on the training set using negative sampling for contrastive learning.

To run the examples, begin with the raw data in `data/example`. Ensure the model is downloaded from Hugging Face and modify the script paths to match your local setup:

```bash
cd src
bash getData.sh
bash train.sh
bash test.sh
```

Results are stored in `data/example/$TOKENIZER_ID/ranking/rank.txt.macro`, formatted as:

```
[query_id] [match_id] [rank]
```

