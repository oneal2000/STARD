TOKENIZER=YOUR_PATH_TO_MODEL
TOKENIZER_ID=roberta

cd ../data/example

python ../../src/build_train.py --tokenizer_name $TOKENIZER --negative_file train.negatives.tsv --qrels qrels.train.tsv \
  --queries train.query.txt --collection corpus.tsv --save_to ${TOKENIZER_ID}/train
python ../../src/tokenize_queries.py --tokenizer_name $TOKENIZER --query_file dev.query.txt --save_to $TOKENIZER_ID/query/dev.query.json
python ../../src/tokenize_passages.py --tokenizer_name $TOKENIZER --file corpus.tsv --save_to $TOKENIZER_ID/corpus

python ../../src/clean_json.py

cd $TOKENIZER_ID
mkdir models
mkdir encoding
mkdir ranking
cd ranking
mkdir intermediate
