TOKENIZER=YOUR_PATH_TO_MODEL
TOKENIZER_ID=roberta




for i in $(seq -f "%02g" 0 9)
do
python -m dense.driver.encode \
  --output_dir ../data/example/$TOKENIZER_ID/models \
  --model_name_or_path ../data/example/$TOKENIZER_ID/models \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --encode_in_path ../data/example/$TOKENIZER_ID/corpus/split${i}.json \
  --tokenizer_name $TOKENIZER  \
  --encoded_save_path ../data/example/$TOKENIZER_ID/encoding/split${i}.pt
done


python -m dense.driver.encode \
  --output_dir ../data/example/$TOKENIZER_ID/models \
  --model_name_or_path ../data/example/$TOKENIZER_ID/models \
  --fp16 \
  --tokenizer_name $TOKENIZER \
  --q_max_len 32 \
  --encode_is_qry \
  --per_device_eval_batch_size 128 \
  --encode_in_path ../data/example/$TOKENIZER_ID/query/dev.query.json \
  --encoded_save_path ../data/example/$TOKENIZER_ID/encoding/qry.pt

for i in $(seq -f "%02g" 0 9)
do
python -m dense.faiss_retriever \
  --query_reps ../data/example/$TOKENIZER_ID/encoding/qry.pt \
  --passage_reps ../data/example/$TOKENIZER_ID/encoding/split${i}.pt \
  --depth 50 \
  --save_ranking_to ../data/example/$TOKENIZER_ID/ranking/intermediate/split${i}
done

python -m dense.faiss_retriever.reducer \
  --score_dir ../data/example/$TOKENIZER_ID/ranking/intermediate \
  --query ../data/example/$TOKENIZER_ID/encoding/qry.pt \
  --save_ranking_to ../data/example/$TOKENIZER_ID/ranking/rank.txt

python score_to_marco.py ../data/example/$TOKENIZER_ID/ranking/rank.txt

