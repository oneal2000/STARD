TOKENIZER=YOUR_PATH_TO_MODEL
TOKENIZER_ID=roberta
MODEL_PATH=YOUR_PATH_TO_MODEL

python -m dense.driver.train --output_dir ../data/example/$TOKENIZER_ID/models \
--model_name_or_path $MODEL_PATH \
--tokenizer_name $TOKENIZER  \
--save_steps 1000 \
--train_dir ../data/example/$TOKENIZER_ID/train \
--fp16 \
--per_device_train_batch_size 4 \
--learning_rate 5e-6 \
--num_train_epochs 10 \
--dataloader_num_workers 2