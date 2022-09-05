NAME=bart-large-eli5-to-qmsum/checkpoint-10056
SPLIT=test

python -u train.py \
  --test_file data/qmsum/${SPLIT}.json \
  --do_predict \
  --model_name_or_path output/${NAME} \
  --output_dir output/${NAME}/prediction_logs_${SPLIT} \
  --overwrite_output_dir \
  --per_device_eval_batch_size 1 \
  --max_source_length 1024 \
  --generation_max_len 256 \
  --val_max_target_length 256 \
  --predict_with_generate