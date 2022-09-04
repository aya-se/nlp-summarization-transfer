NAME=bart-large-qmsum/checkpoint-1257
SPLIT=test

python -u train.py \
  --test_file data/qmsum/${SPLIT}.json \
  --do_predict \
  --model_name_or_path output/${NAME} \
  --output_dir output/${NAME}/prediction_logs_${SPLIT} \
  --prediction_path output/${NAME}/predictions.${SPLIT} \
  --max_source_length 512 \
  --generation_max_len 256 \
  --val_max_target_length 256 \
  --overwrite_output_dir \
  --per_device_eval_batch_size 1 \
  --predict_with_generate