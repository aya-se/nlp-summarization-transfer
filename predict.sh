NAME=bart-large-qmsum/checkpoint-
SPLIT=test
OUTPUT_DIR=output/${NAME}
python -u train.py \
  --test_file data/eli5/${SPLIT}.jsonl \
  --do_predict \
  --model_name_or_path $OUTPUT_DIR \
  --output_dir ${OUTPUT_DIR}/predition_logs_${SPLIT} \
  --prediction_path ${OUTPUT_DIR}/predictions.${SPLIT} \
  --max_source_length 512 \
  --generation_max_len 256 \
  --val_max_target_length 256 \
  --overwrite_output_dir \
  --per_device_eval_batch_size 1 \
  --predict_with_generate