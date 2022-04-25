rm -rf /opt/ml/model/
mkdir -p /opt/ml/input/config/
cp resourceconfig.json /opt/ml/input/config/
pip install textract-trp
export TOKENIZERS_PARALLELISM=false
export SM_OUTPUT_INTERMEDIATE_DIR=/opt/ml/output/intermediate
export SM_CHANNEL_TEXTRACT=/textract/data/textracted/
export SM_CHANNEL_TRAIN=/textract/data/train
export SM_CHANNEL_VALIDATION=/textract/data/val
export GPU_NUM_DEVICES=1
export SM_NUM_GPUS=1
/opt/conda/bin/python3.8 smtc_launcher.py --learning_rate 5e-05 --metric_for_best_model eval_loss --model_name_or_path microsoft/layoutlm-base-uncased --num_train_epochs 25 --per_device_eval_batch_size 14 --per_device_train_batch_size 14 --save_total_limit 10 --seed 42 --task_name mlm --textract_prefix textract-transformers/data/textracted --training_script train.py --warmup_steps 200 --disable_tqdm false --output_dir /opt/ml/model --overwrite_output_dir True --fp16 True --save_strategy no --evaluation_strategy no
#CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python3.8 train.py --early_stopping_patience 10 --greater_is_better false --learning_rate 5e-05 --metric_for_best_model eval_loss --model_name_or_path microsoft/layoutlm-base-uncased --num_train_epochs 25 --per_device_eval_batch_size 16 --per_device_train_batch_size 16 --save_total_limit 10 --seed 42 --task_name mlm --textract_prefix textract-transformers/data/textracted --warmup_steps 200 --disable_tqdm false --output_dir /opt/ml/model --overwrite_output_dir True
