export REFQA_DATA_DIR=/data/nyx/qa/uqa/baselines/RefQA/data
# export PYTORCH_PRETRAINED_BERT_CACHE=/root/pretrained_weights
export OUTPUT_DIR=/data/nyx/qa/uqa/baselines/RefQA/output
export CUDA_VISIBLE_DEVICES=3,4,6

cd ../ 
 
python -m torch.distributed.launch --nproc_per_node=3 run_squad.py \
        --model_type bert \
        --model_name_or_path bert-large-uncased-whole-word-masking \
        --do_train \
        --do_eval \
        --do_lower_case \
        --train_file $REFQA_DATA_DIR/uqa_train_main.json \
        --predict_file $REFQA_DATA_DIR/dev-v1.1.json \
        --learning_rate 3e-5 \
        --num_train_epochs 2 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --output_dir $OUTPUT_DIR \
        --per_gpu_train_batch_size=2 \
        --per_gpu_eval_batch_size=2 \
        --seed 42 \
        --fp16 \
        --overwrite_output_dir \
        --gradient_accumulation_steps 4\
        --logging_steps 1000 \
        --save_steps 1000 ;
