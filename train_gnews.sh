# Please go to https://github.com/ParticleMedia/SimCSE.git and check out dev branch
# the source code entry point is SimCSE/simcse/trainers.py

#!/bin/bash
set -x #echo on
set -e #stop when any command fails

src_path="/home/services/chengniu/SimCSE"
data_path="/mnt/nlp/search/relevance"

export PYTHONPATH=$PYTHONPATH:$src_path

export CUDA_VISIBLE_DEVICES="0"

python $src_path/train.py \
    --model_name_or_path bert-base-uncased \
    --train_file $data_path/simcse_gnews_title.tsv.train.sf.csv \
    --output_dir $data_path/simcse-gnews_title-bert-base-uncased \
    --num_train_epochs 6 \
    --per_device_train_batch_size 128 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end false \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir true \
    --temp 0.05 \
    --do_train true \
    --do_eval false \
    --fp16 false \
