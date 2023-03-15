set -e
set -x

#/home/services/chengniu/SimCSE 0 /mnt/nlp/search/relevance/simcse-gnews-bert-base-uncased simcse-gnew
#/home/services/chengniu/SimCSE 0 /mnt/nlp/search/relevance/simcse-gnews_title-bert-base-uncased simcse-gnews_title
src=$1
gpu=$2
model=$3
redis_q=$4


export PYTHONPATH=$PYTHONPATH:$src
export CUDA_VISIBLE_DEVICES="$gpu"

python ./SimCseRedis.py \
    --model_name_or_path $model \
    --per_device_eval_batch_size 128 \
    --max_seq_length 32 \
    --pooler_type cls \
    --fp16 false \
    --train_file $data_path/simcse_gnews_title.tsv.train.sf.csv \
    --output_dir $data_path/simcse-gnews_title-bert-base-uncased \
    --num_train_epochs 6 \
    --per_device_train_batch_size 128 \
    --learning_rate 5e-5 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end false \
    --eval_steps 125 \
    --overwrite_output_dir true \
    --temp 0.05 \
    --do_train true \
    --do_eval false \
    --redis_q $redis_q \
