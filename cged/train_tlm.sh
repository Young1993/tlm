cache_file=./cache/cged_data.pkl
learning_rate=5e-5
echo 'gpu:' $1

CUDA_VISIBLE_DEVICES=$1 python3 train.py --lr $learning_rate --cache-file $cache_file
