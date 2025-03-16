CUDA_VISIBLE_DEVICES=0 python3 main.py --gin_config_file=configs/ml-20m/hstu-sampled-softmax-n128-large-final-l1000-e21.gin --master_port=12345

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMEXPR_MAX_THREADS=64
