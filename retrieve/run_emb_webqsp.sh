dataset=webqsp
CUDA_LAUNCH_BLOCKING=1
export HF_ENDPOINT=https://hf-mirror.com
python emb.py -d $dataset --dataset-path ~/Documents/data --model-path ~/Documents/data/gte-large-en-v1.5
