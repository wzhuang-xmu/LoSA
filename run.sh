export CUDA_VISIBLE_DEVICES=6

python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5