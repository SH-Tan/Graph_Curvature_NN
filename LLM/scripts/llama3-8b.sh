#!/bin/bash

# Set common variables
model="meta-llama/Meta-Llama-3-8B"
sparsity_ratio=0.5
nsamples=5
alpha=0.9
curvature_save_dir="curv_pkl"
model_device="cuda:0"
compute_device="cuda:1"
seq_len=512
sample_edge_num=1
# cuda_device=0
cuda_device=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    python llm_main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $2 \
    --sparsity_type $3 \
    --save $4 \
    --nsamples $nsamples \
    --model_device $model_device \
    --compute_device $compute_device \
    --save_curvature_dir $curvature_save_dir \
    --alpha $alpha \
    --calib_data $5 \
    --sample_edge_num $sample_edge_num \
    --seqlen $seq_len
}

# llama-7b with magnitude pruning method
echo "Running with graph curvature pruning method"
# run_python_command "curvature" 0 "unstructured" "out/llama_8b/unstructured/curvature/" "c4_independent" 1
run_python_command "curvature" 0.5 "unstructured" "out/llama_8b/unstructured/curvature/" "c4_independent" 1
run_python_command "curvature" 0.5 "unstructured" "out/llama_8b/unstructured/curvature/" "c4_dependent" 1
echo "Finished graph curvature pruning method"


# # llama-7b with wanda pruning method
# echo "Running with wanda pruning method"
# run_python_command "wanda" "unstructured" "out/llama_8b/unstructured/wanda/"
# run_python_command "wanda" "2:4" "out/llama_8b/2-4/wanda/"
# run_python_command "wanda" "4:8" "out/llama_8b/4-8/wanda/"
# echo "Finished wanda pruning method"


# # llama-7b with magnitude pruning method
# echo "Running with magnitude pruning method"
# run_python_command "magnitude" "unstructured" "out/llama_8b/unstructured/magnitude/"
# run_python_command "magnitude" "2:4" "out/llama_8b/2-4/magnitude/"
# run_python_command "magnitude" "4:8" "out/llama_8b/4-8/magnitude/"
# echo "Finished magnitude pruning method"
