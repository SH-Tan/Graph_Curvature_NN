import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import numpy as np
import random
import os
import argparse
from prune import prune_wanda, prune_magnitude, check_sparsity, find_layers, prune_curvature
from eval import eval_ppl, eval_zero_shot

from huggingface_hub import login


def enable_hf_offline_mode():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def safe_hf_login(token):
    try:
        login(token)
        print("Hugging Face login succeeded")
    except Exception as exc:
        enable_hf_offline_mode()
        print(f"Hugging Face login skipped: {exc}")


def _is_network_error(exc):
    msg = str(exc).lower()
    return (
        "name resolution" in msg
        or "connecterror" in msg
        or "connection error" in msg
        or "temporary failure" in msg
        or "offline" in msg
    )


safe_hf_login()

print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights", device = "cpu"):
    print("Loading model:", model_name)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map=device
        )
    except Exception as exc:
        if not _is_network_error(exc):
            raise
        enable_hf_offline_mode()
        print(f"Falling back to local cached model files: {exc}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map=device,
            local_files_only=True,
        )
    
    if hasattr(model, 'hf_device_map'):
        print('hf_device_map = ', model.hf_device_map)
    else:
        # The device map is handled by accelerate under the hood
        print("Model loaded with device_map, but hf_device_map not directly accessible")

    model.seqlen = 8 # model.config.max_position_embeddings
    # print(model.seqlen)
    return model



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=13, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["curvature", "wanda", "magnitude"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--model_device', type=str, default="cuda:0", help='Device for model load.')
    parser.add_argument('--compute_device', type=str, default="cuda:1", help='Device for curvature computing.')
    parser.add_argument('--alpha', type=float, default=0., required=False, help='Alpha used for distribution')
    parser.add_argument('--save_curvature_dir',type=str,default=None,help='Directory to save per-layer curvature pkl files.')
    parser.add_argument('--load_curvature_dir',type=str,default=None,help='Directory to load previously saved per-layer curvature pkl files.')

    parser.add_argument("--eval_zero_shot", type=int, default=0, help='evaluate on downsteam zero shot tasks')
    args = parser.parse_args()
    
    # set random seed
    seed = args.seed
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if torch.cuda.is_available():
        model_device = args.model_device
        compute_device = args.compute_device
    else:
        model_device = "cpu"
        compute_device = "cpu"
        
    print(f"Using {model_device} model device, {compute_device} for computing")

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
    
    # load model
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir, model_device)
    model.eval()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    except Exception as exc:
        if not _is_network_error(exc):
            raise
        enable_hf_offline_mode()
        print(f"Falling back to local cached tokenizer files: {exc}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, local_files_only=True)
    
    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "curvature":
            prune_curvature(args, model, tokenizer, compute_device, prune_n, prune_m)
        
        
    # ################################################################
    # print("*"*30)
    # sparsity_ratio = check_sparsity(model)
    # print(f"sparsity sanity check {sparsity_ratio:.4f}")
    # print("*"*30)
    # ################################################################
    # ppl_test = eval_ppl(args, model, tokenizer, compute_device)
    # print(f"wikitext perplexity {ppl_test}")

    # if not os.path.exists(args.save):
    #     os.makedirs(args.save)
    # save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    # with open(save_filepath, "a+") as f:
    #     print("method\tactual_sparsity\tppl_test", file=f, flush=True)
    #     print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    # if args.eval_zero_shot:
    #     accelerate=False
    #     if "30b" in args.model or "65b" in args.model or "70b" in args.model:
    #         accelerate=True

    #     task_list=["hellaswag","winogrande","openbookqa","arc_easy"]
    #     num_shot = 0
    #     results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate, "cuda:0")
    #     with open(save_filepath, "a+") as f:
    #         print("\n********************************\n", file=f, flush=True)
    #         print("\nzero_shot evaluation results\n\n", file=f, flush=True)
    #         print(results, file=f, flush=True)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()
