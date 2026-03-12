import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

model_name = "mistralai/Mistral-7B-v0.1"
# alternatives:
# model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "Qwen/Qwen2-7B"

print("Loading model:", model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("\n===== MODEL ARCHITECTURE =====\n")
print(model)

print("\n===== PARAMETER SUMMARY =====\n")

total_params = 0
layer_params = defaultdict(int)

for name, param in model.named_parameters():
    num = param.numel()
    total_params += num

    # group by top-level module
    layer_name = name.split('.')[0]
    layer_params[layer_name] += num

    print(f"{name:60} {num/1e6:8.2f} M")

print("\n===== PARAMS PER TOP MODULE =====")

for k,v in layer_params.items():
    print(f"{k:20} {v/1e6:.2f} M")

print("\nTOTAL PARAMETERS:", total_params/1e9, "B")