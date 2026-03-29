import time 
import heapq 
import torch 
import torch.nn as nn 
from layerwrapper import WrappedGPT
from data import get_loaders 
from layerwrapper_curv import collect_layer_data, _make_lm_head_op
from cal_curvature import compute_op_curvature, build_layer_cache
from curv_prune_utils import _curvature_to_param_tensor


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 


def prepare_calibration_input(model, dataloader, device, nsamples):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    # ===== device handling =====
    if isinstance(device, str):
        device = torch.device(device)

    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]
        device = torch.device(f"cuda:{dev}") if isinstance(dev, int) else dev

    # ===== allocate =====
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device
    )

    cache = {
        "i": 0,
        "attention_mask": None,
        "position_ids": None
    }

    # ===== catcher =====
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            i = cache["i"]

            # ✅ stop if enough samples collected
            if i >= nsamples:
                raise StopIteration

            if inp.dim() == 3:
                if inp.shape[0] != 1:
                    raise ValueError(
                        f"Expected calibration batch size 1, got input shape {tuple(inp.shape)}"
                    )
                inp = inp.squeeze(0)
            elif inp.dim() != 2:
                raise ValueError(f"Unexpected calibration input shape {tuple(inp.shape)}")

            inps[i].copy_(inp)   # faster + safer than assignment
            cache["i"] += 1

            # only save once (they're usually same shape)
            if cache["attention_mask"] is None:
                cache["attention_mask"] = kwargs.get("attention_mask", None)

            if cache["position_ids"] is None:
                cache["position_ids"] = kwargs.get("position_ids", None)

            raise ValueError  # stop this forward

    # ===== replace first layer =====
    layers[0] = Catcher(layers[0])

    # ===== run dataloader =====
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
        except StopIteration:
            break  # ✅ stop early when enough samples collected

    # ===== restore layer =====
    layers[0] = layers[0].module

    # ===== outputs placeholder =====
    outs = torch.zeros_like(inps)

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    # ===== restore config =====
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity





def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, args.nsamples
        )
        
    print(inps is None, outs is None, attention_mask is None, position_ids is None)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if hasattr(model, 'hf_device_map')  and (f"model.layers.{i}" in model.hf_device_map):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
            
        for j in range(args.nsamples):
            with torch.no_grad():
                # ✅ compute RoPE correctly
                cos, sin = model.model.rotary_emb(inps[j].unsqueeze(0), position_ids)

                # ✅ ONLY run the current layer
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=(cos, sin)
                )[0]
                
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                # ✅ compute RoPE correctly
                cos, sin = model.model.rotary_emb(inps[j].unsqueeze(0), position_ids)

                # ✅ ONLY run the current layer
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=(cos, sin)
                )[0]
                
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    
    
    
def prune_curvature(args, model, tokenizer, device="cuda:0", prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    model_device = args.model_device      # cuda:0
    compute_device = args.compute_device  # cuda:1

    print("loading calibration data")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")

    with torch.no_grad():
        inps, _, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, model_device, args.nsamples
        )

    # ---- KEEP ON CPU ----
    inps = inps.cpu()
    
    if attention_mask is not None:
        attention_mask = attention_mask.to(model_device)
    if position_ids is not None:
        position_ids = position_ids.to(model_device)

    layers = model.model.layers
    target_ops = ["prev_down_proj", "q_proj", "k_proj", "v_proj",
                  "o_proj", "gate_proj", "up_proj", "down_proj"]
    last_layer_idx = len(layers) - 1

    # Store masks and curvature
    model.removal_mask = [{} for _ in range(len(layers))]
    model.curvature_scores = [{} for _ in range(len(layers))]

    # ---- cross-layer storage ----
    prev_layer_outputs = None
    layer_cache = {}

    for i, layer in enumerate(layers):
        print(f"Processing layer {i}")

        layer = layer.to(model_device)
        layer_subset = find_layers(layer)

        op_modules = {
            short: next(m for n, m in layer_subset.items() if n.endswith(short))
            for short in target_ops if short != "prev_down_proj"
        }
        
        modules_items = list(op_modules.items())

        # ---- init buffers ----
        for short, module in modules_items:
            W = module.weight
            module.min_curvature = torch.full_like(W, float("inf"), device="cpu")
            module.removal_mask = torch.ones_like(W, dtype=torch.bool)
            model.removal_mask[i][short] = module.removal_mask
            
        # ---- allocate CPU buffer ----
        new_inps = torch.empty_like(inps, device="cpu")

        # ---- per-sample loop ----
        for j in range(args.nsamples):
            x = inps[j:j+1].to(model_device, non_blocking=True)

            with torch.no_grad():
                x_out, operations = collect_layer_data(
                    layer, x, attention_mask, position_ids, model, layer_id=i
                )
                
            x_out = x_out.detach().cpu()

            # ---- inject previous layer ----
            if prev_layer_outputs is not None:
                for name in ["gate_proj", "up_proj", "down_proj"]:
                    if name in prev_layer_outputs:
                        operations[f"prev_{name}"] = prev_layer_outputs[name]

            # ---- last layer ----
            if i == last_layer_idx:
                operations.update(_make_lm_head_op(model, x_out, i))
                
            # ---- get layer weight matrix ----   
            if j == 0:
                layer_cache = build_layer_cache(model, operations, i, layer_cache)

            # ---- curvature computation ----
            for short, module in modules_items:
                # skip ONLY real down_proj (not prev_down_proj)
                if short == "down_proj" and i != last_layer_idx:
                    continue

                curv = compute_op_curvature(
                    operations,
                    short,
                    layer_cache,
                    device=compute_device,
                    alpha=args.alpha
                )
                
                assert curv is not None, "curv is None"

                # you will define format later
                param_curv = curv

                torch.minimum(module.min_curvature, param_curv, out=module.min_curvature)
                
            # =========================================================
            # 🔥 DEFERRED prev_down_proj (for previous layer)
            # =========================================================
            if prev_layer_outputs is not None:
                curv = compute_op_curvature(
                    operations,
                    "prev_down_proj",
                    layer_cache,
                    device=compute_device,
                    alpha=args.alpha
                )
                
                assert curv is not None, "prev_down_proj curv is None"

                if curv is not None:
                    prev_i = i - 1
                    if prev_i >= 0:
                        if "down_proj" not in model.curvature_scores[prev_i]:
                            model.curvature_scores[prev_i]["down_proj"] = curv
                        else:
                            torch.minimum(
                                model.curvature_scores[prev_i]["down_proj"],
                                curv,
                                out=model.curvature_scores[prev_i]["down_proj"]
                            )


            # ---- lm_head curvature ----
            if i == last_layer_idx:
                lm_curv = compute_op_curvature(operations, "lm_head", layer_cache, last_layer=True, device=compute_device, alpha=args.alpha)
                assert lm_curv is not None, "lm_head curv is None"
                
                model.curvature_scores[i]["lm_head"] = lm_curv["curvature_score"]

            # ---- store prev outputs (CPU ONLY) ----
            prev_layer_outputs = {
                name: {
                    "node": operations[name]["node"],
                    "extra": operations[name]["extra"],
                }
                for name in ["gate_proj", "up_proj", "down_proj_in", "down_proj"]
                if name in operations
            }

            # ---- next layer input ----
            new_inps[j] = x_out.squeeze(0)

            del operations, x, x_out

            if j % 8 == 0:
                torch.cuda.empty_cache()

        # ---- swap inputs ----
        inps = new_inps

        # ---- store curvature ----
        for short, module in modules_items:
            model.curvature_scores[i][short] = module.min_curvature
            del module.min_curvature

        if i % 2 == 0:
            torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
