import os
import pickle
import time 
import torch 
import torch.nn as nn 
from layerwrapper import WrappedGPT
from data import get_loaders 
from data_c4 import get_loaders_c4
from layerwrapper_curv import collect_layer_data, _make_lm_head_op
from cal_curvature import compute_op_curvature
from curv_tensor_utils import build_layer_cache
from curv_shortest_path_utils import build_shortest_path_cache


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


def _clone_pickleable_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _clone_pickleable_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_pickleable_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_pickleable_to_cpu(v) for v in value)
    return value


def save_layer_curvature_pkl(layer_idx, curvature_scores, save_dir, metadata=None):
    if save_dir is None:
        return None

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"layer_{layer_idx:03d}_curvature.pkl")
    payload = {
        "layer_idx": int(layer_idx),
        "curvature_scores": {
            op_name: _clone_pickleable_to_cpu(curv)
            for op_name, curv in curvature_scores.items()
        },
    }
    if metadata is not None:
        payload["metadata"] = _clone_pickleable_to_cpu(metadata)

    with open(save_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return save_path


def load_layer_curvature_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    layer_idx = int(payload["layer_idx"])
    curvature_scores = {
        op_name: _clone_pickleable_to_cpu(curv)
        for op_name, curv in payload.get("curvature_scores", {}).items()
    }
    metadata = payload.get("metadata", None)
    return layer_idx, curvature_scores, metadata


def load_curvature_pkls(save_dir, num_layers):
    curvature_scores = [{} for _ in range(num_layers)]
    if save_dir is None or not os.path.isdir(save_dir):
        return curvature_scores

    for file_name in sorted(os.listdir(save_dir)):
        if not (file_name.startswith("layer_") and file_name.endswith("_curvature.pkl")):
            continue

        layer_idx, layer_scores, _ = load_layer_curvature_pkl(os.path.join(save_dir, file_name))
        if 0 <= layer_idx < num_layers:
            curvature_scores[layer_idx] = layer_scores

    return curvature_scores


def _curvature_save_metadata(args):
    return {
        "model": getattr(args, "model", None),
        "seed": getattr(args, "seed", None),
        "nsamples": getattr(args, "nsamples", None),
        "alpha": getattr(args, "alpha", None),
        "sparsity_ratio": getattr(args, "sparsity_ratio", None),
        "prune_method": getattr(args, "prune_method", None),
    }


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
    dataloader, _ = get_loaders_c4(
        args.calib_data,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")
    
    model.eval()

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
    # target_ops = ["prev_down_proj", "q_proj", "k_proj", "v_proj",
    #               "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    target_ops = ["v_proj"]
    last_layer_idx = len(layers) - 1

    # Store masks and curvature
    model.removal_mask = [{} for _ in range(len(layers))]
    model.curvature_scores = [{} for _ in range(len(layers))]
    curvature_save_dir = getattr(args, "save_curvature_dir", None)
    curvature_load_dir = getattr(args, "load_curvature_dir", None)
    curvature_metadata = _curvature_save_metadata(args)

    if curvature_load_dir is not None:
        model.curvature_scores = load_curvature_pkls(curvature_load_dir, len(layers))
        print(f"Loaded curvature pkls from {curvature_load_dir}")
        model.config.use_cache = use_cache
        return

    # ---- cross-layer storage ----
    prev_layer_outputs = [None] * args.nsamples
    layer_cache = {}
    sp_cache = {} 

    for i, layer in enumerate(layers):
        print(f"Processing layer {i}")
        
        layer_cache = {}
        sp_cache = {}   # overwrite same layer cache when moving to next layer

        layer = layer.to(model_device)
        layer_subset = find_layers(layer)

        op_modules = {
            short: next(m for n, m in layer_subset.items() if n.endswith(short))
            for short in target_ops
        }
        modules_items = list(op_modules.items())
        has_down_proj_target = "down_proj" in op_modules

        for short, module in modules_items:
            W = module.weight
            module.min_curvature = torch.full_like(W, float("inf"), device="cpu")
            module.removal_mask = torch.ones_like(W, dtype=torch.bool)
            model.removal_mask[i][short] = module.removal_mask

        new_inps = torch.empty_like(inps, device="cpu")

        for j in range(args.nsamples):
            x = inps[j:j+1].to(model_device, non_blocking=True) # [1, seq, hiddensize]
            next_layer = layers[i + 1] if i < last_layer_idx else None

            with torch.no_grad():
                x_out, operations, num_q_heads, num_kv_heads, repeat, head_dim = collect_layer_data(
                    layer,
                    x,
                    attention_mask,
                    position_ids,
                    model,
                    next_layer=next_layer,
                )

            x_out = x_out.detach().cpu()
            
            print(f'Finish getting layer data!!!')

            if prev_layer_outputs[j] is not None:
                for name in ["o_proj", "gate_up_out", "down_proj"]:
                    if name in prev_layer_outputs[j]:
                        operations[f"prev_{name}"] = prev_layer_outputs[j][name]

            if i == last_layer_idx:
                operations.update(_make_lm_head_op(model, x_out))

            if j == 0:
                layer_cache = build_layer_cache(model, operations, i, layer_cache, device=compute_device)

                for short, _ in modules_items:
                    build_shortest_path_cache(
                        operations=operations,
                        layer_cache=layer_cache,
                        short_name=short,
                        sp_cache=sp_cache,
                        device=compute_device,
                    )

                if prev_layer_outputs[j] is not None and has_down_proj_target:
                    build_shortest_path_cache(
                        operations=operations,
                        layer_cache=layer_cache,
                        short_name="prev_down_proj",
                        sp_cache=sp_cache,
                        device=compute_device,
                    )

                if i == last_layer_idx:
                    build_shortest_path_cache(
                        operations=operations,
                        layer_cache=layer_cache,
                        short_name="lm_head",
                        sp_cache=sp_cache,
                        device=compute_device,
                    )

            for short, module in modules_items:
                if short == "down_proj" and i != last_layer_idx:
                    continue
                
                print(f'Layer {i}, op name = {short}')

                curv = compute_op_curvature(
                    operations=operations,
                    short_name=short,
                    layer_id=i,
                    sample_idx=j,
                    layer_cache=layer_cache,
                    sp_cache=sp_cache,
                    device=compute_device,
                    alpha=args.alpha,
                    seq_len = model.seqlen,
                    num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, 
                    head_dim=head_dim, repeat=repeat,
                    sample_edge_num=args.sample_edge_num,
                    dataset_name=args.calib_data,
                )

                assert curv is not None, f"{short} curv is None"

                param_curv = curv
                assert (
                    param_curv.shape == module.weight.shape
                    or param_curv.T.shape == module.weight.shape
                ), f"Unexpected curvature shape for {short}: {tuple(param_curv.shape)} vs {tuple(module.weight.shape)}"

                if param_curv.T.shape == module.weight.shape:
                    param_curv = param_curv.T

                torch.minimum(module.min_curvature, param_curv, out=module.min_curvature)

            if prev_layer_outputs[j] is not None and has_down_proj_target:
                curv = compute_op_curvature(
                    operations=operations,
                    short_name="prev_down_proj",
                    layer_id=i,
                    sample_idx=j,
                    layer_cache=layer_cache,
                    sp_cache=sp_cache,
                    device=compute_device,
                    alpha=args.alpha,
                    seq_len = model.seqlen,
                    num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, 
                    head_dim=head_dim, repeat=repeat,
                    sample_edge_num=args.sample_edge_num,
                    dataset_name=args.calib_data,
                )

                assert curv is not None, "prev_down_proj curv is None"

                prev_i = i - 1
                if prev_i >= 0:
                    prev_weight = model.model.layers[prev_i].mlp.down_proj.weight.detach().cpu()
                    param_curv = curv
                    assert (
                        param_curv.shape == prev_weight.shape
                        or param_curv.T.shape == prev_weight.shape
                    ), f"Unexpected curvature shape for prev down_proj: {tuple(param_curv.shape)} vs {tuple(prev_weight.shape)}"

                    if param_curv.T.shape == prev_weight.shape:
                        param_curv = param_curv.T

                    if "down_proj" not in model.curvature_scores[prev_i]:
                        model.curvature_scores[prev_i]["down_proj"] = param_curv
                    else:
                        torch.minimum(
                            model.curvature_scores[prev_i]["down_proj"],
                            param_curv,
                            out=model.curvature_scores[prev_i]["down_proj"],
                        )

                    save_path = save_layer_curvature_pkl(
                        layer_idx=prev_i,
                        curvature_scores=model.curvature_scores[prev_i],
                        save_dir=curvature_save_dir,
                        metadata=curvature_metadata,
                    )
                    if save_path is not None:
                        print(f"Saved curvature pkl: {save_path}")

            if i == last_layer_idx:
                lm_curv = compute_op_curvature(
                    operations=operations,
                    short_name="lm_head",
                    layer_id=i,
                    sample_idx=j,
                    layer_cache=layer_cache,
                    sp_cache=sp_cache,
                    device=compute_device,
                    alpha=args.alpha,
                    seq_len = model.seqlen,
                    num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, 
                    head_dim=head_dim, repeat=repeat,
                    sample_edge_num=args.sample_edge_num,
                    dataset_name=args.calib_data,
                )
                assert lm_curv is not None, "lm_head curv is None"

                if lm_curv.T.shape == model.lm_head.weight.shape:
                    lm_curv = lm_curv.T
                model.curvature_scores[i]["lm_head"] = lm_curv

            prev_layer_outputs[j] = {
                name: operations[name]
                for name in ["o_proj", "gate_up_out", "down_proj"]
                if name in operations
            }

            new_inps[j] = x_out.squeeze(0)

            del operations, x, x_out

            if j % 8 == 0:
                torch.cuda.empty_cache()

        inps = new_inps

        for short, module in modules_items:
            model.curvature_scores[i][short] = module.min_curvature
            del module.min_curvature

        save_path = save_layer_curvature_pkl(
            layer_idx=i,
            curvature_scores=model.curvature_scores[i],
            save_dir=curvature_save_dir,
            metadata=curvature_metadata,
        )
        if save_path is not None:
            print(f"Saved curvature pkl: {save_path}")

        if i % 2 == 0:
            torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
