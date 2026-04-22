import torch


def _get_prunable_module(model, layer_idx, op_name):
    layer = model.model.layers[layer_idx]

    if op_name == "lm_head":
        return model.lm_head
    if op_name == "down_proj":
        return layer.mlp.down_proj
    if op_name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return getattr(layer.self_attn, op_name)
    if op_name in {"gate_proj", "up_proj"}:
        return getattr(layer.mlp, op_name)

    raise KeyError(f"Unsupported op_name for curvature pruning: {op_name}")


def prune_global_curvature(args, model):
    if not hasattr(model, "curvature_scores"):
        raise AttributeError("Model does not have curvature_scores. Run prune_curvature first.")
    if not hasattr(model, "removal_mask"):
        model.removal_mask = [{} for _ in range(len(model.model.layers))]

    model.eval()

    score_refs = []
    total_finite_params = 0

    for layer_idx, layer_scores in enumerate(getattr(model, "curvature_scores", [])):
        for op_name, curv in layer_scores.items():
            module = _get_prunable_module(model, layer_idx, op_name)
            weight = module.weight.data
            curv_cpu = curv.detach().cpu() if torch.is_tensor(curv) else torch.as_tensor(curv)

            if curv_cpu.shape != weight.shape:
                raise ValueError(
                    f"Curvature shape mismatch for layer {layer_idx} {op_name}: "
                    f"{tuple(curv_cpu.shape)} vs {tuple(weight.shape)}"
                )

            finite_mask = torch.isfinite(curv_cpu)
            finite_count = int(finite_mask.sum().item())
            if finite_count == 0:
                print(f"Skipping layer {layer_idx} {op_name}: no finite curvature scores")
                continue

            score_refs.append(
                {
                    "layer_idx": layer_idx,
                    "op_name": op_name,
                    "module": module,
                    "curv": curv_cpu,
                    "finite_mask": finite_mask,
                }
            )
            total_finite_params += finite_count

    if total_finite_params == 0:
        print("No finite curvature scores found for global curvature pruning")
        return

    # Only parameters with finite curvature scores participate in global pruning.
    prune_count = int(total_finite_params * args.sparsity_ratio)
    if prune_count <= 0:
        print("Global curvature pruning skipped because prune_count is 0")
        return

    all_scores = torch.cat([entry["curv"][entry["finite_mask"]].reshape(-1) for entry in score_refs])
    prune_count = min(prune_count, all_scores.numel())
    topk_indices = torch.topk(all_scores, k=prune_count, largest=True, sorted=False).indices
    global_prune_mask = torch.zeros(all_scores.numel(), dtype=torch.bool)
    global_prune_mask[topk_indices] = True
    threshold = all_scores[topk_indices].min()

    total_pruned = 0
    offset = 0
    with torch.no_grad():
        for entry in score_refs:
            module = entry["module"]
            curv = entry["curv"]
            finite_mask = entry["finite_mask"]
            flat_finite_count = int(finite_mask.sum().item())
            flat_selection = global_prune_mask[offset:offset + flat_finite_count]
            offset += flat_finite_count

            prune_mask_cpu = torch.zeros_like(finite_mask, dtype=torch.bool)
            prune_mask_cpu[finite_mask] = flat_selection
            layer_pruned = int(prune_mask_cpu.sum().item())
            total_pruned += layer_pruned

            prune_mask = prune_mask_cpu.to(device=module.weight.data.device)
            module.weight.data[prune_mask] = 0

            model.removal_mask[entry["layer_idx"]][entry["op_name"]] = ~prune_mask_cpu
            print(
                f"Global curvature prune layer {entry['layer_idx']} {entry['op_name']}: "
                f"pruned={layer_pruned}/{int(finite_mask.sum().item())}"
            )

    print(
        f"Global curvature pruning complete: pruned={total_pruned}/{total_finite_params}, "
        f"finite_curvature_sparsity={args.sparsity_ratio:.6f}, threshold={float(threshold):.6f}"
    )
