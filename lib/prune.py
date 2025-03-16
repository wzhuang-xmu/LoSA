import time
import numpy as np
import torch
import torch.nn as nn
from CKA import cka
from scipy import optimize
import math
import gc

from .sparsegpt import SparseGPT
from .layerwrapper import WrappedGPT

def allocate_ranks(loss_values, rank):
    min_rank = 1
    max_rank = 16
    total_loss = np.sum(loss_values)
    initial_ranks = (loss_values / total_loss) * rank * len(loss_values)
    adjusted_ranks = np.clip(np.round(initial_ranks), min_rank, max_rank).astype(int)
    total_adjusted_rank = np.sum(adjusted_ranks)
    diff = int(total_adjusted_rank - rank*len(loss_values))
    if diff > 0:
        indices = np.where(adjusted_ranks > min_rank)[0]
        sorted_indices = indices[np.argsort(loss_values[indices])]
        idx = 0
        while diff > 0 and idx < len(sorted_indices):
            adj_idx = sorted_indices[idx]
            if adjusted_ranks[adj_idx] > min_rank:
                adjusted_ranks[adj_idx] -= 1
                diff -=1
            idx +=1
    elif diff < 0:
        diff = -diff
        indices = np.where(adjusted_ranks < max_rank)[0]
        sorted_indices = indices[np.argsort(-loss_values[indices])]
        idx = 0
        while diff > 0 and idx < len(sorted_indices):
            adj_idx = sorted_indices[idx]
            if adjusted_ranks[adj_idx] < max_rank:
                adjusted_ranks[adj_idx] += 1
                diff -=1
            idx +=1
    return adjusted_ranks.tolist()


def get_feature_map(args, model, tokenizer, device=torch.device("cuda:0"), dataloader=None):
    model.config.use_cache = False
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            args, model, dataloader, device
        )
    
    layers = model.model.layers

    feature = []
    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        print(f"get dense feature map of layer {i}")

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()
        
        feature.append(outs.clone().detach().cpu())

        inps, outs = outs, inps

    

    return feature


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers and 'lora' not in name:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
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
            sub_module_count = (W == 0).sum().item()
            count += (W == 0).sum().item()
            sub_module_params = W.numel()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params


def prepare_calibration_input(args, model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype

    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device,
    )
    inps.requires_grad = False
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def prune_wanda(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, 
    dataloader=None, calib_dataloader=None, dense_feature=None, prune_iter=0, iters=10,
):
    iter_sparsity_ratio = args.sparsity_ratio-args.sparsity_ratio*(1-prune_iter/iters)**3

    ###### layer-wise sparsity rate ######
    if iter_sparsity_ratio <= 0.5:
        args.delta_ratio = 0.01
    elif iter_sparsity_ratio > 0.5 and iter_sparsity_ratio <= 0.6:
        args.delta_ratio = 0.02
    elif iter_sparsity_ratio > 0.6 and iter_sparsity_ratio <= 0.7:
        args.delta_ratio = 0.03

    if '7' in args.model or '8' in args.model:
        args.nsamples=64
    elif '13' in args.model:
        args.nsamples=32
    
    model.config.use_cache = False
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            args, model, calib_dataloader, device
        )
    
    layers = model.model.layers

    feature = []
    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        print(f"get sparse feature map of layer {i}")

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()
        
        feature.append(inps.clone().detach())

        inps, outs = outs, inps
    
    
    for i in range(len(feature)):
        feature[i] = feature[i].view(args.nsamples, -1)

    similar_matrix = torch.zeros((len(feature), len(feature)), device=feature[0].device)


    for i in range(len(feature)):
        for j in range(len(feature)):
            with torch.no_grad():
                similar_matrix[i][j] = cka.cka(cka.gram_linear(feature[i].float()), cka.gram_linear(feature[j].float()))
        

    def sum_list(a, j):
        b = 0
        for i in range(len(a)):
            if i != j:
                b += a[i]
        return b

    global important, length
    important = []
    temp = []

    for i in range(len(feature)):
        temp.append( sum_list(similar_matrix[i], i) )

    b = sum_list(temp, -1)
    temp = [x/b for x in temp]

    beta = 1
    for i in range(len(feature)):
        important.append( torch.exp(-1* beta *temp[i] ) )
    
    length = len(important)

    important = np.array([t.cpu().numpy() for t in important])
    feature.clear()
    del feature
    torch.cuda.empty_cache()

    # Objective function
    def func(x, sign=1.0):
        """ Objective function """
        sum_fuc =[]
        for idx1 in range(length):
            sum_fuc.append(x[idx1]*important[idx1])
        return sum(sum_fuc)


    # Derivative function of objective function
    def func_deriv(x, sign=1.0):
        """ Derivative of objective function """
        global important
        diff = []
        for i in range(len(important)):
            diff.append(sign * (important[i]))
        return np.array(diff)

    # Constraint function
    def constrain_func(x):
        """ constrain function """
        return np.mean(x) - iter_sparsity_ratio

    bnds = []
    for i in range(length):
        bnds.append((iter_sparsity_ratio-args.delta_ratio, iter_sparsity_ratio+args.delta_ratio))

    bnds = tuple(bnds)
    cons = ({'type': 'eq', 'fun': constrain_func},)

    result = optimize.minimize(func, x0=[1 for i in range(length)], jac=func_deriv, method='SLSQP', bounds=bnds, constraints=cons)
    all_layer_ratio = result.x.tolist()

    args.nsamples = 128
    ###### layer-wise sparsity rate ######

    use_cache = model.config.use_cache
    model.config.use_cache = False
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            args, model, dataloader, device
        )
    
    layers = model.model.layers
    recon_loss = []
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            if inps is not None:
                inps = inps.to(dev)
            if outs is not None:
                outs = outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)

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
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        for h in handles:
            h.remove()


        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            W_mask = (
                torch.zeros_like(W_metric) == 1
            )  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if all_layer_ratio[i] < 0:
                    all_layer_ratio[i] = 0

                # unstructured pruning
                indices = sort_res[1][:, : int(W_metric.shape[1] * all_layer_ratio[i])]
                W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

        dense_feature[i] = dense_feature[i].cuda()
        l2_scaler = torch.norm(dense_feature[i].reshape((-1, dense_feature[i].shape[-1])).t(), p=2, dim=1).detach()
        l2_loss = (((dense_feature[i] - outs) / l2_scaler) ** 2).sum() / outs.shape[-1]
        recon_loss.append(l2_loss.item())
        dense_feature[i] = dense_feature[i].cpu()

        inps, outs = outs, inps

    dense_feature = [feature.cpu() for feature in dense_feature]

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################

    gc.collect()

    return recon_loss


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, 
                    dataloader=None, calib_dataloader = None, dense_feature=None, prune_iter=0, iters=10):

    iter_sparsity_ratio = args.sparsity_ratio-args.sparsity_ratio*(1-prune_iter/iters)**3

    ###### layer-wise sparsity rate ######
    if iter_sparsity_ratio <= 0.5:
        args.delta_ratio = 0.01
    elif iter_sparsity_ratio > 0.5 and iter_sparsity_ratio <= 0.6:
        args.delta_ratio = 0.02
    elif iter_sparsity_ratio > 0.6 and iter_sparsity_ratio <= 0.7:
        args.delta_ratio = 0.03

    if '7' in args.model or '8' in args.model:
        args.nsamples=64
    elif '13' in args.model:
        args.nsamples=32
    model.config.use_cache = False
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            args, model, calib_dataloader, device
        )
    layers = model.model.layers

    feature = []
    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        print(f"get sparse feature map of layer {i}")

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()
        
        feature.append(inps.clone().detach())

        inps, outs = outs, inps
    
    
    for i in range(len(feature)):
        feature[i] = feature[i].view(args.nsamples, -1).to(device)

    similar_matrix = torch.zeros((len(feature), len(feature)), device=feature[0].device)


    for i in range(len(feature)):
        for j in range(len(feature)):
            with torch.no_grad():
                similar_matrix[i][j] = cka.cka(cka.gram_linear(feature[i].float()), cka.gram_linear(feature[j].float()))
        

    def sum_list(a, j):
        b = 0
        for i in range(len(a)):
            if i != j:
                b += a[i]
        return b

    global important, length
    important = []
    temp = []

    for i in range(len(feature)):
        temp.append( sum_list(similar_matrix[i], i) )

    b = sum_list(temp, -1)
    temp = [x/b for x in temp]

    beta = 100
    for i in range(len(feature)):
        important.append( torch.exp(-1* beta *temp[i] ) )
    
    length = len(important)

    important = np.array([t.cpu().numpy() for t in important])
    feature.clear()
    del feature
    torch.cuda.empty_cache()

    # Objective function
    def func(x, sign=1.0):
        """ Objective function """
        sum_fuc =[]
        for idx1 in range(length):
            sum_fuc.append(x[idx1]*important[idx1])
        return sum(sum_fuc)


    # Derivative function of objective function
    def func_deriv(x, sign=1.0):
        """ Derivative of objective function """
        global important
        diff = []
        for i in range(len(important)):
            diff.append(sign * (important[i]))
        return np.array(diff)

    # Constraint function
    def constrain_func(x):
        """ constrain function """
        return np.mean(x) - iter_sparsity_ratio

    bnds = []
    for i in range(length):
        bnds.append((iter_sparsity_ratio-args.delta_ratio, iter_sparsity_ratio+args.delta_ratio))

    bnds = tuple(bnds)
    cons = ({'type': 'eq', 'fun': constrain_func},)

    result = optimize.minimize(func, x0=[1 for i in range(length)], jac=func_deriv, method='SLSQP', bounds=bnds, constraints=cons)
    all_layer_ratio = result.x.tolist()

    args.nsamples = 128
    ###### layer-wise sparsity rate ######

    use_cache = model.config.use_cache
    model.config.use_cache = False

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            args, model, dataloader, device
        )

    layers = model.model.layers
    recon_loss = []

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            if inps is not None:
                inps = inps.to(dev)
            if outs is not None:
                outs = outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(f"pruning layer {i} name {name}")

            if all_layer_ratio[i] < 0:
                all_layer_ratio[i] = 0

            gpts[name].fasterprune(
                all_layer_ratio[i],
                prune_n=prune_n,
                prune_m=prune_m,
                percdamp=0.01,
                blocksize=128,
            )

            gpts[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        dense_feature[i] = dense_feature[i].cuda()
        l2_scaler = torch.norm(dense_feature[i].reshape((-1, dense_feature[i].shape[-1])).t(), p=2, dim=1).detach()
        l2_loss = (((dense_feature[i] - outs) / l2_scaler) ** 2).sum() / outs.shape[-1]
        recon_loss.append(l2_loss.item())
        dense_feature[i] = dense_feature[i].cpu()

        inps, outs = outs, inps

    dense_feature = [feature.cpu() for feature in dense_feature]

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################

    gc.collect()

    return recon_loss