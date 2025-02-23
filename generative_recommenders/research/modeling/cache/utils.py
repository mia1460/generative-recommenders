# utils funcs for selective kv cache reuse
import torch
from typing import Tuple
import time
from generative_recommenders.research.modeling.cache.timer import (
    CUDATimer,
)

def get_fusion_kv(
    cached_k: torch.Tensor, # (L', D)
    cached_v: torch.Tensor, # (L', D)
    x: torch.Tensor, # (L, d)
    wk: torch.Tensor, # (d, D)
    wv: torch.Tensor, # (d, D)
    recompute_mask: torch.Tensor, # (L)
    target_indices: torch.Tensor, # (L')
):
    with CUDATimer("compute full kv", verbose=False) as full_kv_time:
        k = torch.mm(x, wk)
        v = torch.mm(x, wv)

    with CUDATimer("compute selective kv", verbose=False) as selective_kv_time:
        # begin_time = time.time()
        L = recompute_mask.shape[0]
        D = wk.shape[1]
        cached_k_pad = torch.zeros((L, D), device=x.device, dtype=x.dtype)
        cached_v_pad = torch.zeros((L, D), device=x.device, dtype=x.dtype)

        cached_k_pad.index_copy_(0, target_indices, cached_k)
        cached_v_pad.index_copy_(0, target_indices, cached_v)
        # get_kv_pad_end_time = time.time()

        # new_k = cached_k_pad.clone()
        # new_v = cached_v_pad.clone()
        new_x = x[recompute_mask]
        # new_x = new_x.contiguous()
        # get_new_x_end_time = time.time()

        # compute_xkv_begin_time = time.time()
        # k = torch.mm(x, wk)
        # v = torch.mm(x, wv)
        # compute_xkv_end_time = time.time()
        # print(f"new_x.device is {new_x.device}, wk.device is {wk.device}") # cuda:0


        # new_k = torch.mm(new_x, wk)
        # new_v = torch.mm(new_x, wv)

        # if recompute_mask.any():
        # with torch.no_grad(): # will add time
        with CUDATimer("compute selective kv", verbose=False) as selective_kv_compute_time:
            cached_k_pad[recompute_mask] = torch.mm(new_x, wk)
            cached_v_pad[recompute_mask] = torch.mm(new_x, wv)
        # compute_end_time = time.time()

        # cached_k_pad[recompute_mask] = new_k
        # cached_v_pad[recompute_mask] = new_v
        # back_masked_kv_end_time = time.time()

        # return cached_k_pad, cached_v_pad, new_k, new_v
        # print(f"At get_fusion_kv: cached_k.shape@{cached_k.shape}, x.shape@{x.shape}, get_kv_pad need {get_kv_pad_end_time-begin_time:.6f} s, get x[recompute_mask] need {get_new_x_end_time - get_kv_pad_end_time:.6f} s, compute full kv need {compute_xkv_end_time-compute_xkv_begin_time:.6f} s, compute selective kv need {compute_end_time-compute_xkv_end_time:.6f} s, assign cached_kv with recompute_kv need {back_masked_kv_end_time - compute_end_time:.6f}")
        # print(f"At get_fusion_kv: cached_k.shape@{cached_k.shape}, x.shape@{x.shape}, get_kv_pad need {get_kv_pad_end_time-begin_time:.6f} s, get x[recompute_mask] need {get_new_x_end_time - get_kv_pad_end_time:.6f} s, compute selective kv need {compute_end_time-get_new_x_end_time:.6f} s, assign cached_kv with recompute_kv need {back_masked_kv_end_time - compute_end_time:.6f}")
    # print(f"compute full kv need {full_kv_time.get_time():.4f} ms, compute selective kv need {selective_kv_time.get_time():.4f} ms, which truly compute need {selective_kv_compute_time.get_time():.4f} ms")
    return cached_k_pad, cached_v_pad

def get_next_layer_kv_diff_mask(
    cached_k: torch.Tensor, # (L', D)
    cached_v: torch.Tensor, # (L', D)
    compute_k: torch.Tensor, # (L, D)
    compute_v: torch.Tensor, # (L, D)
    target_indices: torch.Tensor, # (L')
    device: torch.device, 
    r: int=20,
):
    # begin_time = time.time()
    L, D = compute_k.shape

    cached_k_pad = torch.zeros((L, D), device=device, dtype=compute_k.dtype)
    cached_v_pad = torch.zeros((L, D), device=device, dtype=compute_v.dtype)

    cached_k_pad.index_copy_(0, target_indices, cached_k)
    cached_v_pad.index_copy_(0, target_indices, cached_v)
    # get_kv_pad_end_time = time.time()

    diff_mask = torch.zeros((L, D), device=device, dtype=torch.bool)
    diff_mask[target_indices] = True
    recompute_mask = torch.ones(L, device=device, dtype=torch.bool)
    recompute_mask[target_indices] = False
    # prepare_mask_end_time = time.time()

    # print(f"cached_k_pad[diff_mask] is {cached_k_pad.shape}, compute_k[diff_mask] is {compute_k[diff_mask].reshape(-1, D).shape}")

    diff_k = torch.sum(torch.abs(cached_k_pad[diff_mask].reshape(-1, D) - compute_k[diff_mask].reshape(-1, D)), dim=1)
    diff_v = torch.sum(torch.abs(cached_v_pad[diff_mask].reshape(-1, D) - compute_v[diff_mask].reshape(-1, D)), dim=1)

    diff = diff_k + diff_v

    k = int(target_indices.shape[0] * r / 100)
    top_r_idx = torch.topk(diff, k, largest=True).indices
    recompute_mask[target_indices] = recompute_mask[target_indices].scatter(0, top_r_idx, torch.ones_like(top_r_idx, dtype=torch.bool))
    # compute_end_time = time.time()
    # print(f"At get_next_layer_kv_diff_mask: cached_k.shape@{cached_k.shape}, compute_k.shape@{compute_k.shape}, get_kv_pad need {get_kv_pad_end_time-begin_time:.6f} s, prepare init mask need {prepare_mask_end_time-get_kv_pad_end_time:.6f} s, compute mask need {compute_end_time-prepare_mask_end_time:.6f} s")

    return recompute_mask
    

    