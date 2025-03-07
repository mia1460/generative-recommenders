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
    # with CUDATimer("compute full kv", verbose=False) as full_kv_time:
    #     k = torch.mm(x, wk)
    #     v = torch.mm(x, wv)

    with CUDATimer("compute selective kv", verbose=False) as selective_kv_time:
        L = recompute_mask.shape[0]
        D = wk.shape[1]
        cached_k_pad = torch.zeros((L, D), device=x.device, dtype=x.dtype)
        cached_v_pad = torch.zeros((L, D), device=x.device, dtype=x.dtype)

        cached_k_pad.index_copy_(0, target_indices, cached_k)
        cached_v_pad.index_copy_(0, target_indices, cached_v)

        new_x = x[recompute_mask]

        with CUDATimer("compute selective kv", verbose=False) as selective_kv_compute_time:
            cached_k_pad[recompute_mask] = torch.mm(new_x, wk)
            cached_v_pad[recompute_mask] = torch.mm(new_x, wv)

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
    
def get_padded_fusion_kv(
    cached_k: torch.Tensor, # (B, N, D)
    cached_v: torch.Tensor, # (B, N, D)
    x: torch.Tensor, # (B, N, d)
    _uvqk: torch.Tensor, 
    _linear_dim: int,
    _attention_dim: int,
    _num_heads: int,
    recompute_mask: torch.Tensor # (B, N)
):
    _, _v, _, _k = torch.split(
        _uvqk,
        [
            _linear_dim * _num_heads,
            _linear_dim * _num_heads,
            _attention_dim * _num_heads,
            _attention_dim * _num_heads,
        ],
        dim=1,
    )
    mask = recompute_mask.unsqueeze(-1)
    fusion_k = torch.where(mask, torch.matmul(x, _k), cached_k)
    fusion_v = torch.where(mask, torch.matmul(x, _v), cached_v)

    return fusion_k, fusion_v

def create_cached_mask(
    cached_lengths: torch.Tensor, # (B,)
    max_sequence_length: int, 
    device: torch.device, 
):
    indices = torch.arange(max_sequence_length, device=device)
    mask = indices < cached_lengths.unsqueeze(1)
    return mask

def get_next_layer_padded_kv_recompute_mask(
    cached_k: torch.Tensor, # (B, N, D)
    cached_v: torch.Tensor, # (B, N, D)
    compute_k: torch.Tensor, # (B, N, D)
    compute_v: torch.Tensor, # (B, N, D)
    cached_mask: torch.Tensor, # (B, N)
    r: int=20,
):
    diff_k = (cached_k - compute_k).abs().sum(dim=-1)
    diff_v = (cached_v - compute_v).abs().sum(dim=-1)
    diff_total =  diff_k + diff_v

    masked_diff = torch.where(cached_mask, diff_total, -torch.inf)
    print(f"masked_diff is {masked_diff}")

    cached_total = cached_mask.sum().float()
    if cached_total == 0:
        return torch.zeros_like(cached_mask, dtype=torch.bool)
    
    k = int(cached_total * r / 100)

    if k <= 0:
        return torch.zeros_like(cached_mask, dtype=torch.bool)
    
    flat_diff = masked_diff.flatten()
    _, flat_indices = torch.topk(flat_diff, k=k, largest=True, sorted=False)
    print(f"flat_indices is {flat_indices}")

    recompute_mask = torch.zeros_like(cached_mask, dtype=torch.bool)
    recompute_mask.view(-1)[flat_indices] = True


    if False and "compute top k for each user":
        _, sorted_indices = torch.sort(masked_diff, dim=1, descending=True)

        m_i = cached_mask.sum(dim=1).float()
        k = torch.ceil(m_i * r / 100).to(torch.int)
        k = torch.where(m_i > 0, k, torch.zeros_like(k))

        indices = torch.arange(cached_mask.size(1), device=cached_mask.device)
        topk_mask = indices.unsqueeze(0) < k.unsqueeze(1)

        rows, cols = torch.where(topk_mask)
        original_cols = sorted_indices[rows, cols]

        recompute_mask = torch.zeros_like(cached_mask)
        recompute_mask[rows, original_cols] = True

    return recompute_mask
    
    