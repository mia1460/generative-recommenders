# utils funcs for selective kv cache reuse
import torch
from typing import Tuple
import time

# ignore begin
def compute_one_token_diff(
    cached_tensor: torch.Tensor, # both [D] & [1, D] is ok, but need cached_tensor.shape == compute_tensor.shape
    compute_tensor: torch.Tensor, # both [D] & [1, D] is ok
):
    assert cached_tensor.shape == compute_tensor.shape, (
        f"[compute_one_token_diff_input_error]: cached_tensor.size(){cached_tensor.shape} != compute_tensor.size(){compute_tensor.shape}"
    )
    diff = torch.sum(torch.abs(cached_tensor-compute_tensor))

    return diff

def get_top_k_pos_mask( 
    diff_tensor: torch.Tensor, # [items_length]
    top_k: int, # get value from get_fusion_cache call
):
    assert len(diff_tensor.shape) == 1, (
        f"[get_top_r_percent_pos_mask_input_error]: diff_tensor.shape need to have only one dim, not {len(diff_tensor.shape)}"
    )
    # print(f" get top _ {top_k} from diff_tensor {diff_tensor}")
    ret_shape = diff_tensor.shape
    _, indices = torch.topk(diff_tensor, top_k)
    mask = torch.zeros(ret_shape, dtype = bool)
    mask[indices] = True
    return mask

def get_fusion_cache( # get one layer one user's k/v cache
    cached_tensor: torch.Tensor, # [items_length, dk/dv]
    compute_tensor: torch.Tensor, # [items_length, dk/dv]
    last_layer_mask: torch.Tensor = None, # [items_length]
    r: int = 20,
):
    ret_tensor = cached_tensor.clone()
    ret_tensor[last_layer_mask] = compute_tensor[last_layer_mask]
    length = cached_tensor.size(0)
    diff_length = length
    top_k = int(r * length / 100)
    if last_layer_mask is not None:
        diff_length = torch.sum(last_layer_mask).item()
        # print(f"diff_length is {diff_length}")
        if diff_length == 0:
            return cached_tensor, last_layer_mask
        top_k = int(r * diff_length / 100)
    # print(f"top_k is {top_k}, length is {length}")
    diff_tensor = torch.zeros(length, dtype = cached_tensor.dtype)
    for i in range(length):
        if last_layer_mask is not None:
            # print(f"the last_layer_mask[i] is {last_layer_mask[i]}")
            if last_layer_mask[i] == True:
                diff_tensor[i] = compute_one_token_diff(cached_tensor[i], compute_tensor[i])
            else:
                diff_tensor[i] = 0.0
                continue
        else:
            diff_tensor[i] = compute_one_token_diff(cached_tensor[i], compute_tensor[i])
    pos_mask = get_top_k_pos_mask(diff_tensor, top_k=top_k)
    # ret_tensor[pos_mask] = compute_tensor[pos_mask]
    # print(f"in get_fusion_cache: pos_mask is {type(pos_mask)}, ret_tensor is {type(ret_tensor)}")
    return ret_tensor, pos_mask

def get_batch_fusion_k_cache( # get one layer's all users' k cache
    cached_k: torch.Tensor, # [B, N, D]
    padded_k: torch.Tensor, # [B, N, D]
    last_layer_k_mask: torch.Tensor, # [B, N]
    r: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = cached_k.size(0)
    # print(f"the N in cached_k is {cached_k.size(1)}")
    fusion_k_all = []
    k_mask_all = []

    for i in range(B): # deal with each user
        cur_user_cache = cached_k[i]
        cur_user_padded = padded_k[i]
        cur_user_mask = last_layer_k_mask[i]
        cur_user_fusion_k, cur_user_mask = get_fusion_cache(
            cached_tensor=cur_user_cache,
            compute_tensor=cur_user_padded,
            last_layer_mask=cur_user_mask,
            r=r,
        )

        fusion_k_all.append(cur_user_fusion_k)
        k_mask_all.append(cur_user_mask)

    ret_fusion_k = torch.stack(fusion_k_all)
    ret_layer_mask = torch.stack(k_mask_all)

    return ret_fusion_k, ret_layer_mask
    
def get_batch_fusion_v_cache( # get one layer's all users' v cache
    cached_v: torch.Tensor, # [sum_i(N_i), D]
    padded_v: torch.Tensor, # [sum_i(N_i), D]
    last_layer_v_mask: torch.Tensor, # [sum_i(N_i)]
    past_lengths: torch.Tensor, # [B]
    N: int,
    r: int = 20, 
):
    B = past_lengths.size(0)
    # N = last_layer_v_mask.size(1)
    # print(f"the N is {N}")
    D = cached_v.size(1)

    padded_cached_v = torch.zeros((B, N, D), dtype = cached_v.dtype, device=cached_v.device)
    padded_padded_v = torch.zeros((B, N, D), dtype = padded_v.dtype, device=padded_v.device)
    padded_last_layer_v_mask = torch.zeros((B, N), dtype = last_layer_v_mask.dtype, device=last_layer_v_mask.device)

    start_idx = 0
    fusion_v_all = []
    v_mask_all = []
    for i in range(B):
        valid_length = past_lengths[i].item()
        # print(f"valid_length is {valid_length}")
        if valid_length > 0:
            cur_user_cache = cached_v[start_idx:start_idx + valid_length, :]
            cur_user_padded = padded_v[start_idx:start_idx + valid_length, :]
            cur_user_mask = last_layer_v_mask[start_idx:start_idx + valid_length]
            # print(f"the shapes are: cur_user_cache is {cur_user_cache.shape}, cur_user_padded is {cur_user_padded.shape}, cur_user_mask is {cur_user_mask.shape}")
            # print(f"cur_user_cache is {cur_user_cache}, cur_user_padded is {cur_user_padded}, cur_user_mask is {cur_user_mask}")
            cur_user_fusion_v, cur_user_mask = get_fusion_cache(
                cached_tensor=cur_user_cache,
                compute_tensor=cur_user_padded,
                last_layer_mask=cur_user_mask,
                r=r,
            )

            fusion_v_all.append(cur_user_fusion_v)
            v_mask_all.append(cur_user_mask)
            start_idx += valid_length

    # print(f"padded_cached_v is {padded_cached_v}\npadded_padded_v is {padded_padded_v}")

    # fusion_v, layer_mask_v = get_batch_fusion_k_cache(
    #     cached_k=padded_cached_v,
    #     padded_k=padded_padded_v,
    #     last_layer_k_mask=padded_last_layer_v_mask,
    #     r=r,
    # )
    # print(f"types are: ret_fusion_v is {type(ret_fusion_v)}, ret_layer_mask_v is {type(ret_layer_mask_v)}")
    
    # ret_fusion_v = torch.cat([fusion_v[i, :past_lengths[i].item(), :] for i in range(B)])
    # # print(f"layer_mask_v from func() is {layer_mask_v}")
    # ret_layer_mask_v = torch.cat([layer_mask_v[i, :past_lengths[i].item()] for i in range(B)])

    ret_fusion_v = torch.cat(fusion_v_all)
    ret_layer_mask_v = torch.cat(v_mask_all)

    return ret_fusion_v, ret_layer_mask_v

def get_next_layer_v_diff_mask( # for getting first layer's top r% v deviation position mask.
    cached_v: torch.Tensor, # (L', D)
    compute_v: torch.Tensor, # (L, D)
    cached_lengths: torch.Tensor,
    x_offsets: torch.Tensor,
    r: int=20, 
): # return a mask, which True means its position need reuse
    L, D = compute_v.shape
    B = len(cached_lengths)

    # transfer cached_v from (L', D) to (L, D), with cached_lengths and x_offsets
    cached_v_pad = torch.zeros(L, D)
    cached_v_start_idx = 0
    for i in range(B):
        effective_len = cached_lengths[i]
        target_start_idx = x_offsets[i]

        valid_vs = cached_v[target_start_idx:target_start_idx+effective_len, :]

        cached_v_pad[target_start_idx: target_start_idx+effective_len, :] = valid_vs

        cached_v_start_idx += effective_len 
    
    diff = torch.sum(torch.abs(cached_v_pad - compute_v), dim=1)
    # print(f"diff is {diff}")

    mask = torch.zeros(L, dtype=torch.bool, device=compute_v.device)

    for user_idx in range(B):

        start_idx = x_offsets[user_idx]
        end_idx = x_offsets[user_idx+1]
        effective_len = cached_lengths[user_idx]

        # print(f"start_idx is {start_idx}, effetive_len is {effective_len}")

        user_diff = diff[start_idx:start_idx+effective_len]
        k = int(effective_len * r / 100)
        # print(f"k is {k}")
        top_r_idx = torch.topk(user_diff, k, largest=True).indices

        mask[start_idx:start_idx+effective_len][top_r_idx] = True
        mask[start_idx+effective_len:end_idx] = True

    return mask

def get_next_layer_k_diff_mask( # for getting first layer's top r% k deviation position mask.
    cached_k: torch.Tensor, # (B, N, D)
    compute_k: torch.Tensor, # (L, D)
    cached_lengths: torch.Tensor,
    x_offsets: torch.Tensor,
    r: int=20, 
):
    B, N, D = cached_k.shape
    L, D = compute_k.shape

    # transfer cached_k from (B, N, D) to (L, D), with cached_lengths and x_offsets
    cached_k_pad = torch.zeros(L, D)
    for i in range(B):
        effective_len = cached_lengths[i]
        target_start_idx = x_offsets[i]

        valid_ks = cached_k[i, :effective_len, :]

        cached_k_pad[target_start_idx: target_start_idx+effective_len, :] = valid_ks
    
    diff = torch.sum(torch.abs(cached_k_pad - compute_k), dim=1)

    mask = torch.zeros((B, N), dtype=torch.bool, device=compute_k.device)

    for user_idx in range(B):
        effective_len = cached_lengths[user_idx]

        print(f"effetive_len is {effective_len}")
        user_diff = diff[user_idx, :effective_len]
        k = int(effective_len * r / 100)
        print(f"k is {k}")
        top_r_idx = torch.topk(user_diff, k, largest=True).indices

        mask[user_idx, :effective_len][top_r_idx] = True
        mask[user_idx, effective_len:] = True

    return mask
# ignore end

def get_cached_lengths(
    cached_k: torch.Tensor,
):
    nonzero_mask = torch.all(cached_k != 0, dim=-1)
    lengths = nonzero_mask.sum(dim=-1)
    return lengths

def get_kv_pad_past(
    cached_k: torch.Tensor, 
    cached_v: torch.Tensor, 
    cached_lengths: torch.Tensor, 
    x_offsets: torch.Tensor, 
    L: int, 
    D: int,
    device: torch.device
):
    """
    将 `cached_k` 和 `cached_v` 根据 `cached_lengths` 和 `x_offsets` 转换为 (L, D) 形状的张量

    Args:
        cached_k (torch.Tensor): 形状为 (B, N, D) 的张量
        cached_v (torch.Tensor): 形状为 (L', D) 的张量
        cached_lengths (torch.Tensor): 形状为 (B,) 的张量，表示每个 batch 的有效 token 数量
        x_offsets (torch.Tensor): 形状为 (B,) 的张量，表示每个 batch 在目标输出中的起始位置
        L (int): 目标输出的长度
        D (int): 嵌入维度

    Returns:
        tuple: 返回填充后的 `cached_k_pad` 和 `cached_v_pad`，形状均为 (L, D)
    """
    # 初始化填充后的张量
    # begin_time = time.time()
    cached_k_pad = torch.zeros((L, D), device=cached_k.device)
    cached_v_pad = torch.zeros((L, D), device=cached_v.device)

    cached_v_start_idx = 0
    for i in range(cached_k.shape[0]):  # 遍历每个 batch
        effective_len = cached_lengths[i]
        target_start_idx = x_offsets[i]

        # 从 cached_k 和 cached_v 获取有效数据
        valid_ks = cached_k[i, :effective_len, :]
        valid_vs = cached_v[cached_v_start_idx:cached_v_start_idx+effective_len, :]

        # 填充到对应位置
        cached_k_pad[target_start_idx: target_start_idx+effective_len, :] = valid_ks
        cached_v_pad[target_start_idx: target_start_idx+effective_len, :] = valid_vs

        # 更新 v 的起始索引
        cached_v_start_idx += effective_len

    # print(f"get_kv_pad need {time.time()-begin_time:.6f} s")
    return cached_k_pad, cached_v_pad

def get_next_layer_kv_diff_mask_old(
    cached_k: torch.Tensor, # (B, N, D)
    cached_v: torch.Tensor, # (L', D)
    compute_k: torch.Tensor, # (L, D)
    compute_v: torch.Tensor, # (L, D)
    cached_lengths: torch.Tensor, # B
    x_offsets: torch.Tensor, # B + 1
    r: int=20,
):
    begin_time = time.time()
    L, D = compute_v.shape
    B, N, D = cached_k.shape

    # transfer cached_k from (B, N, D) to (L, D), with cached_lengths and x_offsets
    # transfer cached_v from (L', D) to (L, D), with cached_lengths and x_offsets
    cached_k_pad = torch.zeros((L, D),device=compute_k.device)
    cached_v_pad = torch.zeros((L, D),device=compute_v.device)
    cached_v_start_idx = 0
    for i in range(B):
        effective_len = cached_lengths[i]
        target_start_idx = x_offsets[i]

        valid_ks = cached_k[i, :effective_len, :]
        valid_vs = cached_v[cached_v_start_idx:cached_v_start_idx+effective_len, :]

        cached_k_pad[target_start_idx: target_start_idx+effective_len, :] = valid_ks
        cached_v_pad[target_start_idx: target_start_idx+effective_len, :] = valid_vs

        cached_v_start_idx += effective_len 
    
    diff_k = torch.sum(torch.abs(cached_k_pad - compute_k), dim=1)
    diff_v = torch.sum(torch.abs(cached_v_pad - compute_v), dim=1)
    # print(f"diff_k is {diff_k}\ndiff_v is {diff_v}")

    diff = diff_k+diff_v
    # print(f"diff is {diff}")

    mask = torch.zeros(L, dtype=torch.bool, device=compute_v.device)

    for user_idx in range(B):
        start_idx = x_offsets[user_idx]
        end_idx = x_offsets[user_idx+1]
        effective_len = cached_lengths[user_idx]

        # print(f"start_idx is {start_idx}, effetive_len is {effective_len}")

        user_diff = diff[start_idx:start_idx+effective_len]
        k = int(effective_len * r / 100)
        # print(f"k is {k}")
        top_r_idx = torch.topk(user_diff, k, largest=True).indices

        mask[start_idx:start_idx+effective_len][top_r_idx] = True
        mask[start_idx+effective_len:end_idx] = True

    print(f"get next layer kv diff mask need {time.time()-begin_time:.6f} s")
    return mask

def get_fusion_k( # get one layer's k with cached_k, x, w, mask, cached_lengths, x_offsets
    cached_k: torch.Tensor, # (B, N, D)
    x: torch.Tensor, # (L, d)
    w: torch.Tensor, # (d, D)
    mask: torch.Tensor, # (L)
    cached_lengths: torch.Tensor, # (B)
    x_offsets: torch.Tensor, # (B)
):
    # transfer cached_k from (B, N, D) to (L, D), with cached_lengths and x_offsets
    L = x.shape[0]
    D = w.shape[1]
    B = cached_k.shape[0]

    cached_k_pad = torch.zeros((L, D), device=x.device)
    for i in range(B):
        effective_len = cached_lengths[i]
        target_start_idx = x_offsets[i]
        valid_ks = cached_k[i, :effective_len, :]
        cached_k_pad[target_start_idx: target_start_idx+effective_len, :] = valid_ks

    print(f"cached_k_pad is {cached_k_pad}")

    new_k = cached_k_pad.clone()

    if mask.any():
        new_k[mask] = torch.mm(x[mask], w)
    
    return new_k

def get_kv_pad(
    cached_k: torch.Tensor, 
    cached_v: torch.Tensor, 
    cached_lengths: torch.Tensor, 
    x_offsets: torch.Tensor, 
    v_source_indices: torch.Tensor,
    v_target_indices: torch.Tensor,
    L: int, 
    D: int,
    device: torch.device
):
    cached_k_pad = torch.ops.fbgemm.jagged_to_padded_dense(
        cached_k.view(-1, D),
        [x_offsets],
        [L],
        padding_value=0.0
    )
    cached_v_pad = torch.zeros((L, D), device=device)
    cached_v_pad.index_copy_(0, v_target_indices, cached_v)
    return cached_k_pad, cached_v_pad

def get_fusion_kv_old( # get one layer's kv with cached_k,cached_v x, wk, wv, mask, cached_lengths, x_offsets
    cached_k: torch.Tensor, # (B, N, D)
    cached_v: torch.Tensor, # (L', D)
    x: torch.Tensor, # (L, d)
    wk: torch.Tensor, # (d, D)
    wv: torch.Tensor, # (d, D)
    mask: torch.Tensor, # (L)
    cached_lengths: torch.Tensor, # (B)
    x_offsets: torch.Tensor, # (B)
):
    begin_time = time.time()
    # transfer cached_k from (B, N, D) to (L, D), with cached_lengths and x_offsets
    L = x.shape[0]
    D = wk.shape[1]

    # transfer cached_k from (B, N, D) to (L, D), with cached_lengths and x_offsets
    # transfer cached_v from (L', D) to (L, D), with cached_lengths and x_offsets
    # v_target_indices = torch.cat([
    #     torch.arange(x_offsets[i], x_offsets[i]+cached_lengths[i], device=x.device)
    #     for i in range(cached_lengths.shape[0])
    # ])
    cached_k_pad, cached_v_pad = get_kv_pad(cached_k, cached_v, cached_lengths, x_offsets, L, D, x.device)
    get_kv_pad_end_time = time.time()

    # print(f"cached_k_pad is {cached_k_pad}\ncached_v_pad is {cached_v_pad}")

    recompute_time = time.time()
    new_k = cached_k_pad.clone()
    new_v = cached_v_pad.clone()

    if mask.any():
        new_k[mask] = torch.mm(x[mask], wk)
        new_v[mask] = torch.mm(x[mask], wv)
    
    print(f"get_kv_pad need {get_kv_pad_end_time-begin_time:.6f} s, get fusion kv need {time.time()-recompute_time:.6f} s")
    return new_k, new_v
    # return new_k, new_v, cached_k_pad, cached_v_pad

def get_same_masks(fusion, correct, cached, atol=1e-6):
    # 判断fusion_k和correct_k是否相同
    mask_fusion_correct = torch.isclose(fusion, correct, atol=atol).all(dim=1)  # 按行判断
    
    # 判断fusion_k和cached_k是否相同
    mask_fusion_cached = torch.isclose(fusion, cached, atol=atol).all(dim=1)  # 按行判断
    
    return mask_fusion_correct, mask_fusion_cached

def is_kv_correct(fusion_k, correct_k, cached_k, fusion_v, correct_v, cached_v, mask, atol=1e-6):
    # 获取k和v的mask
    mask_k_correct, mask_k_cached = get_same_masks(fusion_k, correct_k, cached_k, atol)
    mask_v_correct, mask_v_cached = get_same_masks(fusion_v, correct_v, cached_v, atol)
    
    # 打印掩码信息
    print(f"mask_k_correct: {mask_k_correct}\nmask_k_cached: {mask_k_cached}")
    print(f"mask_v_correct: {mask_v_correct}\nmask_v_cached: {mask_v_cached}")
    
    # 检查k和v是否与期望的mask匹配
    k_correct_check = torch.equal(mask_k_correct, mask) and torch.equal(mask_k_cached, ~mask)
    v_correct_check = torch.equal(mask_v_correct, mask) and torch.equal(mask_v_cached, ~mask)
    
    # 打印最终结果
    print(f"k is correct? {k_correct_check}")
    print(f"v is correct? {v_correct_check}")
    
    return k_correct_check, v_correct_check

def get_fusion_kv(
    cached_k: torch.Tensor, # (L', D)
    cached_v: torch.Tensor, # (L', D)
    x: torch.Tensor, # (L, d)
    wk: torch.Tensor, # (d, D)
    wv: torch.Tensor, # (d, D)
    recompute_mask: torch.Tensor, # (L)
    target_indices: torch.Tensor, # (L')
):
    L = recompute_mask.shape[0]
    D = wk.shape[1]
    cached_k_pad = torch.zeros((L, D), device=x.device, dtype=x.dtype)
    cached_v_pad = torch.zeros((L, D), device=x.device, dtype=x.dtype)

    cached_k_pad.index_copy_(0, target_indices, cached_k)
    cached_v_pad.index_copy_(0, target_indices, cached_v)

    # new_k = cached_k_pad.clone()
    # new_v = cached_v_pad.clone()

    if recompute_mask.any():
        cached_k_pad[recompute_mask] = torch.mm(x[recompute_mask], wk)
        cached_v_pad[recompute_mask] = torch.mm(x[recompute_mask], wv)

    # return cached_k_pad, cached_v_pad, new_k, new_v
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
    L, D = compute_k.shape

    cached_k_pad = torch.zeros((L, D), device=device, dtype=compute_k.dtype)
    cached_v_pad = torch.zeros((L, D), device=device, dtype=compute_v.dtype)

    cached_k_pad.index_copy_(0, target_indices, cached_k)
    cached_v_pad.index_copy_(0, target_indices, cached_v)

    diff_mask = torch.zeros((L, D), device=device, dtype=torch.bool)
    diff_mask[target_indices] = True
    recompute_mask = torch.ones(L, device=device, dtype=torch.bool)
    recompute_mask[target_indices] = False

    # print(f"cached_k_pad[diff_mask] is {cached_k_pad.shape}, compute_k[diff_mask] is {compute_k[diff_mask].reshape(-1, D).shape}")

    diff_k = torch.sum(torch.abs(cached_k_pad[diff_mask].reshape(-1, D) - compute_k[diff_mask].reshape(-1, D)), dim=1)
    diff_v = torch.sum(torch.abs(cached_v_pad[diff_mask].reshape(-1, D) - compute_v[diff_mask].reshape(-1, D)), dim=1)

    diff = diff_k + diff_v

    k = int(target_indices.shape[0] * r / 100)
    top_r_idx = torch.topk(diff, k, largest=True).indices
    recompute_mask[target_indices] = recompute_mask[target_indices].scatter(0, top_r_idx, torch.ones_like(top_r_idx, dtype=torch.bool))

    return recompute_mask
    

    