# for i in enumerate(2):
#     print(i)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '/home/yinj@/workplace/generative-recommenders'))


from utils import get_next_layer_padded_kv_recompute_mask
import torch

cached_k = torch.randn(3, 4, 5)
cached_v = torch.randn(3, 4, 5)
compute_k = torch.randn(3, 4, 5)
compute_v = torch.randn(3, 4, 5)

cached_mask = torch.tensor([
    [True, True, False, False],
    [True, False, False, False],
    [True, True, True, False],
], dtype=torch.bool)

r = 50

recompute_mask = get_next_layer_padded_kv_recompute_mask(cached_k, cached_v, compute_k, compute_v, cached_mask, r)

print(f"recompute_mask is {recompute_mask}")

if False:
    from utils import get_padded_fusion_kv
    import torch


    cached_k = torch.randn(3, 4, 256)
    cached_v = torch.randn(3, 4, 256)
    x = torch.randn(3, 4, 256)
    _uvqk = torch.randn(256, 1024)
    linear_dim = 32
    attention_dim = 32
    num_heads = 8

    recompute_mask = torch.tensor([
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ], dtype=torch.bool)
    print(f"recompute_mask.dtype is {recompute_mask.dtype}")

    batched_mm_output = torch.matmul(x, _uvqk)
    _, v, _, k = torch.split(
        batched_mm_output,
        [
            linear_dim * num_heads,
            linear_dim * num_heads,
            attention_dim * num_heads,
            attention_dim * num_heads,
        ],
        dim=2
    )

    print(f"v.shape is {v.shape}")

    fusion_k, fusion_v = get_padded_fusion_kv(cached_k, cached_v, x, _uvqk, linear_dim, attention_dim, num_heads, recompute_mask)

    print(f"cached_k is {cached_k}\nfusion_k is {fusion_k}\nk is {k}\ncached_v is {cached_v}\nfusion_v is {fusion_v}\nv id {v}\n")




if False:
    from utils import get_fusion_k, get_cached_lengths, get_fusion_kv, get_same_masks, is_kv_correct
    import torch
    # cached_k = torch.randn(4, 5, 5)
    cached_k = torch.randn(11, 5)
    cached_v = torch.randn(11, 5)
    print(f"cached_k is {cached_k}\ncached_v is {cached_v}")
    # print(f"cached_k before zeros is {cached_k}")
    cached_lengths = torch.tensor([2, 2, 3, 4], dtype=torch.int)
    # for i in range(cached_k.shape[0]):
    #     valid_length = cached_lengths[i]
    #     cached_k[i, valid_length:] = 0.0
    # cached_lengths = get_cached_lengths(cached_k)
    # print(f"cached_k after zeros is {cached_k}")

    x = torch.randn(15, 3)
    wk = torch.randn(3, 5)
    wv = torch.randn(3, 5)
    mask = torch.tensor([False,  True,  True, False,  True,  True,  True, False, False,  True,
             True, False, False,  True,  True], dtype=torch.bool)
    x_offsets = torch.tensor([0, 3, 6, 10, 15], dtype=torch.int)
    target_indices = torch.cat([
        torch.arange(x_offsets[i], x_offsets[i]+cached_lengths[i]) for i in range(cached_lengths.shape[0])
    ])
    recompute_mask = torch.ones(15, device=x.device, dtype=torch.bool)
    recompute_mask[target_indices] = False
    # recompute_mask=mask
    print(f"recompute_mask is {recompute_mask}\ntarget_indices is {target_indices}")

    print(f"x is {x}\nwk is {wk}\nwv is {wv}")
    # fusion_k = get_fusion_k(cached_k=cached_k, x=x, w=w, mask=mask, cached_lengths=cached_lengths, x_offsets=x_offsets)
    # fusion_k, fusion_v = get_fusion_kv(cached_k=cached_k, cached_v=cached_v, x=x, wk=wk, wv=wv, mask=mask, cached_lengths=cached_lengths, x_offsets=x_offsets)
    # fusion_k, fusion_v, cached_k, cached_v = get_fusion_kv(cached_k=cached_k, cached_v=cached_v, x=x, wk=wk, wv=wv, mask=mask, cached_lengths=cached_lengths, x_offsets=x_offsets)
    # fusion_k, fusion_v, cached_k_pad, cached_v_pad = get_fusion_kv(cached_k=cached_k, cached_v=cached_v, x=x, wk=wk, wv=wv, recompute_mask=recompute_mask, target_indices=target_indices)
    fusion_k, fusion_v = get_fusion_kv(cached_k=cached_k, cached_v=cached_v, x=x, wk=wk, wv=wv, recompute_mask=recompute_mask, target_indices=target_indices)
    print(f"fusion_k is {fusion_k}\nfusion_v is {fusion_v}")
    correct_k = torch.mm(x, wk)
    correct_v = torch.mm(x, wv)
    print(f"correct_k is {correct_k}\ncorrect_v is {correct_v}")

    k_correct, v_correct = is_kv_correct(fusion_k, correct_k, cached_k_pad, fusion_v, correct_v, cached_v_pad, recompute_mask)

if False:
    from utils import get_cached_lengths
    import torch

    cached_k = torch.randn(4, 5, 5)
    print(f"cached_k before zeros is {cached_k}")
    is_zero = torch.rand(4, 5) < 0.2
    cached_k[is_zero] = 0.0
    print(f"cached_k after zeros is {cached_k}")

    cached_lengths = get_cached_lengths(cached_k)
    print(f"cached_lengths is {cached_lengths}")


if False:
    from utils import get_next_layer_kv_diff_mask, get_cached_lengths
    import torch

    # cached_k = torch.randn(4, 5, 5)
    cached_k = torch.randn(11, 5)
    cached_v = torch.randn(11, 5)

    compute_v = torch.randn(15, 5)
    compute_k = torch.randn(15, 5)
    mask = torch.tensor([False,  True,  True, False,  True,  True,  True, False, False,  True,
             True, False, False,  True,  True], dtype=torch.bool)
    x_offsets = torch.tensor([0, 3, 6, 10, 15], dtype=torch.int)
    cached_lengths = torch.tensor([2, 2, 3, 4], dtype=torch.int)
    target_indices = torch.cat([
        torch.arange(x_offsets[i], x_offsets[i]+cached_lengths[i]) for i in range(cached_lengths.shape[0])
    ])
    # recompute_mask = torch.ones(15, device=compute_k.device, dtype=torch.bool)
    # recompute_mask[target_indices] = False

    # simulate the real condition of k
    # for i in range(cached_k.shape[0]):
    #     valid_length = cached_lengths[i]
    #     cached_k[i, valid_length:] = 0.0
    
    # cached_lengths = get_cached_lengths(cached_k)
    print(f"cached_lengths is {cached_lengths}\ntarget_indices is {target_indices}")

    # x_offsets = torch.tensor([0, 3, 6, 10, 15], dtype=torch.int)
    print(f"cached_k is {cached_k}\ncompute_k is {compute_k}\ncached_v is {cached_v}\ncompute_v is {compute_v}")

    mask = get_next_layer_kv_diff_mask(cached_k=cached_k, cached_v=cached_v, compute_k=compute_k, compute_v=compute_v, target_indices=target_indices, device=compute_k.device, r=50)
    print(f"mask is {mask}")


if False:
    from utils import get_next_layer_v_diff_mask, get_next_layer_k_diff_mask

    import torch

    cached_v = torch.randn(15, 5)
    compute_v = torch.randn(15, 5)
    cached_lengths = torch.tensor([2, 2, 3, 4], dtype=torch.int)  
    x_offsets = torch.tensor([0, 3, 6, 10, 15], dtype=torch.int)  

    print(f"cached_v is {cached_v}\n compute_v is {compute_v}")
    v_mask = get_next_layer_v_diff_mask(cached_v=cached_v, compute_v=compute_v, cached_lengths=cached_lengths, x_offsets=x_offsets, r=50)
    print(f"v_mask is {v_mask}")

    cached_k = torch.randn(4, 5, 5)
    compute_k = torch.randn(4, 5, 5)

    print(f"cached_k is {cached_k}\n compute_k is {compute_k}")
    k_mask = get_next_layer_k_diff_mask(cached_k=cached_k, compute_k=compute_k, cached_lengths=cached_lengths, r=50)
    print(f"k_mask is {k_mask}")

    import time
    # masked_normed_x = normed_x * mask.float()
    x = torch.randn(20000, 50)
    w = torch.randn(50,50)
    mask = torch.zeros(20000, dtype=bool)
    num_true = mask.numel() // 2  # 计算需要多少个 True
    indices = torch.randperm(mask.numel())[:num_true]  # 随机选择 50% 的位置
    mask[indices] = True
    st = time.time()
    ans = torch.mm(x, w)
    et = time.time()
    print(f"full time is {et-st}")
    x = x[mask]
    st = time.time()
    ans = torch.mm(x, w)
    et = time.time()
    print(f"zero time is {et-st}")

if False:
    from utils import compute_one_token_diff, get_top_k_pos_mask, get_fusion_cache, get_batch_fusion_k_cache,get_batch_fusion_v_cache

    import torch

    tensor = torch.zeros(10, dtype=torch.float32)
    print(tensor.element_size() * tensor.nelement())

    tensor1 = torch.tensor([[1.0, 2.0, 3.0]])
    tensor2 = torch.tensor([[4.0, 2.5, 1.5]])

    print(f"tensor1.shape is {tensor1.shape}")

    print(f"the diff of tensor1 & tensor2 is {compute_one_token_diff(tensor1, tensor2)}")

    diff_tensor = torch.tensor([1, 2, 5, 6, 2, 3, 5, 6, 7, 2, 7, 9, 3, 9, 10])
    print(f"diff_tensor.shape is {diff_tensor.shape}")

    print(f"the top_r_mask is {get_top_k_pos_mask(diff_tensor, 3)}")

    cached_tensor = torch.randn(10, 3)
    compute_tensor = torch.randn(10, 3)
    print(f"cached_tensor is {cached_tensor}\ncompute_tensor is {compute_tensor}")

    pos_mask = torch.tensor([False,  True,  True, True,  True, False, True,  True, False, False])

    ret_tensor, pos_mask = get_fusion_cache(cached_tensor, compute_tensor, last_layer_mask = pos_mask, r=50)
    print(f"ret_tensor is {ret_tensor}\npos_mask is {pos_mask}")

    cached_k = torch.randn(2,3,5)
    padded_k = torch.randn(2,3,5)
    print(f"cached_k is {cached_k}\npadded_k is {padded_k}")
    # last_layer_mask = torch.tensor([
    #     [True, True, False],
    #     [True, False, False]
    # ])
    last_layer_mask = torch.tensor([
        [False, False, False],
        [False, False, False]
    ])

    ret_fusion_k, ret_layer_mask = get_batch_fusion_k_cache(
        cached_k=cached_k,
        padded_k=padded_k,
        last_layer_k_mask=last_layer_mask,
        r=50
    )

    print(f"ret_fusion_k is {ret_fusion_k}\nret_layer_mask is {ret_layer_mask}")

    cached_v = torch.randn(3,5)
    padded_v = torch.randn(3,5)
    print(f"cached_v is {cached_v}\npadded_v is {padded_v}")
    last_layer_mask = torch.tensor([True, True, True])
    # last_layer_mask = torch.tensor([False, False, False])
    past_lengths = torch.tensor([2, 1])

    ret_fusion_v, ret_layer_mask = get_batch_fusion_v_cache(
        cached_v=cached_v,
        padded_v=padded_v,
        last_layer_v_mask=last_layer_mask,
        past_lengths=past_lengths,
        N=3,
        r=50,
    )

    print(f"ret_fusion_v is {ret_fusion_v}\nret_layer_mask is {ret_layer_mask}")
    print(f"cached_v is {cached_v}\npadded_v is {padded_v}")

    ret_fusion_v, ret_layer_mask = get_batch_fusion_v_cache(
        cached_v=cached_v,
        padded_v=padded_v,
        last_layer_v_mask=ret_layer_mask,
        past_lengths=past_lengths,
        N=3,
        r=50,
    )

    print(f"ret_fusion_v is {ret_fusion_v}\nret_layer_mask is {ret_layer_mask}")
