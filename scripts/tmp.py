import os
import sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '/home/yinj@/workplace/generative-recommenders'))

# from generative_recommenders.ops.triton.triton_jagged import _get_bmm_configs

# configs = _get_bmm_configs()
# print(f"return configs from _get_bmm_configs() is {configs}")

# pos_emb = torch.nn.Embedding(211, 256)
# print(torch.arange(211).unsqueeze(0).repeat(128, 1).shape)
# print(pos_emb(
#     torch.arange(211).unsqueeze(0).repeat(128, 1)
# ))

# bucketization_fn = lambda x: (
#     torch.log(torch.abs(x).clamp(min=1)) / 0.301
# ).long()
# N = 8
# num_buckets = 128
# all_timestamps = torch.tensor([974698514, 974698754, 974698754, 974698754, 974698754,974698754,         0,         0])
# ext_timestamps = torch.tensor([974698514, 974698754, 974698754, 974698754, 974698754,974698754,         0,         0, 0])
# print(f"ext_timestamps@{ext_timestamps.shape}[0] is {ext_timestamps[0]}")
# print(f"minus before is {ext_timestamps[1:].unsqueeze(1)}, after is {ext_timestamps[:-1].unsqueeze(0)}, answer is {ext_timestamps[1:].unsqueeze(1) - ext_timestamps[:-1].unsqueeze(0)}")
# # causal masking. Otherwise [:, :-1] - [:, 1:] works
# bucketed_timestamps = torch.clamp(
#     bucketization_fn(
#         ext_timestamps[1:].unsqueeze(1) - ext_timestamps[:-1].unsqueeze(0)
#     ),
#     min=0,
#     max=num_buckets,
# ).detach()
# print(f"bucketed_timestamp@{bucketed_timestamps.shape} is {bucketed_timestamps}")

# # 示例输入
# x = torch.tensor([974698514, 974698754, 974698754, 974698754, 974698754,974698754,         0,         0,])
# result = bucketization_fn(x)
# print(result)

# x = torch.randn((3, 4, 5))
# print(x)

# row_idx = torch.arange(4).unsqueeze(0)
# cached_lengths = torch.tensor([1, 2, 3])
# past_lengths = torch.tensor([3, 3, 4])
# cached = cached_lengths.view(3, 1)
# past = past_lengths.view(3, 1)

# mask = (row_idx >= cached) & (row_idx <= past)
# print(mask)

# x = x * mask.unsqueeze(-1)

# # delta_x_offsets = torch.tensor([[2], [2], [3]])
# # for b in range(3):
# #     indicies = delta_x_offsets[b]
# #     if indicies:
# #         new_x = torch.zeros((1, 4, 5))
# #         new_x[0, indicies, :] = x[b, indicies, :]
# #         x[b, :, :] = new_x

# print(x)


# def merge_kv(cached_k, cached_v, new_k, new_v, mask):
#     B, N, D = cached_k.shape
#     M = 90

no_cache_path = '/home/yinj@/datas/grkvc/return_encoded_embs/1281_with_no_cache_20250308_180805.pt'
fully_cache_path = '/home/yinj@/datas/grkvc/return_encoded_embs/1281_with_fully_cache_20250308_180811.pt'

no_cache = torch.load(no_cache_path)
fully_cache = torch.load(fully_cache_path)

equal_mask = torch.isclose(no_cache, fully_cache, atol=1e-7)

print(f"is equal? {equal_mask.all()}")