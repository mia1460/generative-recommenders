import torch
import torch.utils.bottleneck as bn
import fbgemm_gpu
import os
import datetime

# 假设以下是你的代码
# 初始化一些变量
rank = int(os.getenv("RANK", 0))
device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
# B = 128d
n = 211
num_heads = 8
linear_dim = 32
x_offsets = torch.tensor([    0,    38,   233,   279,   310,   365,   552,   752,   797,   850,
         1050,  1071,  1271,  1453,  1505,  1705,  1757,  1813,  1934,  1979,
         2100,  2150,  2172,  2198,  2237,  2304,  2504,  2630,  2652,  2807,
         3007,  3025,  3061,  3125,  3146,  3194,  3213,  3246,  3446,  3516,
         3598,  3646,  3668,  3697,  3767,  3794,  3994,  4053,  4189,  4222,
         4309,  4428,  4449,  4649,  4749,  4949,  4968,  4996,  5103,  5140,
         5207,  5259,  5299,  5340,  5371,  5565,  5613,  5687,  5887,  5915,
         6000,  6167,  6195,  6248,  6379,  6490,  6632,  6832,  6893,  6916,
         7116,  7159,  7359,  7439,  7639,  7839,  7864,  8064,  8085,  8285,
         8326,  8366,  8414,  8467,  8534,  8604,  8649,  8778,  8948,  9011,
         9047,  9067,  9086,  9108,  9137,  9174,  9212,  9237,  9257,  9276,
         9476,  9563,  9608,  9777,  9804,  9859,  9940,  9964,  9997, 10054,
        10074, 10098, 10118, 10143, 10161, 10361, 10437, 10637, 10689], device=device)
B: int = x_offsets.size(0) - 1
qk_attn = torch.randn(B, num_heads, n, n).cuda()
v = torch.randn(10689, num_heads*linear_dim).cuda()
assert qk_attn.is_cuda and v.is_cuda and x_offsets.is_cuda
torch.cuda.synchronize()

def your_function():
    L = x_offsets[-1].item()
    torch.cuda.synchronize()
    attn_output = torch.ops.fbgemm.dense_to_jagged(
        torch.einsum(
            "bhnm,bmhd->bnhd",
            qk_attn,
            torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]).reshape(
                B, n, num_heads, linear_dim
            ),
        ).reshape(B, n, num_heads * linear_dim),
        [x_offsets],
        L,
    )[0]
    return attn_output

def minimal_test():
    x = torch.tensor([100], device=device)
    L = x[-1].item()
    torch.cuda.synchronize()
    return L

def your_function_with_profile():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_dir = f'/home/yinj@/datas/grkvc/profile_logs/test_item/profile_{timestamp}'
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        prof.step()
        your_function()
        # minimal_test()
    print(f"profile logs saved at dir: {profile_dir}")

your_function_with_profile()
# your_function()

# 使用 bottleneck 工具分析代码
# bn.profile(your_function)