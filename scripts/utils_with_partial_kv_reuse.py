import torch
# torch.ops.load_library('/home/yinj@/tools/miniconda3/envs/grKVCPy310/lib/python3.10/site-packages/fbgemm_gpu/fbgemm_gpu_py.so')
from collections import OrderedDict
import time
import datetime
from generative_recommenders.research.data.eval import (
    _avg,
    add_to_summary_writer,
    eval_metrics_v2_from_tensors,
    get_eval_state,
)
from generative_recommenders.research.modeling.sequential.features import (
    movielens_seq_features_from_row,
)

# import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

def get_state_dict(
    ckpt_path: str,
    device: torch.device,
):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    fixed_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("module."):
            new_key = new_key.replace("module.", "")
        if "_embedding_" in new_key and "_item_emb" in new_key:
            new_key = new_key.replace("_embedding_", "_embedding_module.")
        fixed_state_dict[new_key] = v   

    return fixed_state_dict     

def get_filtered_config(
    config_path: str,
    filter_prefix: str,
):
    with open(config_path, "r") as f:
        lines = f.readlines()

    filtered_config = [line for line in lines if line.strip().startswith(filter_prefix)]

    return filtered_config

def print_eval_metrics(
        eval_dict: dict, 
        # epoch: int,
        # rank: int,
        world_size: int,
        metrics: list = ["ndcg@10", "ndcg@50", "hr@10", "hr@50", "mrr"]
):
    """
    通用评估指标打印函数（带缓存版本）
    
    参数：
    eval_dict -- 包含各指标张量列表的字典，结构为 {metric_name: [tensor1, tensor2...]}
    epoch -- 当前epoch数
    rank -- 当前进程的rank编号
    world_size -- 分布式训练中的总进程数
    metrics -- 需要计算的指标列表，默认包含常用推荐系统指标
    """
    # 拼接张量维度
    processed_dict = {k: torch.cat(v, dim=-1) for k, v in eval_dict.items()}
    
    # 计算聚合指标
    metric_results = {
        name: _avg(processed_dict[name], world_size=world_size)
        for name in metrics
    }
    
    # 构建格式化输出
    metric_str = ", ".join([f"{k.upper()} {v:.4f}" for k, v in metric_results.items()])
    # print(f"rank {rank}: eval @ epoch {epoch}, metrics are:\n{metric_str}")
    print(f"metrics are: {metric_str}")
    
    # return metric_results  # 可选：返回计算结果供后续使用

def load_ckpt(
    ckpt_path: str, 
    model,
    device: torch.device,
):
    state_dict = get_state_dict(ckpt_path, device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print("Error in loading model state dict:", e)

    print(f"loading checkpoint from: {ckpt_path}")

def get_base_cache_and_lengths(
    data_loader,
    device,
    gr_output_length,
    eval_state,
    model,
    eval_batch_size,
    main_module_bf16,
    world_size
):
    start_time = time.time()
    print(f"begin saving base cache list and cached lengths list...")
    base_cache_list = []
    cached_lengths_list = []
    eval_dict_all = None
    for eval_iter, row in enumerate(iter(data_loader)):
        print(f"-saving the {eval_iter}'s cache and past_lengths...")
        seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
            row, device=device, max_output_length=gr_output_length + 1
        )
        eval_dict, updated_cache = eval_metrics_v2_from_tensors(
            eval_state,
            model,
            seq_features,
            target_ids=target_ids,
            target_ratings=target_ratings,
            user_max_batch_size=eval_batch_size,
            dtype=torch.bfloat16 if main_module_bf16 else None,
            cache=None,
            return_cache_states=True,
            selective_reuse=False,
        )  
        # base_dir = "/home/yinj@/datas/grkvc/cached_uvqk"
        # saved_path = os.path.join(base_dir, "saved_cache_with_ckpt_" + os.path.basename(checkpoint_path))
        # torch.save(updated_cache, saved_path)
        # print(f"cache{updated_cache[0][0].shape} saved at {saved_path}")
        cached_k = [cached_ks[2] for cached_ks in updated_cache]
        cached_v = [cached_vs[0] for cached_vs in updated_cache]  
        base_cache_list.append([(cached_v[i].cpu(), None, cached_k[i].cpu(), None) for i in range(model._num_blocks)])
        cached_lengths_list.append(seq_features.past_lengths.cpu())
        if eval_dict_all is None:
            eval_dict_all = {}
            for k, v in eval_dict.items():
                eval_dict_all[k] = []
        for k, v in eval_dict.items():
            eval_dict_all[k] = eval_dict_all[k] + [v]
        del eval_dict
    end_time = time.time()
    print(f"saved base cache list and cached lengths list need: {end_time - start_time:.2f}s")
    print_eval_metrics(
        eval_dict=eval_dict_all,
        world_size=world_size,
    )
    return base_cache_list, cached_lengths_list

def save_base_cache_and_lengths(
    data_loader,
    device,
    gr_output_length,
    eval_state,
    model,
    eval_batch_size,
    main_module_bf16,
    world_size,
    base_cache_path,
    cached_lengths_path,
):
    base_cache_list, cached_lengths_list = get_base_cache_and_lengths(
        data_loader=data_loader,
        device=device,
        gr_output_length=gr_output_length,
        eval_state=eval_state,
        model=model,
        eval_batch_size=eval_batch_size,
        main_module_bf16=main_module_bf16,
        world_size=world_size
    )
    torch.save(base_cache_list, base_cache_path)
    torch.save(cached_lengths_list, cached_lengths_path)
    print(f"base_cache_list saved at {base_cache_path}, cached_lengths_list saved at {cached_lengths_path}")

def run_an_e2e(
    cache_use_type: str, # ["no", "fully", "selective"]
    data_loader,
    device,
    gr_output_length,
    eval_state,
    model,
    eval_batch_size,
    main_module_bf16,
    world_size,
    enable_profiler=False,  # 新增的参数，用于控制是否启用 profiler
    base_cache_list=None,
    cached_lengths_list=None,
    recompute_ratio=20,
    return_cache_states=False,
):
    selective_reuse = True

    if cache_use_type == "no":
        selective_reuse = False
    elif cache_use_type == "fully":
        assert base_cache_list is not None and cached_lengths_list is not None, \
            f"base_cache_list: {base_cache_list}, cached_lengths_list: {cached_lengths_list}"
        selective_reuse = True
        recompute_ratio = 0
    elif cache_use_type == "selective":
        assert base_cache_list is not None and cached_lengths_list is not None, \
            f"base_cache_list: {base_cache_list}, cached_lengths_list: {cached_lengths_list}" 
        selective_reuse = True
        recompute_ratio = recompute_ratio
    else:
        print(f"[error] wrong cache_use_type in run_an_e2e")
    
    start_time = time.time()
    print(f"============== begin evaling use {cache_use_type} cache...==============")
    if cache_use_type == "selective":
        print(f"!!!recompute_ratio is {recompute_ratio}!!!")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_dir = f'/home/yinj@/datas/grkvc/profile_logs/profile_{timestamp}'
    
    if enable_profiler:  # 如果启用 profiler，则开始 profiler
        print(f"!!!!!!!!!!!!!!!!! profile begin !!!!!!!!!!!!!!!!!!")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            eval_dict_all = None
            for eval_iter, row in enumerate(iter(data_loader)):
                # print(f"-evaling the iter@{eval_iter}...")
                seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                    row, device=device, max_output_length=gr_output_length + 1
                )
                with torch.autograd.profiler.record_function(f"{cache_use_type}_cache"):
                    if return_cache_states:
                        eval_dict, updated_cache = eval_metrics_v2_from_tensors(
                            eval_state,
                            model,
                            seq_features,
                            target_ids=target_ids,
                            target_ratings=target_ratings,
                            user_max_batch_size=eval_batch_size,
                            dtype=torch.bfloat16 if main_module_bf16 else None,
                            cache=base_cache_list[eval_iter] if cache_use_type != "no" else None,
                            cached_lengths=cached_lengths_list[eval_iter] if cache_use_type != "no" else None,
                            return_cache_states=return_cache_states,
                            selective_reuse=selective_reuse,
                            recompute_ratio=recompute_ratio,
                        )  
                    else:
                        eval_dict = eval_metrics_v2_from_tensors(
                            eval_state,
                            model,
                            seq_features,
                            target_ids=target_ids,
                            target_ratings=target_ratings,
                            user_max_batch_size=eval_batch_size,
                            dtype=torch.bfloat16 if main_module_bf16 else None,
                            cache=base_cache_list[eval_iter] if cache_use_type != "no" else None,
                            cached_lengths=cached_lengths_list[eval_iter] if cache_use_type != "no" else None,
                            return_cache_states=return_cache_states,
                            selective_reuse=selective_reuse,
                            recompute_ratio=recompute_ratio,
                        )            
                if eval_dict_all is None:
                    eval_dict_all = {}
                    for k, v in eval_dict.items():
                        eval_dict_all[k] = []
                for k, v in eval_dict.items():
                    eval_dict_all[k] = eval_dict_all[k] + [v]
                del eval_dict
                torch.cuda.synchronize()
                prof.step()
        print(f"profile logs saved at dir: {profile_dir}")
    else:  # 如果没有启用 profiler，则直接执行 eval
        eval_dict_all = None
        total_time = 0
        for eval_iter, row in enumerate(iter(data_loader)):
            # print(f"-evaling the iter@{eval_iter}...")
            seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                row, device=device, max_output_length=gr_output_length + 1
            )
            # print(f"seq_features.past_embeddings is {seq_features.past_embeddings}")
            with torch.autograd.profiler.record_function(f"{cache_use_type}_cache"):
                # print(f"cached_k[0][46] is {base_cache_list[0][-1][2][47]}")
                if return_cache_states:
                    eval_dict, updated_cache, encode_time= eval_metrics_v2_from_tensors(
                        eval_state,
                        model,
                        seq_features,
                        target_ids=target_ids,
                        target_ratings=target_ratings,
                        user_max_batch_size=eval_batch_size,
                        dtype=torch.bfloat16 if main_module_bf16 else None,
                        cache=base_cache_list[eval_iter] if cache_use_type != "no" else None,
                        cached_lengths=cached_lengths_list[eval_iter] if cache_use_type != "no" else None,
                        # cache=base_cache_list[eval_iter], # for testing kv cache same
                        # cached_lengths=cached_lengths_list[eval_iter], # for testing kv cache same
                        return_cache_states=return_cache_states,
                        selective_reuse=selective_reuse,
                        recompute_ratio=recompute_ratio,
                        return_encode_time=True,
                    )  
                else:
                    eval_dict, encode_time = eval_metrics_v2_from_tensors(
                        eval_state,
                        model,
                        seq_features,
                        target_ids=target_ids,
                        target_ratings=target_ratings,
                        user_max_batch_size=eval_batch_size,
                        dtype=torch.bfloat16 if main_module_bf16 else None,
                        cache=base_cache_list[eval_iter] if cache_use_type != "no" else None,
                        cached_lengths=cached_lengths_list[eval_iter] if cache_use_type != "no" else None,
                        return_cache_states=return_cache_states,
                        selective_reuse=selective_reuse,
                        recompute_ratio=recompute_ratio,
                        return_encode_time=True
                    )
                total_time += encode_time            
            if eval_dict_all is None:
                eval_dict_all = {}
                for k, v in eval_dict.items():
                    eval_dict_all[k] = []
            for k, v in eval_dict.items():
                eval_dict_all[k] = eval_dict_all[k] + [v]
            del eval_dict
            if eval_iter == 0:
                break
            # torch.cuda.synchronize()

    end_time = time.time()
    if enable_profiler:
        print(f"eval use {cache_use_type} cache need: {end_time - start_time:.2f}s")
    else:
        print(f"eval use {cache_use_type} cache need: {total_time:.2f} ms")
    print_eval_metrics(
        eval_dict=eval_dict_all,
        world_size=world_size,
    )
