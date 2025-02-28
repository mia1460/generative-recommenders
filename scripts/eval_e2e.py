import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide excessive tensorflow debug messages
os.environ['NUMEXPR_MAX_THREADS'] = '32'
# import numexpr
import random
import time
from datetime import date
from typing import Dict, Optional
import gin
import torch
# torch.ops.load_library('/home/yinj@/tools/miniconda3/envs/grKVCPy310/lib/python3.10/site-packages/fbgemm_gpu/fbgemm_gpu_py.so')
import torch.distributed as dist
import sys
import fbgemm_gpu  # noqa: F401, E402
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

sys.path.append(os.path.join(os.path.dirname(__file__), '/home/yinj@/workplace/generative-recommenders'))

from generative_recommenders.research.data.eval import (
    _avg,
    add_to_summary_writer,
    eval_metrics_v2_from_tensors,
    get_eval_state,
)
from generative_recommenders.research.data.reco_dataset import get_reco_dataset
from generative_recommenders.research.indexing.utils import get_top_k_module
from generative_recommenders.research.modeling.sequential.autoregressive_losses import (
    BCELoss,
    InBatchNegativesSampler,
    LocalNegativesSampler,
)
from generative_recommenders.research.modeling.sequential.embedding_modules import (
    EmbeddingModule,
    LocalEmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.encoder_utils import (
    get_sequential_encoder,
)
from generative_recommenders.research.modeling.sequential.features import (
    movielens_seq_features_from_row,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders.research.modeling.sequential.losses.sampled_softmax import (
    SampledSoftmaxLoss,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
    LayerNormEmbeddingPostprocessor,
)
from generative_recommenders.research.modeling.similarity_utils import get_similarity_function
from generative_recommenders.research.trainer.data_loader import create_data_loader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from generative_recommenders.research.modeling.cache.UsersCache import (
    UsersCache,
)
from generative_recommenders.research.data.dataset import DatasetV2, MultiFileDatasetV2
# import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

from utils import load_ckpt, get_filtered_config, print_eval_metrics, save_base_cache_and_lengths, run_an_e2e



# [superparams]
config_file = '/home/yinj@/workplace/generative-recommenders/configs/ml-20m/hstu-sampled-softmax-n128-large-final.gin'
dataset_name = "ml-20m"
max_sequence_length = 200
local_batch_size = 128
main_module = "HSTU"
dropout_rate = 0.2
user_embedding_norm = "l2_norm"
num_epochs = 101
item_embedding_dim = 256
learning_rate = 1e-3
weight_decay = 0
num_warmup_steps = 0
interaction_module_type = "DotProduct"
top_k_method = "MIPSBruteForceTopK"
loss_module = "SampledSoftmaxLoss"
num_negatives = 128
sampling_strategy = "local"
temperature = 0.05
item_l2_norm = True
l2_norm_eps = 1e-6
enable_tf32 = True
main_module_bf16 = False

world_size = torch.cuda.device_count()
rank = int(os.getenv("RANK", 0))
device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
max_item_id=131262
gr_output_length=10
all_item_ids = [x + 1 for x in range(max_item_id)]

# for terminal the epochs in advance
TEST_SIGNAL = True
TEST_EPOCH_BREAK_NUM=1

# for kv cache
USE_KV_CACHE = True
# EVAL_BATCH_SIZE=local_batch_size # when use_kv_cache & wanna an epoch for all users
USE_0_CACHE=True

# for ckpt config
USE_CKPT_FILE = True

# for cacheblend
RECOMPUTE_RATIO=20

# for save base_cache_list and cached_lengths_list
NEED_SAVE_BC = False

# [eval_body]
# load config
filtered_config = get_filtered_config(config_file, "hstu_encoder")
gin.parse_config(filtered_config)

# create eval datasets
eval_base_path = '/home/yinj@/datas/grkvc/use_data/ml_20m_sasrec_format_by_user_test_loss_last5.csv'
eval_delta_path = '/home/yinj@/datas/grkvc/use_data/ml_20m_sasrec_format_by_user_test.csv'
eval_base_dataset = DatasetV2(
    ratings_file=eval_base_path,
    padding_length = max_sequence_length + 1,
    ignore_last_n=1,
    chronological=True,
    sample_ratio=1.0
)
eval_delta_dataset = DatasetV2(
    ratings_file=eval_delta_path,
    padding_length = max_sequence_length + 1,
    ignore_last_n=1,
    chronological=True,
    sample_ratio=1.0
)

# create eval sampler&loader
eval_base_sampler, eval_base_loader = create_data_loader(
    eval_base_dataset,
    batch_size=local_batch_size,
    world_size=world_size,
    rank=rank,
    shuffle=False,
    drop_last=False
)
eval_delta_sampler, eval_delta_loader = create_data_loader(
    eval_delta_dataset,
    batch_size=local_batch_size,
    world_size=world_size,
    rank=rank,
    shuffle=False,
    drop_last=False
)

# create item embedding table
embedding_module = LocalEmbeddingModule(
    num_items=max_item_id,
    item_embedding_dim=item_embedding_dim,
)

# create interaction module
interaction_module, interaction_module_debug_str = get_similarity_function(
    module_type=interaction_module_type,
    query_embedding_dim=item_embedding_dim,
    item_embedding_dim=item_embedding_dim,
)

# create input preproc module
input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
    max_sequence_len=max_sequence_length + gr_output_length + 1,
    embedding_dim=item_embedding_dim,
    dropout_rate=dropout_rate,
)

# create output postproc module
output_postproc_module = (
    L2NormEmbeddingPostprocessor(
        embedding_dim=item_embedding_dim,
        eps=1e-6,
    )
    if user_embedding_norm == "l2_norm"
    else LayerNormEmbeddingPostprocessor(
        embedding_dim=item_embedding_dim,
        eps=1e-6,
    )
)

# create model -- hstu_encoder
model = get_sequential_encoder(
    module_type=main_module,
    max_sequence_length=max_sequence_length,
    max_output_length=gr_output_length + 1,
    embedding_module=embedding_module,
    interaction_module=interaction_module,
    input_preproc_module=input_preproc_module,
    output_postproc_module=output_postproc_module,
    verbose=True,
)
model_debug_str = model.debug_str()

# create negatives sampler
if sampling_strategy == "in-batch":
    negatives_sampler = InBatchNegativesSampler(
        l2_norm=item_l2_norm,
        l2_norm_eps=l2_norm_eps,
        dedup_embeddings=True,
    )
    sampling_debug_str = (
        f"in-batch{f'-l2-eps{l2_norm_eps}' if item_l2_norm else ''}-dedup"
    )
elif sampling_strategy == "local":
    negatives_sampler = LocalNegativesSampler(
        num_items=max_item_id,
        item_emb=model._embedding_module._item_emb,
        all_item_ids=all_item_ids,
        l2_norm=item_l2_norm,
        l2_norm_eps=l2_norm_eps,
    )
else:
    raise ValueError(f"Unrecognized sampling strategy {sampling_strategy}.")
sampling_debug_str = negatives_sampler.debug_str()

# move model and others need to GPU
if main_module_bf16:
    model = model.to(torch.bfloat16)
model = model.to(device)
negatives_sampler = negatives_sampler.to(device)

# init logging
date_str = date.today().strftime("%Y-%m-%d")
model_subfolder = f"{dataset_name}-l{max_sequence_length}"
model_desc = (
        f"{model_subfolder}"
        + f"/{model_debug_str}_{interaction_module_debug_str}_{sampling_debug_str}"
        + f"{f'-ddp{world_size}' if world_size > 1 else ''}-b{local_batch_size}-lr{learning_rate}-wu{num_warmup_steps}-wd{weight_decay}{'' if enable_tf32 else '-notf32'}-{date_str}"
    )
log_dir = f"/home/yinj@/datas/grkvc/logs/{model_desc}"
if rank == 0:
    writer = SummaryWriter(log_dir=log_dir)
    logging.info(f"Rank {rank}: writing logs to {log_dir}")
else:
    writer = None
    logging.info(f"Rank {rank}: disabling summary writer")


# load ckpt
base_ckpt_path='/home/yinj@/datas/grkvc/ckpts/ml-20m/model_base.ckpt'
model_1_path='/home/yinj@/datas/grkvc/ckpts/ml-20m/model_1.ckpt'

ckpt_prefix = '/home/yinj@/datas/grkvc/ckpts/ml-20m/model_'

load_ckpt(
    ckpt_path = base_ckpt_path,
    model = model,
    device = device
)

# model.eval()
model.eval()

# get eval_state
eval_state = get_eval_state(
    model=model,
    all_item_ids=all_item_ids,
    negatives_sampler=negatives_sampler,
    top_k_module_fn=lambda item_embeddings, item_ids: get_top_k_module(
        top_k_method=top_k_method,
        model=model,
        item_embeddings=item_embeddings,
        item_ids=item_ids,
    ),
    device=device,
    float_dtype=torch.bfloat16 if main_module_bf16 else None,
)

base_cache_path='/home/yinj@/datas/grkvc/base_cache_and_cached_lengths/base_cache_list.pt'
cached_lengths_path='/home/yinj@/datas/grkvc/base_cache_and_cached_lengths/cached_lengths_list.pt'
print(f"eval_batch_size is {local_batch_size}")
if NEED_SAVE_BC:
    save_base_cache_and_lengths(data_loader=eval_base_loader, device=device, gr_output_length=gr_output_length, eval_state=eval_state, model=model, eval_batch_size=local_batch_size, main_module_bf16=main_module_bf16, world_size=world_size, base_cache_path=base_cache_path, cached_lengths_path=cached_lengths_path)

base_cache_list = torch.load(base_cache_path)
cached_lengths_list = torch.load(cached_lengths_path)
# print(f"base_cache_list[0][0][0].shape is {base_cache_list[0][0][0]}, cached_lengths_list[0] is {cached_lengths_list[0]}")
# print(f"the first user's cached length is {cached_lengths_list[0][0]}") # 43
# print(f"base_cache: the first user's cached_k is {base_cache_list[0][-1][2][42]}")

if True:
    run_an_e2e(
        cache_use_type = "no",
        data_loader=eval_delta_loader,
        device=device,
        gr_output_length=gr_output_length,
        eval_state=eval_state,
        model=model,
        eval_batch_size=local_batch_size,
        main_module_bf16=main_module_bf16,
        world_size=world_size,
        return_cache_states=True,
        # base_cache_list=base_cache_list,
        # cached_lengths_list=cached_lengths_list,
        # enable_profiler=True,
    )

if False:
    run_an_e2e(
        cache_use_type = "fully",
        data_loader=eval_delta_loader,
        device=device,
        gr_output_length=gr_output_length,
        eval_state=eval_state,
        model=model,
        eval_batch_size=local_batch_size,
        main_module_bf16=main_module_bf16,
        world_size=world_size,
        base_cache_list=base_cache_list,
        cached_lengths_list=cached_lengths_list,
        # enable_profiler=True,
    )

if False:
    run_an_e2e(
        cache_use_type = "selective",
        data_loader=eval_delta_loader,
        device=device,
        gr_output_length=gr_output_length,
        eval_state=eval_state,
        model=model,
        eval_batch_size=local_batch_size,
        main_module_bf16=main_module_bf16,
        world_size=world_size,
        base_cache_list=base_cache_list,
        cached_lengths_list=cached_lengths_list,
        recompute_ratio=20,
        enable_profiler=True,
    )