# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""
Implements HSTU (Hierarchical Sequential Transduction Unit) in
Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations
(https://arxiv.org/abs/2402.17152, ICML'24).
"""

import abc
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from generative_recommenders.research.modeling.sequential.embedding_modules import (
    EmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    OutputPostprocessorModule,
)
from generative_recommenders.research.modeling.sequential.utils import (
    get_current_embeddings,
)
from generative_recommenders.research.modeling.similarity_module import (
    SequentialEncoderWithLearnedSimilarityModule,
)
from generative_recommenders.research.rails.similarities.module import SimilarityModule

from generative_recommenders.research.modeling.cache.utils import get_next_layer_padded_kv_recompute_mask, get_padded_fusion_kv, get_recompute_indices


TIMESTAMPS_KEY = "timestamps"


class RelativeAttentionBiasModule(torch.nn.Module):
    @abc.abstractmethod
    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: [B, N] x int64
        Returns:
            torch.float tensor broadcastable to [B, N, N]
        """
        pass


class RelativePositionalBias(RelativeAttentionBiasModule):
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        del all_timestamps
        n: int = self._max_seq_len
        t = F.pad(self._w[: 2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        return t[..., r:-r]


class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """

    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int,
        bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )
        self._num_buckets: int = num_buckets
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
            bucketization_fn
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = self._max_seq_len
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2
        # print(f"_pos_w@{self._pos_w.shape} after pad is {F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N).shape}, while t.shape is {t.shape}\nall_timestamps@{all_timestamps.shape}[0] is {all_timestamps[0]}")

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        )
        # use_delta_inference = True
        # if use_delta_inference:
        #     first_one = ext_timestamps[:, 1:].unsqueeze(2) # [128, 211, 1]
        #     second_one = ext_timestamps[:, :-1].unsqueeze(1) # [128, 1, 211]
        #     bucket_input = first_one - second_one
        #     print(f"bucket_inputs are: [0] is {bucket_input[0, 44:49, :]}\n[1] is {bucket_input[1, :5, ]}")
        #     print(f"first_one are: [0] is {first_one[0, 43:49, :]}\n[1] is {first_one[1, :5, ]}")
        #     print(f"second_one are: [0] is {second_one[0, :, 43:49]}\n[1] is {second_one[1, :, :5]}")

        # print(f"ext_timestamps@{ext_timestamps.shape}[0] is {ext_timestamps[0]}")
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        # print(f"ext_timestamps[:, 1:].unsqueeze(2).shape is {ext_timestamps[:, 1:].unsqueeze(2).shape}, ext_timestamps[:, :-1].unsqueeze(1).shape is {ext_timestamps[:, :-1].unsqueeze(1).shape}") # [128, 211, 1], [128, 1, 211]
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1) # [128, 211, 211]
            ),
            min=0,
            max=self._num_buckets,
        ).detach()
        # print(f"bucketed_timestamp@{bucketed_timestamps.shape}[0] is {bucketed_timestamps[0][0]}")
        rel_pos_bias = t[:, :, r:-r]
        # print(f"bucketed_timestamps @ {bucketed_timestamps.shape}, rel_pos_bias @ {rel_pos_bias.shape}") #[128, 211, 211] [1, 211, 211]
        # print(f"is equal? bucketed_timestamps[0, 42:48, :] & bucketed_timestamps[1, :5, :] : {bucketed_timestamps[0, 42:48, :]}\m{(bucketed_timestamps[1, :5, :])}")
        # print(f"is equal? rel_pos_bias[0] & 1: {rel_pos_bias[0].equal(rel_pos_bias[1])}")
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        # print(f"rel_pos_bias@{rel_pos_bias.shape}[0] is {rel_pos_bias[0]}\nrel_ts_bias@{rel_ts_bias.shape}[0] is {rel_ts_bias[0]}\nrel_attn_bias@{(rel_pos_bias + rel_ts_bias).shape}[0] is {(rel_pos_bias + rel_ts_bias)[0]}")
        ret_rel_attn_bias = rel_pos_bias + rel_ts_bias
        use_delta_inference = False
        if use_delta_inference:
            print(f"rel_pos_bias is {rel_pos_bias}")
        return ret_rel_attn_bias


HSTUCacheState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _hstu_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cached_q: Optional[torch.Tensor],
    cached_k: Optional[torch.Tensor],
    delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]],
    x_offsets: torch.Tensor,
    all_timestamps: Optional[torch.Tensor],
    invalid_attn_mask: torch.Tensor,
    rel_attn_bias: RelativeAttentionBiasModule,
    use_all_padded: bool = False,
    delta_recompute_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B: int = x_offsets.size(0) - 1
    n: int = invalid_attn_mask.size(-1)
    if use_all_padded:
        B, m, D = q.shape
    # print(f"m is {m}")
    if delta_x_offsets is not None:
        if use_all_padded:
            padded_k = k
            padded_q = q
        else:
            padded_q, padded_k = cached_q, cached_k
            flattened_offsets = delta_x_offsets[1] + torch.arange(
                start=0,
                end=B * n,
                step=n,
                device=delta_x_offsets[1].device,
                dtype=delta_x_offsets[1].dtype,
            )
            assert isinstance(padded_q, torch.Tensor)
            assert isinstance(padded_k, torch.Tensor)
            padded_q = (
                padded_q.view(B * n, -1)
                .index_copy_(
                    dim=0,
                    index=flattened_offsets,
                    source=q,
                )
                .view(B, n, -1)
            )
            padded_k = (
                padded_k.view(B * n, -1)
                .index_copy_(
                    dim=0,
                    index=flattened_offsets,
                    source=k,
                )
                .view(B, n, -1)
            )
    else:
        if use_all_padded:
            padded_q = q
            padded_k = k
        else:
            padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
                values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
            )
            padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
                values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
            )

    if use_all_padded:
        qk_attn = torch.einsum(
            "bmhd,bnhd->bhmn",
            # padded_q.view(B, n, num_heads, attention_dim),
            padded_q.view(B, m, num_heads, attention_dim),
            padded_k.view(B, n, num_heads, attention_dim),
        )
    else:
        qk_attn = torch.einsum(
            "bmhd,bnhd->bhmn",
            # padded_q.view(B, n, num_heads, attention_dim),
            padded_q.view(B, n, num_heads, attention_dim),
            padded_k.view(B, n, num_heads, attention_dim),
        )        

    if all_timestamps is not None:
        rab = rel_attn_bias(all_timestamps)
        if delta_x_offsets is not None:
            batch_indices = torch.arange(B).view(-1, 1)
            rab = rab[batch_indices, delta_x_offsets[1]]
            if False and "use mask for select":
                rab = rab[delta_recompute_mask].contiguous().view(B, m, n)
            # print(f"is equal? {indices_rab.equal(rab)}")
        # print(f"qk_attn is {qk_attn.shape}, rab is {rab.shape}")
        qk_attn = qk_attn + rab.unsqueeze(1)

    qk_attn = F.silu(qk_attn) / n
    
    attn_mask = invalid_attn_mask.unsqueeze(0)
    # print(f"attn_mask.shape is {attn_mask.shape}, qk_attn.shape is {qk_attn.shape}")
    if delta_x_offsets is not None:
        attn_mask = attn_mask.expand(B, -1, -1)[batch_indices, delta_x_offsets[1]]
        if False and "use mask for select":
            attn_mask = attn_mask.expand(B, -1, -1)[delta_recompute_mask].contiguous().view(B, m, n)
        # print(f"attn_mask[0] is {attn_mask[0][0]}")
        # print(f"is equal? {indices_attn_mask.equal(attn_mask)}")
        qk_attn = qk_attn * attn_mask.unsqueeze(1)
    else:
        qk_attn = qk_attn * attn_mask.unsqueeze(0)
    # print(f"qk_attn.shape is {qk_attn.shape}, attn_mask.shape is {attn_mask.shape}")

    if use_all_padded:
        attn_output = torch.einsum(
            "bhnm,bmhd->bnhd",
            qk_attn,
            v.view(B, n, num_heads, linear_dim),
        ).contiguous().view(B, m, num_heads * linear_dim)
    else:
        attn_output = torch.ops.fbgemm.dense_to_jagged(
            torch.einsum(
                "bhnm,bmhd->bnhd",
                qk_attn,
                torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]).reshape(
                    B, n, num_heads, linear_dim
                ),
            ).reshape(B, n, num_heads * linear_dim),
            [x_offsets],
        )[0]

    

    return attn_output, padded_q, padded_k


class SequentialTransductionUnitJagged(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        relative_attention_bias_module: Optional[RelativeAttentionBiasModule] = None,
        normalization: str = "rel_bias",
        linear_config: str = "uvqk",
        concat_ua: bool = False,
        epsilon: float = 1e-6,
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._rel_attn_bias: Optional[RelativeAttentionBiasModule] = (
            relative_attention_bias_module
        )
        self._normalization: str = normalization
        self._linear_config: str = linear_config
        if self._linear_config == "uvqk":
            self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
                torch.empty(
                    (
                        embedding_dim,
                        linear_hidden_dim * 2 * num_heads
                        + attention_dim * num_heads * 2,
                    )
                ).normal_(mean=0, std=0.02),
            )
        else:
            raise ValueError(f"Unknown linear_config {self._linear_config}")
        self._linear_activation: str = linear_activation
        self._concat_ua: bool = concat_ua
        self._o = torch.nn.Linear(
            in_features=linear_hidden_dim * num_heads * (3 if concat_ua else 1),
            out_features=embedding_dim,
        )
        torch.nn.init.xavier_uniform_(self._o.weight)
        self._eps: float = epsilon

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=[self._linear_dim * self._num_heads], eps=self._eps
        )

    def forward(  # pyre-ignore [3]
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[HSTUCacheState] = None,
        return_cache_states: bool = False,
        use_all_padded: bool = False,
        recompute_mask: torch.Tensor = None, # when selective_reuse == True, this param decide which postion to recompute(True) and reuse(False), its size is (B, N) 
        need_compute_mask: bool = False, # only for whether compute mask
        cached_mask: torch.Tensor = None, # when need_compute_mask == True, this param is need for decide which postion to be included
        r: int = 20, # when need_compute_mask == True, this param is need for decide top r% kv deviation position mask
        use_delta_x: bool = True, # add for selective compute to decide whether fully compute or not
        cached_lengths: torch.Tensor = None,
    ):
        """
        Args:
            x: (\sum_i N_i, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: optional (B, N) x int64.
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
            delta_x_offsets: optional 2-tuple ((B,) x int32, (B,) x int32).
                For the 1st element in the tuple, each element is in [0, x_offsets[-1]). For the
                2nd element in the tuple, each element is in [0, N).
            cache: Optional 4-tuple of (v, padded_q, padded_k, output) from prior runs,
                where all except padded_q, padded_k are jagged.
        Returns:
            x' = f(x), (\sum_i N_i, D) x float.
        """
        if use_all_padded:
            B, N, D = x.shape
        n: int = invalid_attn_mask.size(-1)
        # print(f"n is {n}, N is {N}")
        cached_q = None
        cached_k = None
        if delta_x_offsets is not None:
            # In this case, for all the following code, x, u, v, q, k become restricted to
            # [delta_x_offsets[0], :].
            assert cache is not None
            cached_v, cached_q, cached_k, cached_outputs = cache
            
            if use_all_padded:
                if n == N and use_delta_x:
                    # print(f"[INFO] split x")
                    batch_indices = torch.arange(B).view(-1, 1)
                    x = x[batch_indices, delta_x_offsets[1]]
                    if False and "use mask for select x":
                        x = x[x_mask.squeeze(-1)].contiguous().view(B, -1, D)
                    # print(f"is equal? {indices_x.equal(x)}")
            else:
                x = x[delta_x_offsets[0], :]

        normed_x = self._norm_input(x)

        if self._linear_config == "uvqk":
            if use_all_padded:
                batched_mm_output = torch.matmul(normed_x, self._uvqk)
                if self._linear_activation == "silu":
                    batched_mm_output = F.silu(batched_mm_output)
                elif self._linear_activation == "none":
                    batched_mm_output = batched_mm_output
                u, v, q, k = torch.split(
                    batched_mm_output,
                    [
                        self._linear_dim * self._num_heads,
                        self._linear_dim * self._num_heads,
                        self._attention_dim * self._num_heads,
                        self._attention_dim * self._num_heads,
                    ],
                    dim=2,
                )
            else:
                batched_mm_output = torch.mm(normed_x, self._uvqk)
                if self._linear_activation == "silu":
                    batched_mm_output = F.silu(batched_mm_output)
                elif self._linear_activation == "none":
                    batched_mm_output = batched_mm_output             
                u, v, q, k = torch.split(
                    batched_mm_output,
                    [
                        self._linear_dim * self._num_heads,
                        self._linear_dim * self._num_heads,
                        self._attention_dim * self._num_heads,
                        self._attention_dim * self._num_heads,
                    ],
                    dim=1,
                )
        else:
            raise ValueError(f"Unknown self._linear_config {self._linear_config}")

        if delta_x_offsets is not None and use_delta_x:
            if use_all_padded:
                valid_indices = delta_x_offsets[1]
                k = cached_k.scatter_(
                    dim=1,
                    index=valid_indices.unsqueeze(-1).expand(-1, -1, D),
                    src=k
                )
                v = cached_v.scatter_(
                    dim=1,
                    index=valid_indices.unsqueeze(-1).expand(-1, -1, D),
                    src=v
                )
                # print(f"k.shape is {k.shape}, v.shape is {v.shape}")

                if False and "only when the k is padded":
                    k = torch.where(fusion_mask, k, cached_k)
                    v = torch.where(fusion_mask, v, cached_v)
            else:
                v = cached_v.index_copy_(dim=0, index=delta_x_offsets[0], source=v)

        # print(f"k[0] is {k[0]}\nv[0] is {v[0]}")

        if need_compute_mask:
            assert delta_x_offsets is not None and use_all_padded
            # print(f"[INFO] compute next layer recompute_mask")
            # print(f"!!!recompute_ratio is {r}!!!")
            delta_x_indices, delta_lengths, valid_mask = get_recompute_indices(
                cached_k=cached_k,
                cached_v=cached_v,
                compute_k=k,
                compute_v=v,
                valid_mask=cached_mask,
                cached_lengths=cached_lengths,
                past_lengths=x_offsets,
                delta_x_indices=delta_x_offsets[1],
                delta_lengths=delta_x_offsets[0],
                use_percentage=True,
                r=r,
            )
            # print(f"after get_recompute_indices: the delta_lengths is {delta_lengths}")
            if False and " use padded and not delta ":
                recompute_mask = get_next_layer_padded_kv_recompute_mask(
                    cached_k=cached_k,
                    cached_v=cached_v,
                    compute_k=k,
                    compute_v=v,
                    cached_mask=cached_mask,
                    r=r,
                )
            # print(f"recompute_mask[0] is {recompute_mask[0]}")


        if use_all_padded:
            B: int = x.size(0)
        else:
            B: int = x_offsets.size(0) - 1

        if self._normalization == "rel_bias" or self._normalization == "hstu_rel_bias":
            assert self._rel_attn_bias is not None
            attn_output, padded_q, padded_k = _hstu_attention_maybe_from_cache(
                num_heads=self._num_heads,
                attention_dim=self._attention_dim,
                linear_dim=self._linear_dim,
                q=q,
                k=k,
                v=v,
                cached_q=cached_q,
                cached_k=cached_k,
                delta_x_offsets=delta_x_offsets if use_delta_x else None,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                rel_attn_bias=self._rel_attn_bias,
                use_all_padded=use_all_padded,
                delta_recompute_mask=None,
            )
        elif self._normalization == "softmax_rel_bias":
            if delta_x_offsets is not None:
                B = x_offsets.size(0) - 1
                padded_q, padded_k = cached_q, cached_k
                flattened_offsets = delta_x_offsets[1] + torch.arange(
                    start=0,
                    end=B * n,
                    step=n,
                    device=delta_x_offsets[1].device,
                    dtype=delta_x_offsets[1].dtype,
                )
                assert padded_q is not None
                assert padded_k is not None
                padded_q = (
                    padded_q.view(B * n, -1)
                    .index_copy_(
                        dim=0,
                        index=flattened_offsets,
                        source=q,
                    )
                    .view(B, n, -1)
                )
                padded_k = (
                    padded_k.view(B * n, -1)
                    .index_copy_(
                        dim=0,
                        index=flattened_offsets,
                        source=k,
                    )
                    .view(B, n, -1)
                )
            else:
                padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
                    values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                )
                padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
                    values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                )

            qk_attn = torch.einsum("bnd,bmd->bnm", padded_q, padded_k)
            if self._rel_attn_bias is not None:
                qk_attn = qk_attn + self._rel_attn_bias(all_timestamps)
            qk_attn = F.softmax(qk_attn / math.sqrt(self._attention_dim), dim=-1)
            qk_attn = qk_attn * invalid_attn_mask
            attn_output = torch.ops.fbgemm.dense_to_jagged(
                torch.bmm(
                    qk_attn,
                    torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]),
                ),
                [x_offsets],
            )[0]
        else:
            raise ValueError(f"Unknown normalization method {self._normalization}")

        if use_all_padded:
            if delta_x_offsets is None:
                attn_output = attn_output
            # else:
            #     attn_mask = delta_x_mask
                # attn_output = attn_output * attn_mask
                # print(f"attn_output is {attn_output.shape}")
        else:
            attn_output = (
                attn_output
                if delta_x_offsets is None
                else attn_output[delta_x_offsets[0], :]
            )
        if self._concat_ua: # False
            a = self._norm_attn_output(attn_output)
            o_input = torch.cat([u, a, u * a], dim=-1)
        else:
            o_input = u * self._norm_attn_output(attn_output)

        new_outputs = (
            self._o(
                F.dropout(
                    o_input,
                    p=self._dropout_ratio,
                    training=self.training,
                )
            )
            + x
        )

        if delta_x_offsets is not None:
            if use_all_padded:
                attn_output = attn_output
            else:
                new_outputs = cached_outputs.index_copy_(
                    dim=0, index=delta_x_offsets[0], source=new_outputs
                )

        if return_cache_states and delta_x_offsets is None:
            v = v.contiguous()

        if need_compute_mask:
            return new_outputs, (v, padded_q, padded_k, new_outputs), delta_x_indices, delta_lengths, valid_mask
        else:
            return new_outputs, (v, padded_q, padded_k, new_outputs)


class HSTUJagged(torch.nn.Module):
    def __init__(
        self,
        modules: List[SequentialTransductionUnitJagged],
        autocast_dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()

        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(
            modules=modules
        )
        self._autocast_dtype: Optional[torch.dtype] = autocast_dtype

    def jagged_forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
        use_all_padded: bool = False,
        cached_mask: torch.Tensor = None, 
        r: int = 20, 
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        Args:
            x: (\sum_i N_i, D) x float
            x_offsets: (B + 1) x int32
            all_timestamps: (B, 1 + N) x int64
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}
            return_cache_states: bool. True if we should return cache states.

        Returns:
            x' = f(x), (\sum_i N_i, D) x float
        """
        cache_states: List[HSTUCacheState] = []
        recompute_mask = None
        cached_lengths = None
        selective_reuse_flag = use_all_padded and delta_x_offsets is not None and cached_mask is not None
        if selective_reuse_flag:
            cached_lengths = torch.sum(cached_mask, dim=1)
            delta_x_indices = None
            delta_lengths = None
            valid_mask = None
            need_compute_mask = False

        with torch.autocast(
            "cuda",
            enabled=self._autocast_dtype is not None,
            dtype=self._autocast_dtype or torch.float16,
        ):
            for i, layer in enumerate(self._attention_layers):
                # print(f"================== layer {i} ====================")
                if selective_reuse_flag and i == 0:
                    # print(f"selective use cache and the first layer")
                    use_delta_x = False
                    need_compute_mask = True
                    cached_mask = cached_mask
                    cached_lengths = cached_lengths
                    r = r
                elif selective_reuse_flag:
                    use_delta_x = True
                    need_compute_mask = False
                    cached_mask = None
                    r = None
                else:
                    use_delta_x = True # for fully reuse
                    recompute_mask = None
                    need_compute_mask = False
                    cached_mask = None
                    r = None
                # print(f"delta_x_offsets is {delta_x_offsets[0]}")
                if need_compute_mask:
                    x, cache_states_i, delta_x_indices, delta_lengths, valid_mask = layer(
                        x=x,
                        x_offsets=x_offsets,
                        all_timestamps=all_timestamps,
                        invalid_attn_mask=invalid_attn_mask,
                        delta_x_offsets=delta_x_offsets,
                        cache=cache[i] if cache is not None else None,
                        return_cache_states=return_cache_states,
                        use_all_padded=use_all_padded,
                        recompute_mask=recompute_mask,
                        need_compute_mask=need_compute_mask,
                        cached_mask=cached_mask,
                        r=r,
                        use_delta_x=use_delta_x,
                        cached_lengths=cached_lengths,
                    )                
                else:
                    x, cache_states_i = layer(
                        x=x,
                        x_offsets=x_offsets,
                        all_timestamps=all_timestamps,
                        invalid_attn_mask=invalid_attn_mask,
                        delta_x_offsets=delta_x_offsets,
                        cache=cache[i] if cache is not None else None,
                        return_cache_states=return_cache_states,
                        use_all_padded=use_all_padded,
                        recompute_mask=recompute_mask,
                        need_compute_mask=need_compute_mask,
                        cached_mask=cached_mask,
                        r=r,                    
                        use_delta_x=use_delta_x,
                        cached_lengths=cached_lengths,    
                    )
                if need_compute_mask:
                    # print(f"[INFO] change the delta_x_offsets")
                    # print(f"before: the delta_x_offsets is {delta_x_offsets[1][0]}")
                    delta_x_offsets = (delta_lengths, delta_x_indices)
                    # print(f"after: the delta_x_offsets is {delta_x_offsets[1][0]}\ndelta_lengths is {delta_x_offsets[0]}")
                    # cached_mask = valid_mask
                if return_cache_states:
                    cache_states.append(cache_states_i)
                # torch.cuda.synchronize()
        # print(f"at the end of jagged_forward, delta_x_offsets[0] is {delta_x_offsets[0]}")

        if selective_reuse_flag:
            return x, cache_states, delta_x_offsets[0]
        else:
            return x, cache_states

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
        use_all_padded: bool = False,
        cached_mask: torch.Tensor = None,
        r: int = 20,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        Args:
            x: (B, N, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: (B, 1 + N) x int64
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
        Returns:
            x' = f(x), (B, N, D) x float
        """
        # print(f"before jagged_forward x.shape is {x.shape}")
        if use_all_padded == False:
            if len(x.size()) == 3:
                x = torch.ops.fbgemm.dense_to_jagged(x, [x_offsets])[0]
        

        if use_all_padded:
            if cached_mask is not None:
                y, cache_states, delta_lengths = self.jagged_forward(
                    x=x,
                    x_offsets=x_offsets,
                    all_timestamps=all_timestamps,
                    invalid_attn_mask=invalid_attn_mask,
                    delta_x_offsets=delta_x_offsets,
                    cache=cache,
                    return_cache_states=return_cache_states,
                    use_all_padded=use_all_padded,     
                    cached_mask=cached_mask,
                    r=r,           
                )
            else:
                y, cache_states = self.jagged_forward(
                    x=x,
                    x_offsets=x_offsets,
                    all_timestamps=all_timestamps,
                    invalid_attn_mask=invalid_attn_mask,
                    delta_x_offsets=delta_x_offsets,
                    cache=cache,
                    return_cache_states=return_cache_states,
                    use_all_padded=use_all_padded,     
                    cached_mask=cached_mask,
                    r=r,           
                )
        else:
            jagged_x, cache_states = self.jagged_forward(
                x=x,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                delta_x_offsets=delta_x_offsets,
                cache=cache,
                return_cache_states=return_cache_states,
                use_all_padded=use_all_padded,
                cached_mask=cached_mask,
                r=r,
            )
            y = torch.ops.fbgemm.jagged_to_padded_dense(
                values=jagged_x,
                offsets=[x_offsets],
                max_lengths=[invalid_attn_mask.size(1)],
                padding_value=0.0,
            )
        # print(f"in HSTUJagged.forward, the delta_x_offsets[0] is {delta_x_offsets[0]}")

        if use_all_padded and cached_mask is not None:
            return y, cache_states, delta_lengths
        else:
            return y, cache_states


class HSTU(SequentialEncoderWithLearnedSimilarityModule):
    """
    Implements HSTU (Hierarchical Sequential Transduction Unit) in
    Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations,
    https://arxiv.org/abs/2402.17152.

    Note that this implementation is intended for reproducing experiments in
    the traditional sequential recommender setting (Section 4.1.1), and does
    not yet use optimized kernels discussed in the paper.
    """

    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        linear_dim: int,
        attention_dim: int,
        normalization: str,
        linear_config: str,
        linear_activation: str,
        linear_dropout_rate: float,
        attn_dropout_rate: float,
        embedding_module: EmbeddingModule,
        similarity_module: SimilarityModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        enable_relative_attention_bias: bool = True,
        concat_ua: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__(ndp_module=similarity_module)

        self._embedding_dim: int = embedding_dim
        self._item_embedding_dim: int = embedding_module.item_embedding_dim
        self._max_sequence_length: int = max_sequence_len
        self._embedding_module: EmbeddingModule = embedding_module
        self._input_features_preproc: InputFeaturesPreprocessorModule = (
            input_features_preproc_module
        )
        self._output_postproc: OutputPostprocessorModule = output_postproc_module
        self._num_blocks: int = num_blocks
        self._num_heads: int = num_heads
        self._dqk: int = attention_dim
        self._dv: int = linear_dim
        self._linear_activation: str = linear_activation
        self._linear_dropout_rate: float = linear_dropout_rate
        self._attn_dropout_rate: float = attn_dropout_rate
        self._enable_relative_attention_bias: bool = enable_relative_attention_bias
        self._hstu = HSTUJagged(
            modules=[
                SequentialTransductionUnitJagged(
                    embedding_dim=self._embedding_dim,
                    linear_hidden_dim=linear_dim,
                    attention_dim=attention_dim,
                    normalization=normalization,
                    linear_config=linear_config,
                    linear_activation=linear_activation,
                    num_heads=num_heads,
                    # TODO: change to lambda x.
                    relative_attention_bias_module=(
                        RelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=max_sequence_len
                            + max_output_len,  # accounts for next item.
                            num_buckets=128,
                            bucketization_fn=lambda x: (
                                torch.log(torch.abs(x).clamp(min=1)) / 0.301
                            ).long(),
                        )
                        if enable_relative_attention_bias
                        else None
                    ),
                    dropout_ratio=linear_dropout_rate,
                    attn_dropout_ratio=attn_dropout_rate,
                    concat_ua=concat_ua,
                )
                for _ in range(num_blocks)
            ],
            autocast_dtype=None,
        )
        # causal forward, w/ +1 for padding.
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (
                        self._max_sequence_length + max_output_len,
                        self._max_sequence_length + max_output_len,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        self._verbose: bool = verbose
        self.reset_params()

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if ("_hstu" in name) or ("_embedding_module" in name):
                if self._verbose:
                    print(f"Skipping init for {name}")
                continue
            try:
                torch.nn.init.xavier_normal_(params.data)
                if self._verbose:
                    print(
                        f"Initialize {name} as xavier normal: {params.data.size()} params"
                    )
            except:
                if self._verbose:
                    print(f"Failed to initialize {name}: {params.data.size()} params")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def debug_str(self) -> str:
        debug_str = (
            f"HSTU-b{self._num_blocks}-h{self._num_heads}-dqk{self._dqk}-dv{self._dv}"
            + f"-l{self._linear_activation}d{self._linear_dropout_rate}"
            + f"-ad{self._attn_dropout_rate}"
        )
        if not self._enable_relative_attention_bias:
            debug_str += "-norab"
        return debug_str

    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
        use_all_padded: bool = False,
        cached_mask: torch.Tensor = None,
        r: int = 20,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        [B, N] -> [B, N, D].
        """
        device = past_lengths.device
        float_dtype = past_embeddings.dtype
        B, N, _ = past_embeddings.size()

        past_lengths, user_embeddings, _ = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )
        # print(f"user_embeddings before _hstu is {user_embeddings.shape}") # [128, 211, 256]
        # print(f"user_embeddings is equal? {user_embeddings[0, 44:49].equal(user_embeddings[1, :5])}")
        if use_all_padded:
            x_offsets = past_lengths
        else:
            x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(past_lengths)

        float_dtype = user_embeddings.dtype
        if cached_mask is not None:
            user_embeddings, cached_states, delta_lengths = self._hstu(
                x=user_embeddings,
                x_offsets=x_offsets,
                all_timestamps=(
                    past_payloads[TIMESTAMPS_KEY]
                    if TIMESTAMPS_KEY in past_payloads
                    else None
                ),
                invalid_attn_mask=1.0 - self._attn_mask.to(float_dtype),
                delta_x_offsets=delta_x_offsets,
                cache=cache,
                return_cache_states=return_cache_states,
                use_all_padded=use_all_padded,
                cached_mask=cached_mask,
                r=r,
            )  
        else:          
            user_embeddings, cached_states = self._hstu(
                x=user_embeddings,
                x_offsets=x_offsets,
                all_timestamps=(
                    past_payloads[TIMESTAMPS_KEY]
                    if TIMESTAMPS_KEY in past_payloads
                    else None
                ),
                invalid_attn_mask=1.0 - self._attn_mask.to(float_dtype),
                delta_x_offsets=delta_x_offsets,
                cache=cache,
                return_cache_states=return_cache_states,
                use_all_padded=use_all_padded,
                cached_mask=cached_mask,
                r=r,
            )
        # print(f"user_embeddings.shape is {user_embeddings.shape}")
        # print(f"user_embeddings after _hstu is {user_embeddings.shape}") # [128, 211, 256]
        # print(f"user_embeddings is equal? {user_embeddings[0, 44:49].equal(user_embeddings[1, :5])}")
        if cached_mask is not None:
            return self._output_postproc(user_embeddings), cached_states, delta_lengths
        else:
            return self._output_postproc(user_embeddings), cached_states

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
        use_all_padded: bool = False,
    ) -> torch.Tensor:
        """
        Runs the main encoder.

        Args:
            past_lengths: (B,) x int64
            past_ids: (B, N,) x int64 where the latest engaged ids come first. In
                particular, past_ids[i, past_lengths[i] - 1] should correspond to
                the latest engaged values.
            past_embeddings: (B, N, D) x float or (\sum_b N_b, D) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).

        Returns:
            encoded_embeddings of [B, N, D].
        """
        encoded_embeddings, _ = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )
        return encoded_embeddings

    def _encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]],
        cache: Optional[List[HSTUCacheState]],
        return_cache_states: bool,
        use_all_padded: bool = False,
        cached_mask: torch.Tensor = None,
        r: int = 20,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[HSTUCacheState]]]:
        """
        Args:
            past_lengths: (B,) x int64.
            past_ids: (B, N,) x int64.
            past_embeddings: (B, N, D,) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).
            return_cache_states: bool.

        Returns:
            (B, D) x float, representing embeddings for the current state.
        """
        if cached_mask is not None:
            encoded_seq_embeddings, cache_states, delta_lengths = self.generate_user_embeddings(
                past_lengths=past_lengths,
                past_ids=past_ids,
                past_embeddings=past_embeddings,
                past_payloads=past_payloads,
                delta_x_offsets=delta_x_offsets,
                cache=cache,
                return_cache_states=return_cache_states,
                use_all_padded=use_all_padded,
                cached_mask=cached_mask,
                r=r,
            )  # [B, N, D]
        else:
            encoded_seq_embeddings, cache_states = self.generate_user_embeddings(
                past_lengths=past_lengths,
                past_ids=past_ids,
                past_embeddings=past_embeddings,
                past_payloads=past_payloads,
                delta_x_offsets=delta_x_offsets,
                cache=cache,
                return_cache_states=return_cache_states,
                use_all_padded=use_all_padded,
                cached_mask=cached_mask,
                r=r,
            )  # [B, N, D]
        if use_all_padded and delta_x_offsets is not None:
            # delta_lengths = torch.full(size=(encoded_seq_embeddings.shape[0],), fill_value=encoded_seq_embeddings.shape[1], dtype=torch.int32 , device=encoded_seq_embeddings.device)
            # delta_lengths = past_lengths
            # delta_lengths[0] = encoded_seq_embeddings.shape[1]
            if cached_mask is not None:
                delta_lengths = delta_lengths
                # print(f"delta_lengths.max is {torch.max(delta_lengths)}")
            else:
                delta_lengths = delta_x_offsets[0]
            # print(f"delta_lenths is {delta_lengths}")
            current_embeddings = get_current_embeddings(
                lengths=delta_lengths, encoded_embeddings=encoded_seq_embeddings
            )
        else:
            current_embeddings = get_current_embeddings(
                lengths=past_lengths, encoded_embeddings=encoded_seq_embeddings
            )
        # print(f"current_embddings.shape is {current_embeddings.shape}")
        if return_cache_states:
            return current_embeddings, cache_states
        else:
            return current_embeddings

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
        use_all_padded: bool = False,
        cached_mask: torch.Tensor = None,
        r: int = 20,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[HSTUCacheState]]]:
        """
        Runs encoder to obtain the current hidden states.

        Args:
            past_lengths: (B,) x int.
            past_ids: (B, N,) x int.
            past_embeddings: (B, N, D) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).

        Returns:
            (B, D,) x float, representing encoded states at the most recent time step.
        """
        # print(f"!!!!Attention!!!! delta_x_offsets is None? {delta_x_offsets is None}")
        return self._encode(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
            use_all_padded=use_all_padded,
            cached_mask=cached_mask,
            r=r,
        )
