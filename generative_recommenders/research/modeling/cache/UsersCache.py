import numpy as np
import torch

from typing import List

class UserCache:
    def __init__(
        self,
        user_id: int, 
        items_length: int,
        item_ids: torch.Tensor, # [items_length]
        layer_num: int,
        cached_k: List[torch.Tensor], # List[1, items_length, dq)]
        cached_v: List[torch.Tensor], # List[(items_length, dv)]
    ):
        self.user_id = user_id
        self.items_length = items_length
        self.item_ids = item_ids
        self.layer_num = layer_num

        self.cached_k = cached_k
        self.cached_v = cached_v

    def update_kv_cache(
        self,
        layer_id: int, # the range of this value is from 0 to self.layer_num
        items_length: int,
        item_ids: torch.Tensor, 
        updated_k: torch.Tensor, # [1, items_length, dq]
        updated_v: torch.Tensor, # [item_length, dv]
    ):
        self.item_ids = item_ids
        self.items_length = items_length
        # print(f"updated_k is {updated_k}\nupdated_v is {updated_v}")

        if layer_id <= self.layer_num :
            self.cached_k[layer_id] = updated_k
            self.cached_v[layer_id] = updated_v
        else:
            raise ValueError(f"[when update]Layer {layer_id} is out of bounds for this user cache.")
    
    def get_kv_cache(
        self,
        layer_id: int,
    ):
        if layer_id <= self.layer_num:
            return self.cached_k[layer_id], self.cached_v[layer_id]
        else:
            raise ValueError(f"[when get]Layer {layer_id} is out of bounds for this user cache.")
    
class UsersCache:
    def __init__(
        self,
        dk: int,
        dv: int,
        layer_num: int,
        k_dtype: torch.dtype = torch.float32,
        v_dtype: torch.dtype = torch.float32,
    ):
        self.users_cache = {}
        self.dk = dk
        self.dv = dv
        self.layer_num = layer_num
        self.k_dtype = k_dtype
        self.v_dtype = v_dtype

    def _padding_tensor(self, tensor, target_shape):
        padded_tensor = torch.zeros(target_shape, dtype=tensor.dtype)

        tensor_shape = tensor.shape
        slices = tuple(slice(0, min(tensor_shape[i], target_shape[i])) for i in range(len(tensor_shape)))

        padded_tensor[slices] = tensor[slices]

        return padded_tensor

    def get_batched_user_cache(
        self,
        user_ids: torch.Tensor, # [B]
        item_lengths: torch.Tensor, # [B]
        max_sequence_length: int,
    ):
        B = user_ids.size(dim=0)
        N = max_sequence_length

        ret_k = [[] for _ in range(self.layer_num)]
        ret_v = [[] for _ in range(self.layer_num)]

        for i in range(B):
            user_id = user_ids[i].item()
            length = item_lengths[i].item()

            target_k_shape = (1, N, self.dk)
            target_v_shape = (length, self.dv)

            if user_id in self.users_cache.keys():
                # print(f"succesfully hit the user_{user_id}")
                user_cache = self.users_cache[user_id]
                for layer_id in range(self.layer_num):
                    cached_k, cached_v = user_cache.get_kv_cache(layer_id)
                    # print(f"[in get_batched_user_cache()]the shape of cached_k is {cached_k.shape}")

                    padded_k = self._padding_tensor(cached_k, target_k_shape)
                    padded_v = self._padding_tensor(cached_v, target_v_shape)

                    # print(f"[test mask 4 padded_k in UsersCache]: {torch.sum(torch.all(padded_k==0, dim=2) == False, dim=1)}, while length is {user_cache.items_length}")

                    ret_k[layer_id].append(padded_k)
                    ret_v[layer_id].append(padded_v)
            else:
                for layer_id in range(self.layer_num):
                    ret_k[layer_id].append(torch.zeros(target_k_shape, dtype=self.k_dtype))
                    ret_v[layer_id].append(torch.zeros(target_v_shape, dtype=self.v_dtype))

        ret_cached_k = [torch.cat(layer, dim=0) for layer in ret_k]
        ret_cached_v = [torch.cat(layer, dim=0) for layer in ret_v]

        return ret_cached_k, ret_cached_v
    
    def update_batched_user_cache(
        self,
        user_ids: torch.Tensor, # [B]
        item_lengths: torch.Tensor, # [B]
        user_items_list: torch.Tensor, # [B, N]
        updated_k: List[torch.Tensor], # layer_num[B, N, dk]
        updated_v: List[torch.Tensor], # layer_num[sum_i(item_lengths[i]), dv]
    ):
        B = user_ids.size(dim=0)
        layer_num = len(updated_k)
        if layer_num != self.layer_num:
            print(f"layer_num is {layer_num}, self.layer_num is {self.layer_num}")
            raise ValueError(f"layer_num != self.layer_num")

        cur_v_length = 0
        for i in range(B):
            user_id = user_ids[i].item()
            # print(f"user_id is {user_id}")
            length = item_lengths[i].item()
            item_ids = user_items_list[i]
            if user_id in self.users_cache.keys():
                # print(f"cached user_id:{user_id}")
                user_cache = self.users_cache[user_id]
                for layer_id in range(layer_num):
                    # print(f"in layer_{layer_id}:\nupdated_k is {updated_k[layer_id][i, :, :]}\nupdated_v is {updated_v[layer_id][cur_v_length:cur_v_length+length, :]}")
                    user_cache.update_kv_cache(
                        layer_id=layer_id,
                        items_length=length,
                        item_ids=item_ids,
                        updated_k=updated_k[layer_id][i, :length, :].unsqueeze(0),
                        updated_v=updated_v[layer_id][cur_v_length:cur_v_length+length, :]
                    )
            else:
                # print(f"miss user_id:{user_id}")
                self.users_cache[user_id] = UserCache(
                                                user_id=user_id, 
                                                items_length=length,
                                                item_ids=item_ids, 
                                                layer_num=layer_num,
                                                cached_k=[updated_k[j][i, :length, :].unsqueeze(0) for j in range(layer_num)],
                                                cached_v=[updated_v[j][cur_v_length:cur_v_length+length, :] for j in range(layer_num)],
                                            )
            # print(f"[in update_batched_user_cacahe()]the shape of cached_k in {self.users_cache[user_id].cached_k[0].shape}")
                # print(f"whether the user_id is in users_cache {user_id in self.users_cache.keys()}")
            
            cur_v_length += length
        
    def del_user_cache(
        self,
        user_id,
    ):
        if user_id in self.users_cache:
            del self.users_cache[user_id]
        else:
            raise ValueError(f"User {user_id} not found in the cache.")
        

            
        