import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '/home/yinj@/workplace/generative-recommenders'))

from generative_recommenders.ops.triton.triton_jagged import _get_bmm_configs

configs = _get_bmm_configs()
print(f"return configs from _get_bmm_configs() is {configs}")