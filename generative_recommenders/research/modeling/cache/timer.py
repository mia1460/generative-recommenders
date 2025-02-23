import torch

class CUDATimer:
    def __init__(self, name, verbose=True):
        self.name = name
        self.verbose = verbose
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    def __enter__(self):
        self.start_event.record()
        return self
    
    def __exit__(self, *args):
        self.end_event.record()
        self.end_event.synchronize()  # 确保事件完成
        self.elapsed_time = self.start_event.elapsed_time(self.end_event)
        if self.verbose:
            print(f"[CUDA Timer] {self.name}: {self.elapsed_time:.2f} ms")
    
    def get_time(self):
        """手动获取时间（适用于非上下文管理器场景）"""
        return self.elapsed_time