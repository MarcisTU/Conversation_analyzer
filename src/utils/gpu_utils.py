import torch


class AudioUtils:
    @staticmethod
    def get_gpu_usage():
        r_mb = torch.cuda.memory_reserved(0) / 1_000 / 1_000
        a_mb = torch.cuda.memory_allocated(0) / 1_000 / 1_000
        return {'reserved': r_mb, 'used': {a_mb}}
