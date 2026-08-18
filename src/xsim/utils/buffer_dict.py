import torch


class BufferDict(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        for k, v in d.items():
            self.register_buffer(k, v)
        self._keys = list(d.keys())

    def keys(self):
        return self._keys

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)
