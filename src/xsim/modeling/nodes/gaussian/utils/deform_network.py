# Contains modified code from OmniRe / drivestudio
# (https://github.com/ziyc/drivestudio, MIT, Copyright (c) 2024 Ziyu Chen).
# See THIRD_PARTY_LICENSES.md.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append((nn.Identity(), 1))
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append((p_fn, freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs * freq) for fn, freq in self.embed_fns], -1)

    def __call__(self, inputs):
        return self.embed(inputs)


def get_embedder(multires, i=1):
    if i == -1: return nn.Identity(), 3

    embedder_obj = Embedder(
        include_input=True,
        input_dims=i,
        max_freq_log2=multires - 1,
        num_freqs=multires,
        log_sampling=True,
        periodic_fns=[torch.sin, torch.cos]
    )
    return embedder_obj, embedder_obj.out_dim


class ConditionalDeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, embed_dim=10,
                 x_multires=10, t_multires=10,
                 deform_quat=True, deform_scale=True):
        super(ConditionalDeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.embed_dim = embed_dim
        self.deform_quat = deform_quat
        self.deform_scale = deform_scale
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(x_multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch + embed_dim

        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )

        self.gaussian_warp = nn.Linear(W, 3)
        if self.deform_quat:
            self.gaussian_rotation = nn.Linear(W, 4)
        if self.deform_scale:
            self.gaussian_scaling = nn.Linear(W, 3)

    def get_embedding(self, x, t, condition):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)
        return torch.cat([x_emb, t_emb, condition], dim=-1)

    def deform_network(self, h0):
        h = h0

        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h, inplace=True)
            if i in self.skips:
                h = torch.cat([h0, h], dim=-1)

        d_xyz = self.gaussian_warp(h)
        scaling, rotation = None, None
        if self.deform_scale:
            scaling = self.gaussian_scaling(h)
        if self.deform_quat:
            rotation = self.gaussian_rotation(h)
        return d_xyz, rotation, scaling

    def forward(self, x, t, condition):
        h0 = self.get_embedding(x, t, condition)
        return self.deform_network(h0)
