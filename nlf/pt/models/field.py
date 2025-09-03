import numpy as np
import torch
import torch.nn as nn

from nlf.paths import PROJDIR
from nlf.pt.util import get_config


def build_field():
    FLAGS = get_config()
    layer_dims = (
            [FLAGS.field_hidden_size] * FLAGS.field_hidden_layers +
            [(FLAGS.backbone_link_dim + 1) * (FLAGS.depth + 2)])

    gps_mlp = GPSBaseModel(pos_enc_dim=512, hidden_dim=2048, output_dim=FLAGS.field_posenc_dim)
    return GPSField(gps_mlp, layer_dims=layer_dims)


class GPSField(nn.Module):
    def __init__(self, lbo_mlp, layer_dims):
        super().__init__()
        FLAGS = get_config()
        self.posenc_dim = FLAGS.field_posenc_dim
        self.lbo_mlp = lbo_mlp
        self.eigva = np.load(f'{PROJDIR}/canonical_eigval3.npy')[1:].astype(np.float32)

        # TODO: the first hidden layer's weights should be regularized
        self.hidden_layers = nn.ModuleList([
            nn.Linear(layer_dims[i - 1] if i > 0 else FLAGS.field_posenc_dim, layer_dims[i])
            for i in range(len(layer_dims) - 1)
        ])
        self.output_layer = nn.Linear(layer_dims[-2], layer_dims[-1])
        self.out_dim = layer_dims[-1]
        self.r_sqrt_eigva = torch.sqrt(
            1.0 / torch.tensor(np.load(f'{PROJDIR}/canonical_eigval3.npy')[1:],
                               dtype=torch.float32))

    def forward(self, inp):
        lbo = self.lbo_mlp(inp.reshape(-1, 3))[..., :self.posenc_dim]
        inp_shape = inp.shape
        lbo = torch.reshape(lbo, inp_shape[:-1] + (self.posenc_dim,))
        lbo = lbo * self.r_sqrt_eigva[:self.posenc_dim] * 0.1

        x = lbo
        for layer in self.hidden_layers:
            x = nn.functional.gelu(layer(x))
        return self.output_layer(x)


class GPSBaseModel(nn.Module):
    def __init__(self, pos_enc_dim=512, hidden_dim=2048, output_dim=1024):
        super().__init__()
        self.pos_enc_dim = pos_enc_dim
        self.factor = 1 / np.sqrt(np.float32(self.pos_enc_dim))
        self.W_r = nn.Linear(3, self.pos_enc_dim // 2, bias=False)
        nn.init.normal_(self.W_r.weight, std=12)
        self.dense1 = nn.Linear(self.pos_enc_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, output_dim)

        nodes = np.load(f'{PROJDIR}/canonical_nodes3.npy')
        self.mini = torch.tensor(np.min(nodes, axis=0), dtype=torch.float32)
        self.maxi = torch.tensor(np.max(nodes, axis=0), dtype=torch.float32)
        self.center = (self.mini + self.maxi) / 2

    def forward(self, inp):
        x = (inp - self.center) / (self.maxi - self.mini)
        x = self.W_r(x)
        x = torch.sin(torch.cat([x, x + np.pi / 2], dim=-1)) * self.factor
        x = nn.functional.gelu(self.dense1(x))
        return self.dense2(x)
