from functools import partial

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel


def select_layers(nb, num_layers):
    select = nb.split('_')[1]
    nb = int(nb.split('_')[0])
    if nb == 1:
        return [num_layers-1]
    elif nb == 4:
        if select == 'uniform':
            return [num_layers//4-1, num_layers//2-1, num_layers//4*3-1, num_layers-1]
        elif select == 'last':
            return [num_layers-4, num_layers-3, num_layers-2, num_layers-1]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def create_linear_input(x_tokens_list, multilayers, use_avgpool):
    intermediate_output = [x_tokens_list[idx] for idx in multilayers]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    def __init__(self, out_dim, multilayers, use_avgpool, heads):
        super().__init__()
        self.out_dim = out_dim
        self.multilayers = multilayers
        self.use_avgpool = use_avgpool
        self.heads = heads

        self.linear_heads = nn.ModuleDict({str(i): nn.Linear(out_dim, h) for i, h in enumerate(self.heads)})
        for i in range(len(self.heads)):
            self.linear_heads[str(i)].weight.data.normal_(mean=0.0, std=0.01)
            self.linear_heads[str(i)].bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.multilayers, self.use_avgpool)
        return [self.linear_heads[str(i)](output) for i in range(len(self.heads))]


class PoseDINOv2(nn.Module):
    def __init__(self, backbone, heads, layers, avgpool=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        self.autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)

        sample_input = torch.zeros((2, 3, 224, 224))
        sample_x_tokens_list = self.get_intermediate_layers(sample_input)
        self.multilayers = select_layers(layers, len(sample_x_tokens_list))

        sample_output = create_linear_input(sample_x_tokens_list, self.multilayers, avgpool)
        self.lc = LinearClassifier(sample_output.shape[-1], self.multilayers, avgpool, heads=heads)

    def get_intermediate_layers(self, x):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.backbone(x, output_hidden_states=True)['hidden_states'][1:]
        features = [[x[:, 1:, :], x[:, 0, :]] for x in features]
        return features

    def forward(self, x):
        x_tokens_list = self.get_intermediate_layers(x)
        return self.lc(x_tokens_list)


def bin_to_continuous(bins, num_bins, min_value=0.0, max_value=1.0, **kwargs):
    bins = bins % num_bins
    z = (bins + 0.5) / num_bins
    return z * (max_value - min_value) + min_value
