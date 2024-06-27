import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from src.layers_pooler import Merger
from src.utils import MLP

class LSTMDecoder(nn.Module):
    def __init__(self, dim_in, dim_out, curve_resolution, n_layers=5):
        super().__init__()
        self.lstm = \
            nn.LSTM(
                input_size=dim_out,
                hidden_size=dim_in,
                num_layers=n_layers,
                batch_first=True
            )
        self.curve_resolution = curve_resolution
        self.readout = nn.Linear(dim_in, dim_out)
        self.n_layers = n_layers

    def forward(self, x, y=None):
        # use teacher forcing
        bs, *_ = x.shape

        graph_emb = x.unsqueeze(0).repeat(self.n_layers, 1, 1)
        decoder_hidden = graph_emb, graph_emb
        if y is not None:
            decoder_input = \
                torch.cat([
                    torch.zeros(bs, 1, 1, device=x.device),
                    y.c_shape[:, :-1, :].detach()], dim=1)
            out, _ = self.lstm(decoder_input, decoder_hidden)
        else:
            decoder_input = torch.zeros(bs, 1, 1, device=x.device)
            out = []
            for t in range(self.curve_resolution):
                decoder_output, decoder_hidden = self.lstm(decoder_input, decoder_hidden)
                decoder_output = self.readout(decoder_output)
                out.append(decoder_output)
                decoder_input = decoder_output
                # decoder_input = y[:,t]
            out = torch.cat(out, dim=1)
        return out


# class MLPDecoder(nn.Module):
#     def __init__(self, dim_in, dim_out, curve_resolution, n_layers=5, dropout=0.1, act='ELU'):
#         super().__init__()
#         multipliers = np.arange(n_layers+1)/(n_layers)
#         dim_out_mlp = dim_out*curve_resolution
#         if dim_in < dim_out_mlp:
#             dim_li = multipliers*(dim_out_mlp-dim_in) + dim_in
#         else:
#             dim_li = multipliers[::-1]*(dim_in-dim_out_mlp) + dim_out_mlp
#         dim_li = dim_li.astype(int)
#
#         self.mlp = MLP(dim_li, act=act, dropout=dropout)
#         self.dim_out = dim_out
#         self.curve_resolution = curve_resolution
#
#     def forward(self, x, **kwargs):
#         bs, _ = x.shape
#         x = self.mlp(x)
#         x = x.reshape(bs, self.curve_resolution, self.dim_out)
#         return x

class CNNDecoder(nn.Module):
    def __init__(self, dim_in, dim_out, curve_resolution, upscale_rate=2, kernel_size=1, act='ReLU'):
        super().__init__()
        assert dim_in >= dim_out
        assert curve_resolution % upscale_rate == 0
        n_layers = int(np.log(curve_resolution) / np.log(upscale_rate))

        multipliers = np.arange(n_layers + 1)[::-1] / (n_layers)
        dim_li = multipliers * (dim_in - dim_out) + dim_out
        dim_li = dim_li.astype(int)

        layers = []
        for dim_in_lyr, dim_out_lyr in zip(dim_li[:-1], dim_li[1:]):
            layers.append(CNNUpscale(
                dim_in_lyr, dim_out_lyr,
                upscale_rate=upscale_rate, kernel_size=kernel_size))
        self.layers = nn.ModuleList(layers)
        self.act = getattr(nn, act)()

    def forward(self, x, **kwargs):
        x = x.unsqueeze(1)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act(x)
        x = self.layers[-1](x)
        return x


class CNNLSTMDecoderOriginal(nn.Module):
    def __init__(self, dim_in, dim_out, curve_resolution, merge_name, stride=2,
                 n_layers_cnn=3, use_nonlinearity=True, n_start_tokens=2, kernel_size=4,
                 n_layers_rnn=3, module_rnn='GRU'):
        super().__init__()
        self.fixed_upsample = nn.Upsample(size=curve_resolution, mode='nearest')
        self.merger = Merger(dim_in, dim_in, merge_name, num_inputs=3)

        assert kernel_size % 2 == 0
        cnn_kwargs = {
            'stride': stride,
            'kernel_size': kernel_size,
            'padding': int((kernel_size - stride) / 2)
        }
        n_layers_cnn_modified = max(0, n_layers_cnn)# - int(np.log2(n_start_tokens)))
        self.cnn = \
            CNNSameScale(
                dim_in, n_layers_cnn_modified, use_nonlinearity,
                n_start_tokens=n_start_tokens, **cnn_kwargs
            )
        self.lstm = RNNEncoder(dim_in, module_rnn, n_layers_rnn, dim_in)
        self.readout = nn.Linear(dim_in, dim_out)
        self.curve_resolution = curve_resolution

    def forward(self, emb_nodes, emb_edges, emb_rho, **kwargs):
        x = self.merger(emb_nodes, emb_edges, emb_rho)
        x = self.cnn(x)
        x = self.fixed_upsample(x)
        x = x.transpose(-1, -2)
        x = self.lstm(x)
        x = self.readout(x)
        return x


class CNNLSTMDecoder(nn.Module):
    def __init__(self, dim_in, dim_out, curve_resolution, upscale_rate=2, kernel_size=1, act='ReLU', n_layers_rnn=3,
                 module_rnn='GRU'):
        super().__init__()
        self.fixed_upsample = nn.Upsample(size=curve_resolution, mode='nearest')
        self.cnn = \
            CNNDecoder(dim_in, dim_in, curve_resolution, upscale_rate=upscale_rate, kernel_size=kernel_size, act=act)
        self.lstm = RNNEncoder(dim_in, module_rnn, n_layers_rnn, dim_in)
        self.readout = nn.Linear(dim_in, dim_out)
        self.curve_resolution = curve_resolution

    def forward(self, x, **kwargs):
        x = self.cnn(x)
        # x = self.fixed_upsample(x)
        x = self.lstm(x.transpose(-1, -2))
        x = self.readout(x)
        assert x.shape[1:] == self.curve_resolution, 1
        return x


class CNNUpscale(nn.Module):
    def __init__(self, dim_in, dim_out, upscale_rate=2, kernel_size=1):
        super().__init__()
        assert kernel_size % 2 == 1
        self.conv_li = \
            nn.ModuleList([
                nn.Conv1d(dim_in, dim_out, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2)) \
                for _ in range(upscale_rate)])

    def forward(self, x):
        '''
        :param x: B x L x C
        :return: B x L*U x C
        [[
            [[2], [1], [1]],
            [[1], [1], [2]],
            [[0], [0], [9]]
        ],[
            [[9], [9], [9]],
            [[7], [8], [7]],
            [[3], [2], [1]]
        ]]
        --> [[
            [[2], [9], [1], [9], [1], [9]],
            [[1], [7], [1], [8], [2], [7]],
            [[0], [3], [0], [2], [9], [1]]
        ]]
        '''
        out_li = []
        for conv in self.conv_li:
            out_li.append(conv(x.transpose(-1, -2)).transpose(-1, -2))
        out = torch.stack(out_li, dim=1)
        bs, u, l, c = out.shape
        out = out.transpose(1, 2).contiguous().reshape(bs, u * l, c)
        return out


class CNNSameScale(nn.Module):
    def __init__(self, dim_hidden, num_layers, use_nonlinearity,
                 n_start_tokens=2, kernel_size=3, stride=2, padding=1):
        super(CNNSameScale, self).__init__()
        self.n_start_tokens = n_start_tokens
        dim_li = [self.n_start_tokens]
        upsample_layer_li = []
        for _ in range(num_layers):
            upsample_layer_li.append(nn.Sequential(
                nn.ConvTranspose1d(
                    dim_hidden, dim_hidden,
                    kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm1d(dim_hidden),
                nn.ELU(),
                nn.Dropout1d(p=0.5),
            ))
            dim = (dim_li[-1] - 1) * stride - 2 * padding + kernel_size
            dim_li.append(dim)

        self.upsample_layer_li = nn.ModuleList(upsample_layer_li)
        self.dim_li = dim_li

        self.act = nn.ELU()  # TODO: do I need nonlinearities?
        self.use_nonlinearity = use_nonlinearity

    def forward(self, latent):
        # latent = latent.view(-1,1)
        latent = latent.unsqueeze(-1).repeat(1, 1, self.n_start_tokens)
        for i, upsample_layer in enumerate(self.upsample_layer_li):
            latent = upsample_layer(latent)
            if self.use_nonlinearity and i != len(self.upsample_layer_li) - 1:
                latent = self.act(latent)
        # todo: fIX DIMENSIONS
        return latent


class RNNSameScale(nn.Module):
    def __init__(self, module, dim_input, dim_hidden, dropout=0.2):
        super(RNNSameScale, self).__init__()
        # Setup
        self.out_dim = 2 * dim_hidden  # because bidirection
        self.dropout = dropout

        # Recurrent layer
        self.layer = getattr(nn, module)(
            dim_input, dim_hidden, bidirectional=True, num_layers=1, batch_first=True)

        # Regularizations
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)

    def forward(self, input_x):
        # Forward RNN
        if not self.training:
            self.layer.flatten_parameters()
        output, _ = self.layer(input_x)
        # Normalization
        if self.dropout > 0:
            output = self.dp(output)
        return output


class RNNEncoder(nn.Module):
    def __init__(self, dim_hidden, module, num_layers, dim_rnn):
        super(RNNEncoder, self).__init__()
        latest_size = dim_hidden
        rnns = []
        for _ in range(num_layers):
            rnn_layer = RNNSameScale(
                module,
                latest_size,
                dim_rnn // 2
            )
            latest_size = rnn_layer.out_dim
            rnns.append(rnn_layer)
        self.rnns = nn.ModuleList(rnns)

    def forward(self, curve):
        for rnn in self.rnns:
            curve = curve + rnn(curve)  # Note: LSTM implicitly does tanh activation
        return curve