# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from nemo.utils import logging


def rnn(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    norm: Optional[str] = None,
    forget_gate_bias: Optional[float] = 1.0,
    dropout: Optional[float] = 0.0,
    norm_first_rnn: Optional[bool] = None,
    t_max: Optional[int] = None,
    weights_init_scale: float = 1.0,
    hidden_hidden_bias_scale: float = 0.0,
    proj_size: int = 0,
) -> torch.nn.Module:
    """
    Utility function to provide unified interface to common LSTM RNN modules.

    Args:
        input_size: Input dimension.

        hidden_size: Hidden dimension of the RNN.

        num_layers: Number of RNN layers.

        norm: Optional string representing type of normalization to apply to the RNN.
            Supported values are None, batch and layer.

        forget_gate_bias: float, set by default to 1.0, which constructs a forget gate
                initialized to 1.0.
                Reference:
                [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf)

        dropout: Optional dropout to apply to end of multi-layered RNN.

        norm_first_rnn: Whether to normalize the first RNN layer.

        t_max: int value, set to None by default. If an int is specified, performs Chrono Initialization
            of the LSTM network, based on the maximum number of timesteps `t_max` expected during the course
            of training.
            Reference:
            [Can recurrent neural networks warp time?](https://openreview.net/forum?id=SJcKhk-Ab)

        weights_init_scale: Float scale of the weights after initialization. Setting to lower than one
            sometimes helps reduce variance between runs.

        hidden_hidden_bias_scale: Float scale for the hidden-to-hidden bias scale. Set to 0.0 for
            the default behaviour.

    Returns:
        A RNN module
    """
    if norm not in [None, "batch", "layer"]:
        raise ValueError(f"unknown norm={norm}")

    if norm is None:
        return LSTMDropout(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            forget_gate_bias=forget_gate_bias,
            t_max=t_max,
            weights_init_scale=weights_init_scale,
            hidden_hidden_bias_scale=hidden_hidden_bias_scale,
            proj_size=proj_size,
        )

    if norm == "batch":
        return BNRNNSum(
            input_size=input_size,
            hidden_size=hidden_size,
            rnn_layers=num_layers,
            batch_norm=True,
            dropout=dropout,
            forget_gate_bias=forget_gate_bias,
            t_max=t_max,
            norm_first_rnn=norm_first_rnn,
            weights_init_scale=weights_init_scale,
            hidden_hidden_bias_scale=hidden_hidden_bias_scale,
            proj_size=proj_size,
        )

    if norm == "layer":
        return torch.jit.script(
            ln_lstm(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                forget_gate_bias=forget_gate_bias,
                t_max=t_max,
                weights_init_scale=weights_init_scale,
                hidden_hidden_bias_scale=hidden_hidden_bias_scale,
            )
        )


class OverLastDim(torch.nn.Module):
    """Collapses a tensor to 2D, applies a module, and (re-)expands the tensor.
    An n-dimensional tensor of shape (s_1, s_2, ..., s_n) is first collapsed to
    a tensor with shape (s_1*s_2*...*s_n-1, s_n). The module is called with
    this as input producing (s_1*s_2*...*s_n-1, s_n') --- note that the final
    dimension can change. This is expanded to (s_1, s_2, ..., s_n-1, s_n') and
    returned.
    Args:
        module (torch.nn.Module): Module to apply. Must accept a 2D tensor as
            input and produce a 2D tensor as output, optionally changing the
            size of the last dimension.
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *dims, _ = x.size()

        reduced_dims = 1
        for dim in dims:
            reduced_dims *= dim

        x = x.view(reduced_dims, -1)
        x = self.module(x)
        x = x.view(*dims, -1)
        return x


class ResetMaskLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, 
                 num_layers: int = 1, bias: bool = True,
                 batch_first: bool = False, dropout: float = 0.0,
                 proj_size: int = 0, device=None, dtype=None):
        super(ResetMaskLSTM, self).__init__()
        
        # Store settings
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.proj_size = proj_size
        
        # Create underlying LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=False,  # As specified, always False
            proj_size=proj_size,
            device=device,
            dtype=dtype
        )
        
        # Create trainable initial states
        output_size = proj_size if proj_size > 0 else hidden_size
        self.h0 = nn.Parameter(torch.zeros(num_layers, 1, output_size, device=device, dtype=dtype))
        self.c0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size, device=device, dtype=dtype))
        
    def forward(self, input: torch.Tensor, 
                reset_mask: torch.Tensor,
                hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with reset mask capability.
        
        Args:
            input: Input sequence tensor of shape (seq_len, batch, input_size) or 
                  (batch, seq_len, input_size) if batch_first=True
            reset_mask: Binary mask of shape (seq_len, batch) or (batch, seq_len) 
                       indicating positions to reset hidden states
            hx: Initial hidden state tuple (h_0, c_0) or None
                
        Returns:
            output: Tensor of shape (seq_len, batch, hidden_size) or 
                    (batch, seq_len, hidden_size) if batch_first=True
            hidden: Tuple of tensors (h_n, c_n) representing the final hidden state
        """
        # Get dimensions
        if self.batch_first:
            seq_len = input.size(1)
            batch_size = input.size(0)
            mask_permute = (1, 0)  # For adjusting mask if needed
        else:
            seq_len = input.size(0)
            batch_size = input.size(1)
            mask_permute = (0, 1)  # Keep mask as is

        
        # Make sure reset_mask matches expected format
        if reset_mask.dim() != 2:
            raise ValueError("reset_mask should be 2D with shape (seq_len, batch) or (batch, seq_len)")

#        reset_mask.fill_(False)
            
        # Ensure reset_mask has the right shape for processing
        if self.batch_first and reset_mask.size(0) == seq_len and reset_mask.size(1) == batch_size:
            # Transpose if mask is in (seq_len, batch) but batch_first=True
            reset_mask = reset_mask.transpose(0, 1)
        elif not self.batch_first and reset_mask.size(0) == batch_size and reset_mask.size(1) == seq_len:
            # Transpose if mask is in (batch, seq_len) but batch_first=False
            reset_mask = reset_mask.transpose(0, 1)

#        print("THREE DIMENSION", input.shape, reset_mask.shape)
#        print("seq_len, batch_size", seq_len, batch_size)
        
        # Initialize hidden state if not provided
        if hx is None:
            h = self.h0.expand(self.num_layers, batch_size, -1).contiguous()
            c = self.c0.expand(self.num_layers, batch_size, -1).contiguous()
            hx = (h, c)
        
        # Process sequence step by step
        outputs = []
        h, c = hx
        
        for t in range(seq_len):
            # Extract current input step
            if self.batch_first:
                x_t = input[:, t:t+1, :]  # Keep time dimension for LSTM
            else:
                x_t = input[t:t+1, :, :]
            
            # Check reset mask for current step
            if self.batch_first:
                mask_t = reset_mask[:, t]
            else:
                mask_t = reset_mask[t, :]
            
            # Reset hidden states according to mask
            # For positions where mask is True (1), use initial state
            mask_expanded_h = mask_t.unsqueeze(0).unsqueeze(2).expand_as(h).bool()
            mask_expanded_c = mask_t.unsqueeze(0).unsqueeze(2).expand_as(c).bool()
            
            h_reset = self.h0.expand(self.num_layers, batch_size, -1)
            c_reset = self.c0.expand(self.num_layers, batch_size, -1)
            
            h = torch.where(mask_expanded_h, h_reset, h)
            c = torch.where(mask_expanded_c, c_reset, c)
            
            # Run LSTM cell for current timestep
            out_t, (h, c) = self.lstm(x_t, (h, c))
            
            # Store output
            outputs.append(out_t)
        
        # Concatenate outputs
        if self.batch_first:
            output = torch.cat(outputs, dim=1)
        else:
            output = torch.cat(outputs, dim=0)
        
        return output, (h, c)
    
    def flatten_parameters(self):
        """
        Calls LSTM's flatten_parameters method to enable faster execution.
        """
        self.lstm.flatten_parameters()


class LSTMDropout(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: Optional[float],
        forget_gate_bias: Optional[float],
        t_max: Optional[int] = None,
        weights_init_scale: float = 1.0,
        hidden_hidden_bias_scale: float = 0.0,
        proj_size: int = 0,
    ):
        """Returns an LSTM with forget gate bias init to `forget_gate_bias`.
        Args:
            input_size: See `torch.nn.LSTM`.
            hidden_size: See `torch.nn.LSTM`.
            num_layers: See `torch.nn.LSTM`.
            dropout: See `torch.nn.LSTM`.

            forget_gate_bias: float, set by default to 1.0, which constructs a forget gate
                initialized to 1.0.
                Reference:
                [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf)

            t_max: int value, set to None by default. If an int is specified, performs Chrono Initialization
                of the LSTM network, based on the maximum number of timesteps `t_max` expected during the course
                of training.
                Reference:
                [Can recurrent neural networks warp time?](https://openreview.net/forum?id=SJcKhk-Ab)

            weights_init_scale: Float scale of the weights after initialization. Setting to lower than one
                sometimes helps reduce variance between runs.

            hidden_hidden_bias_scale: Float scale for the hidden-to-hidden bias scale. Set to 0.0 for
                the default behaviour.

        Returns:
            A `torch.nn.LSTM`.
        """
        super(LSTMDropout, self).__init__()

        self.lstm = ResetMaskLSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, proj_size=proj_size
        )

#        if t_max is not None:
#            # apply chrono init
#            for name, v in self.lstm.named_parameters():
#                if 'bias' in name:
#                    p = getattr(self.lstm, name)
#                    n = p.nelement()
#                    hidden_size = n // 4
#                    p.data.fill_(0)
#                    p.data[hidden_size : 2 * hidden_size] = torch.log(
#                        torch.nn.init.uniform_(p.data[0:hidden_size], 1, t_max - 1)
#                    )
#                    # forget gate biases = log(uniform(1, Tmax-1))
#                    p.data[0:hidden_size] = -p.data[hidden_size : 2 * hidden_size]
#                    # input gate biases = -(forget gate biases)
#
#        elif forget_gate_bias is not None:
#            for name, v in self.lstm.named_parameters():
#                if "bias_ih" in name:
#                    bias = getattr(self.lstm, name)
#                    bias.data[hidden_size : 2 * hidden_size].fill_(forget_gate_bias)
#                if "bias_hh" in name:
#                    bias = getattr(self.lstm, name)
#                    bias.data[hidden_size : 2 * hidden_size] *= float(hidden_hidden_bias_scale)

        self.dropout = torch.nn.Dropout(dropout) if dropout else None

        for name, v in self.named_parameters():
            if 'weight' in name or 'bias' in name:
                v.data *= float(weights_init_scale)

    def forward(
        self, x: torch.Tensor, reset_mask: torch.Tensor, h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, h = self.lstm(input=x, reset_mask=reset_mask, hx=h)

        if self.dropout:
            x = self.dropout(x)

        return x, h


class RNNLayer(torch.nn.Module):
    """A single RNNLayer with optional batch norm."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: torch.nn.Module = torch.nn.LSTM,
        batch_norm: bool = True,
        forget_gate_bias: Optional[float] = 1.0,
        t_max: Optional[int] = None,
        weights_init_scale: float = 1.0,
        hidden_hidden_bias_scale: float = 0.0,
        proj_size: int = 0,
    ):
        super().__init__()

        if batch_norm:
            self.bn = OverLastDim(torch.nn.BatchNorm1d(input_size))

        if isinstance(rnn_type, torch.nn.LSTM) and not batch_norm:
            # batch_norm will apply bias, no need to add a second to LSTM
            self.rnn = LSTMDropout(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                dropout=0.0,
                forget_gate_bias=forget_gate_bias,
                t_max=t_max,
                weights_init_scale=weights_init_scale,
                hidden_hidden_bias_scale=hidden_hidden_bias_scale,
                proj_size=proj_size,
            )
        else:
            self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size, bias=not batch_norm)

    def forward(
        self, x: torch.Tensor, hx: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if hasattr(self, 'bn'):
            x = x.contiguous()
            x = self.bn(x)
        x, h = self.rnn(x, hx=hx)
        return x, h

    def _flatten_parameters(self):
        self.rnn.flatten_parameters()


class BNRNNSum(torch.nn.Module):
    """RNN wrapper with optional batch norm.
    Instantiates an RNN. If it is an LSTM it initialises the forget gate
    bias =`lstm_gate_bias`. Optionally applies a batch normalisation layer to
    the input with the statistics computed over all time steps.  If dropout > 0
    then it is applied to all layer outputs except the last.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: torch.nn.Module = torch.nn.LSTM,
        rnn_layers: int = 1,
        batch_norm: bool = True,
        dropout: Optional[float] = 0.0,
        forget_gate_bias: Optional[float] = 1.0,
        norm_first_rnn: bool = False,
        t_max: Optional[int] = None,
        weights_init_scale: float = 1.0,
        hidden_hidden_bias_scale: float = 0.0,
        proj_size: int = 0,
    ):
        super().__init__()
        self.rnn_layers = rnn_layers

        self.layers = torch.nn.ModuleList()
        for i in range(rnn_layers):
            final_layer = (rnn_layers - 1) == i

            self.layers.append(
                RNNLayer(
                    input_size,
                    hidden_size,
                    rnn_type=rnn_type,
                    batch_norm=batch_norm and (norm_first_rnn or i > 0),
                    forget_gate_bias=forget_gate_bias,
                    t_max=t_max,
                    weights_init_scale=weights_init_scale,
                    hidden_hidden_bias_scale=hidden_hidden_bias_scale,
                    proj_size=proj_size,
                )
            )

            if dropout is not None and dropout > 0.0 and not final_layer:
                self.layers.append(torch.nn.Dropout(dropout))

            input_size = hidden_size

    def forward(
        self, x: torch.Tensor, hx: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx = self._parse_hidden_state(hx)

        hs = []
        cs = []
        rnn_idx = 0
        for layer in self.layers:
            if isinstance(layer, torch.nn.Dropout):
                x = layer(x)
            else:
                x, h_out = layer(x, hx=hx[rnn_idx])
                hs.append(h_out[0])
                cs.append(h_out[1])
                rnn_idx += 1
                del h_out

        h_0 = torch.stack(hs, dim=0)
        c_0 = torch.stack(cs, dim=0)
        return x, (h_0, c_0)

    def _parse_hidden_state(
        self, hx: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[List[None], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Dealing w. hidden state:
        Typically in pytorch: (h_0, c_0)
            h_0 = ``[num_layers * num_directions, batch, hidden_size]``
            c_0 = ``[num_layers * num_directions, batch, hidden_size]``
        """
        if hx is None:
            return [None] * self.rnn_layers
        else:
            h_0, c_0 = hx

            if h_0.shape[0] != self.rnn_layers:
                raise ValueError(
                    'Provided initial state value `h_0` must be of shape : '
                    '[num_layers * num_directions, batch, hidden_size]'
                )

            return [(h_0[i], c_0[i]) for i in range(h_0.shape[0])]

    def _flatten_parameters(self):
        for layer in self.layers:
            if isinstance(layer, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
                layer._flatten_parameters()


class StackTime(torch.nn.Module):
    """
    Stacks time within the feature dim, so as to behave as a downsampling operation.
    """

    def __init__(self, factor: int):
        super().__init__()
        self.factor = int(factor)

    def forward(self, x: List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        # T, B, U
        x, x_lens = x
        seq = [x]
        for i in range(1, self.factor):
            tmp = torch.zeros_like(x)
            tmp[:-i, :, :] = x[i:, :, :]
            seq.append(tmp)
        x_lens = torch.ceil(x_lens.float() / self.factor).int()
        return torch.cat(seq, dim=2)[:: self.factor, :, :], x_lens


def ln_lstm(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: Optional[float],
    forget_gate_bias: Optional[float],
    t_max: Optional[int],
    weights_init_scale: Optional[float] = None,  # ignored
    hidden_hidden_bias_scale: Optional[float] = None,  # ignored
) -> torch.nn.Module:
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""
    # The following are not implemented.
    if dropout is not None and dropout != 0.0:
        raise ValueError('`dropout` not supported with LayerNormLSTM')

    if t_max is not None:
        logging.warning("LayerNormLSTM does not support chrono init via `t_max`")

    if weights_init_scale is not None:
        logging.warning("`weights_init_scale` is ignored for LayerNormLSTM")

    if hidden_hidden_bias_scale is not None:
        logging.warning("`hidden_hidden_bias_scale` is ignored for LayerNormLSTM")

    return StackedLSTM(
        num_layers,
        LSTMLayer,
        first_layer_args=[LayerNormLSTMCell, input_size, hidden_size, forget_gate_bias],
        other_layer_args=[LayerNormLSTMCell, hidden_size, hidden_size, forget_gate_bias],
    )


class LSTMLayer(torch.nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(
        self, input: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class LayerNormLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, forget_gate_bias):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.randn(4 * hidden_size, hidden_size))

        # LayerNorm provide learnable biases
        self.layernorm_i = torch.nn.LayerNorm(4 * hidden_size)
        self.layernorm_h = torch.nn.LayerNorm(4 * hidden_size)
        self.layernorm_c = torch.nn.LayerNorm(hidden_size)

        self.reset_parameters()

        self.layernorm_i.bias.data[hidden_size : 2 * hidden_size].fill_(0.0)
        self.layernorm_h.bias.data[hidden_size : 2 * hidden_size].fill_(forget_gate_bias)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(
        self, input: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = state
        igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


def init_stacked_lstm(
    num_layers: int, layer: torch.nn.Module, first_layer_args: List, other_layer_args: List
) -> torch.nn.ModuleList:
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args) for _ in range(num_layers - 1)]
    return torch.nn.ModuleList(layers)


class StackedLSTM(torch.nn.Module):
    def __init__(self, num_layers: int, layer: torch.nn.Module, first_layer_args: List, other_layer_args: List):
        super(StackedLSTM, self).__init__()
        self.layers: torch.nn.ModuleList = init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args)

    def forward(
        self, input: torch.Tensor, states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        if states is None:
            temp_states: List[Tuple[torch.Tensor, torch.Tensor]] = []
            batch = input.size(1)
            for layer in self.layers:
                temp_states.append(
                    (
                        torch.zeros(batch, layer.cell.hidden_size, dtype=input.dtype, device=input.device),
                        torch.zeros(batch, layer.cell.hidden_size, dtype=input.dtype, device=input.device),
                    )
                )

            states = temp_states

        output_states: List[Tuple[torch.Tensor, torch.Tensor]] = []
        output = input
        for i, rnn_layer in enumerate(self.layers):
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states.append(out_state)
            i += 1
        return output, output_states


def label_collate(labels, device=None):
    """Collates the label inputs for the rnn-t prediction network.
    If `labels` is already in torch.Tensor form this is a no-op.

    Args:
        labels: A torch.Tensor List of label indexes or a torch.Tensor.
        device: Optional torch device to place the label on.

    Returns:
        A padded torch.Tensor of shape (batch, max_seq_len).
    """

    if isinstance(labels, torch.Tensor):
        return labels.type(torch.int64)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(f"`labels` should be a list or tensor not {type(labels)}")

    batch_size = len(labels)
    max_len = max(len(label) for label in labels)

    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
    for e, l in enumerate(labels):
        cat_labels[e, : len(l)] = l
    labels = torch.tensor(cat_labels, dtype=torch.int64, device=device)

    return labels
