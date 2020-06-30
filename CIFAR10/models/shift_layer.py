import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn import init
from torch.nn.parameter import Parameter

import math

# this function construct an additive pot quantization levels set, with clipping threshold = 1,
def build_power_value(B=2, additive=True, gridnorm=True):
    base_a = [0.]
    base_b = [0.]
    base_c = [0.]
    if additive:
        if B == 2:
            for i in range(3):
                base_a.append(2 ** (-i - 1))
        elif B == 4:
            for i in range(3):
                base_a.append(2 ** (-2 * i - 1))
                base_b.append(2 ** (-2 * i - 2))
        elif B == 6:
            for i in range(3):
                base_a.append(2 ** (-3 * i - 1))
                base_b.append(2 ** (-3 * i - 2))
                base_c.append(2 ** (-3 * i - 3))
        elif B == 3:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-i - 1))
                else:
                    base_b.append(2 ** (-i - 1))
                    base_a.append(2 ** (-i - 2))
        elif B == 5:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-2 * i - 1))
                    base_b.append(2 ** (-2 * i - 2))
                else:
                    base_c.append(2 ** (-2 * i - 1))
                    base_a.append(2 ** (-2 * i - 2))
                    base_b.append(2 ** (-2 * i - 3))
        else:
            pass
    else:
        for i in range(2 ** B - 1):
            base_a.append(2 ** (-i - 1))
    values = []
    for a in base_a:
        for b in base_b:
            for c in base_c:
                values.append((a + b + c))
    values = torch.Tensor(list(set(values)))
    if gridnorm:
        values = values.mul(1.0 / torch.max(values))
    return values


def weight_quantization(b, grids, train_alpha=True, gridnorm=True):

    def uniform_quant(x, b):
        xdiv = x.mul((2 ** b - 1))
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    def power_quant(x, value_s):
        shape = x.shape
        xhard = x.view(-1)
        value_s = value_s.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]  # project to nearest quantization level
        xhard = value_s[idxs].view(shape)
        # xout = (xhard - x).detach() + x
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            if gridnorm:
                input.div_(alpha)                           # weights are first divided by alpha
                input_c = input.clamp(min=-1, max=+1)       # then clipped to [-1,1]
            else:
                input_c = input.clamp(min=-alpha.item(), max=alpha.item())
            sign = input_c.sign()
            input_abs = input_c.abs()
            input_q = power_quant(input_abs, grids).mul(sign)  # project to Q^a(alpha, B)
            ctx.save_for_backward(input, input_q, alpha)
            if gridnorm:
                input_q = input_q.mul(alpha)               # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()             # grad for weights will not be clipped
            input, input_q, alpha = ctx.saved_tensors
            if gridnorm:
                i = (input.abs()>1.0).float()
            else:
                i = (input.abs()>alpha.item()).float()
            sign = input.sign()
            if train_alpha:
                grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
            else:
                grad_alpha = None
            return grad_input, grad_alpha

    return _pq().apply


class weight_shift_fn(nn.Module):
    def __init__(self, w_bit, power=True, additive=True, train_alpha=True, weightnorm=True, gridnorm=True, wgt_alpha_init=3.0):
        super(weight_shift_fn, self).__init__()
        assert (w_bit <=5 and w_bit > 0) or w_bit == 32
        self.w_bit = w_bit-1
        self.power = power if w_bit>2 else False
        self.grids = build_power_value(self.w_bit, additive=additive, gridnorm=gridnorm)
        self.weight_q = weight_quantization(b=self.w_bit, grids=self.grids, train_alpha=train_alpha, gridnorm=gridnorm)
        if train_alpha:
            self.register_parameter('wgt_alpha', Parameter(torch.tensor(wgt_alpha_init)))
        else:
            self.wgt_alpha = torch.tensor(wgt_alpha_init)
        self.weightnorm = weightnorm

    def forward(self, shift, sign):
        weight = 2**shift.round() * sign.round()
        if self.weightnorm:
            mean = weight.mean()
            std = weight.std()
            weight = weight.add(-mean).div(std)      # weights normalization
        weight_q = self.weight_q(weight, self.wgt_alpha)
        return weight_q


def act_quantization(b, grid, power=True, train_alpha=True, gridnorm=True):

    def uniform_quant(x, b=3):
        xdiv = x.mul(2 ** b - 1)
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    def power_quant(x, grid):
        shape = x.shape
        xhard = x.view(-1)
        value_s = grid.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
        xhard = value_s[idxs].view(shape)
        return xhard

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            if gridnorm:
                input=input.div(alpha)
                input_c = input.clamp(max=1.0)
            else:
                input_c = input.clamp(max=alpha.item())
            if power:
                input_q = power_quant(input_c, grid)
            else:
                input_q = uniform_quant(input_c, b)
            ctx.save_for_backward(input, input_q, alpha)
            if gridnorm:
                input_q = input_q.mul(alpha)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q, alpha = ctx.saved_tensors
            if gridnorm:
                i = (input > 1.0).float()
            else:
                i = (input > alpha.item()).float()
            if train_alpha:
                grad_alpha = (grad_output * (i + (input_q - input) * (1 - i))).sum()
            else:
                grad_alpha = None
            grad_input = grad_input*(1-i)
            return grad_input, grad_alpha

    return _uq().apply


class ShiftConv2d(nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros',
                       additive=True, train_alpha=True, weightnorm=True, gridnorm=True, wgt_alpha_init=3.0, act_alpha_init=8.0):
        super(ShiftConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        transposed = False
        output_padding = _pair(0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode


        self.layer_type = 'ShiftConv2d'
        self.bit = 4
        # TODO: remove redundancy as variables similar weight_shift_range is calculated in other functions in the code
        self.weight_shift_range = (-1 * (2**(self.bit - 1) - 1 -1), 0) # we use ternary weights to represent sign
        self.weight_quant = weight_shift_fn(w_bit=self.bit, power=True, additive=additive, train_alpha=train_alpha, weightnorm=weightnorm, gridnorm=gridnorm, wgt_alpha_init=wgt_alpha_init)
        self.act_grid = build_power_value(self.bit, additive=additive, gridnorm=gridnorm)
        self.act_alq = act_quantization(self.bit, self.act_grid, power=True, train_alpha=train_alpha, gridnorm=gridnorm)
        if train_alpha:
            self.act_alpha = torch.nn.Parameter(torch.tensor(act_alpha_init))
        else:
            self.act_alpha = torch.tensor(act_alpha_init)

        if transposed:
            self.shift = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.sign = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.shift = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.sign = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, x): 
        weight_q = 2**self.shift.round() * self.sign.round().sign()  # self.weight_quant(self.shift, self.sign)
        x = self.act_alq(x, self.act_alpha)
        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        self.shift.data.uniform_(*self.weight_shift_range) # (-0.1, 0.1)
        self.sign.data.uniform_(-1, 1) # (-0.1, 0.1)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.shift)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def show_params(self):
        wgt_alpha = round(self.weight_quant.wgt_alpha.data.item(), 3)
        act_alpha = round(self.act_alpha.data.item(), 3)
        print('clipping threshold weight alpha: {:2f}, activation alpha: {:2f}'.format(wgt_alpha, act_alpha))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, *args, **kargs):
        weight = state_dict[prefix + "weight"]
        #print(weight)
        shift = weight.abs().log() / math.log(2)
        sign = weight.sign()

        self.shift.data = shift
        self.sign.data = sign
        #state_dict[prefix + "shift"] = shift
        #state_dict[prefix + "sign"] = sign
        return super(ShiftConv2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, True, *args, **kargs)


# 8-bit quantization for the first and the last layer
class first_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(first_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.layer_type = 'FConv2d'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q-self.weight).detach()+self.weight
        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class last_fc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(last_fc, self).__init__(in_features, out_features, bias)
        self.layer_type = 'LFC'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q-self.weight).detach()+self.weight
        return F.linear(x, weight_q, self.bias)
