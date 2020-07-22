import torch
import numpy as np
import math

def round_to_fixed(input, fraction=16, integer=16): 
    assert integer >= 1, integer 
    if integer == 1: 
        return torch.sign(input) - 1 
    delta = math.pow(2.0, -(fraction))
    bound = math.pow(2.0, integer-1) 
    min_val = - bound 
    max_val = bound - 1 
    rounded = torch.floor(input / delta) * delta

    clipped_value = torch.clamp(rounded, min_val, max_val)
    return clipped_value 

def get_shift_and_sign(x, rounding='deterministic'):
    sign = torch.sign(x)
    
    x_abs = torch.abs(x)
    shift = round(torch.log(x_abs) / np.log(2), rounding)
    
    return shift, sign    

def round_power_of_2(x, rounding='deterministic'):
    shift, sign = get_shift_and_sign(x, rounding)    
    x_rounded = (2.0 ** shift) * sign
    return x_rounded

def round(x, rounding='deterministic'):
    assert(rounding in ['deterministic', 'stochastic'])
    if rounding == 'stochastic':
        x_floor = x.floor()
        return x_floor + torch.bernoulli(x - x_floor)
    else:
        return x.round()

def shift_l2_norm(opt, weight_decay):
    shift_params = opt.param_groups[2]['params']
    l2_norm = 0
    for shift in shift_params:
        l2_norm += torch.sum((2**shift)**2)
    return weight_decay * 0.5 * l2_norm

def optim(opt_name, params_dict, lr, momentum, weight_decay):
    # define optimizer
    optimizer = None 
    if(opt_name.lower() == "sgd"):
        optimizer = torch.optim.SGD(params_dict, lr, momentum=momentum, weight_decay=weight_decay)
    elif(opt_name.lower() == "adadelta"):
        optimizer = torch.optim.Adadelta(params_dict, lr, weight_decay=weight_decay)
    elif(opt_name.lower() == "adagrad"):
        optimizer = torch.optim.Adagrad(params_dict, lr, weight_decay=weight_decay)
    elif(opt_name.lower() == "adam"):
        optimizer = torch.optim.Adam(params_dict, lr, weight_decay=weight_decay)
    elif(opt_name.lower() == "rmsprop"):
        optimizer = torch.optim.RMSprop(params_dict, lr, momentum=momentum, weight_decay=weight_decay)
    elif(opt_name.lower() == "radam"):
        optimizer = optim.RAdam(params_dict, lr, weight_decay=weight_decay)
    elif(opt_name.lower() == "ranger"):
        optimizer = optim.Ranger(params_dict, lr, weight_decay=weight_decay)
    else:
        raise ValueError("Optimizer type: ", opt_name, " is not supported or known")

    return optimizer