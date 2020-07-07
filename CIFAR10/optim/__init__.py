from .ranger import *
from .radam import *


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
        optimizer = RAdam(params_dict, lr, weight_decay=weight_decay)
    elif(opt_name.lower() == "ranger"):
        optimizer = Ranger(params_dict, lr, weight_decay=weight_decay)
    else:
        raise ValueError("Optimizer type: ", opt_name, " is not supported or known")

    return optimizer