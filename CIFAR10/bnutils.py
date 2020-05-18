import torch

def freeze_weights(model):
    for name, module in model._modules.items():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            module.weight.requires_grad = False
            model._modules[name] = module
        
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = freeze_weights(module)
    return model

def freeze_biases(model):
    for name, module in model._modules.items():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if module.bias is not None:
                module.bias.requires_grad = False
                model._modules[name] = module
        
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = freeze_biases(module)
    return model
    
def freeze_gamma(model):
    for name, module in model._modules.items():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            if module.weight is not None:
                module.weight.requires_grad = False
                model._modules[name] = module
        
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = freeze_gamma(module)
    return model
    
def freeze_beta(model):
    for name, module in model._modules.items():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            if module.bias is not None:
                module.bias.requires_grad = False
                model._modules[name] = module
                
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = freeze_beta(module)

    return model