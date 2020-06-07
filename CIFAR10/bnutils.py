import torch
import copy

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
    
def get_batchnorm_layers(model):
    bnlayers = []
    for name, module in model._modules.items():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            bnlayers.append(module)
                
        if len(list(module.children())) > 0:
            # recurse
            bnlayers.extend(get_batchnorm_layers(module))

    return bnlayers  
    
def update_mean_var(model, dataloader, device=torch.device("cuda")):
    for bnlayer in get_batchnorm_layers(model):
        update_layer_mean_var(model, bnlayer, dataloader, device)
        print("\t\tCompleted one layer")
        
    return model

def update_layer_mean_var(model, bnlayer, dataloader, device=torch.device("cuda")):
    def get_mean_var_hook(self, inputs):
        global mean_iter
        global iter_size
        mean_iter = inputs[0].mean([0, 2, 3])
        iter_size = len(inputs[0])
        
    
    mean_sum = 0
    data_size = 0
    model.eval()
    handle = bnlayer.register_forward_pre_hook(get_mean_var_hook)
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            model(inputs)
            
            mean_sum += mean_iter
            data_size += iter_size
            
    mean = mean_sum / data_size
    bnlayer.running_mean = mean
    print("\tUpdated mean")
    handle.remove()
    
    def get_deviation_var_hook(self, inputs):
        global deviation_iter
        deviation_iter = ((inputs[0] - mean.view(1,-1,1,1))**2).mean([0, 2, 3])
      
    sum_deviation = 0   
    handle = bnlayer.register_forward_pre_hook(get_deviation_var_hook)   
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            model(inputs)
            
            sum_deviation += deviation_iter
    
    var = sum_deviation / data_size
    bnlayer.running_var = var
    print("\tUpdated var")
    handle.remove()