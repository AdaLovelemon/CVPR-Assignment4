import torch
import torch.nn as nn
from torch import optim

def Criterion(config):
    name = config['Training']['criterion']
    if name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif name == 'MSELoss':
        return nn.MSELoss()
    elif name == 'L1Loss':
        return nn.L1Loss
    else:
        raise ValueError('Error Loss functions currently not available')
    
def Optimizer(config, model_parameters):
    name = config['Training']['optimizer']
    lr = config['Training']['lr']
    momentum = config['Training']['momentum']
    weight_decay = config['Training']['weight_decay']
    if name == 'Adam':
        return optim.Adam(model_parameters, lr, weight_decay=weight_decay)
    elif name == 'SGD':
        return optim.SGD(model_parameters, lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError('Error Optimizers currently not available')
    

def Scheduler(config, optimizer):
    name = config['Training']['lr_scheduler']
    step_size = config['Training']['step_size']
    gamma = config['Training']['gamma']
    if name == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == 'MultiStepLR':
        return optim.lr_scheduler.MultiStepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == 'ExponentialLR':
        return optim.lr_scheduler.ExponentialLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, step_size=step_size, gamma=gamma)
    elif name == 'CyclicLR':
        return optim.lr_scheduler.CyclicLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == 'OneCycleLR':
        return optim.lr_scheduler.OneCycleLR(optimizer, step_size=step_size, gamma=gamma)
    
    else:
        print("This Scheduler Not Existed")
        return None