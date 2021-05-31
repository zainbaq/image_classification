import torch

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer == None:
        return model
    else:
        optimizer.load_state_dict['optimizer_state_dict']
        return model, optimizer

