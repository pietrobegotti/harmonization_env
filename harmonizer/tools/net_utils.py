import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class NetUtils(nn.Module):

    def __init__(self):
        
        super(NetUtils, self).__init__()

        self.blocks = {}
        self.type_ = ""
        
    def _reinitialize_weights(self, block):
        """Reinitialize weights for the specified block(s)"""
        
        elems = self.blocks[block]
        if not isinstance(elems, list): elems = [elems]
        
        for e in elems:
            # Iterate through all modules, including those inside Sequential
            for module in e.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')  
                    nn.init.zeros_(module.bias) 
                elif isinstance(module, nn.BatchNorm1d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def time_embed(self, t):
    
        if t.dim() != 2 or t.size(1) != 1: raise ValueError(f"Input tensor must have shape (batch_size, 1), but got {t.shape}")

        return torch.cat([
            torch.sin(t * (torch.pi / 2)),
            torch.cos(t * (torch.pi / 2)),
            torch.sin(t * (2 * torch.pi / 3)),
            torch.cos(t * (2 * torch.pi / 3))
        ], dim = 1)
    
    def one_hot(self, x, num_classes):

        mask = (x != -1).float() # placeholder -1 is output as the 0 vector

        valid_x = torch.where(x == -1, torch.zeros_like(x), x)
        one_hot_tensor = F.one_hot(valid_x.long(), num_classes = num_classes).to(dtype = torch.float32)

        return one_hot_tensor * mask.unsqueeze(-1)
    
    def save(self, path = None):
        
        if path == None: path = self.path
        torch.save(self.state_dict(), path)
 
    @classmethod
    def load_from_checkpoint(cls, path, device = 'cpu', **kwargs):
        
        if not os.path.exists(path): raise FileNotFoundError(f"Model file not found: {path}")
        
        model = cls(path = path, device = device, **kwargs)
        state_dict = torch.load(path, map_location = device)
        
        # Handle different save formats
        if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict: state_dict = state_dict['state_dict']
            
        model.load_state_dict(state_dict)
        model.eval()
        
        return model