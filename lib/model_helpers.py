import torch 
import torch.nn as nn
import numpy as np

# Positional encoding class 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model,max_len =1000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x
    
class KVCache:
    def __init__(self, num_layers, num_heads, d_model):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.cache = [ {'key': None, 'value': None} for _ in range(num_layers) ]
    
    def update_cache(self, layer_idx, key, value):
        if self.cache[layer_idx]['key'] is None:
            self.cache[layer_idx]['key'] = key
            self.cache[layer_idx]['value'] = value
        else:
            self.cache[layer_idx]['key'] = torch.cat([self.cache[layer_idx]['key'], key], dim=1)
            self.cache[layer_idx]['value'] = torch.cat([self.cache[layer_idx]['value'], value], dim=1)
    
    def get_cache(self, layer_idx):
        return self.cache[layer_idx]['key'], self.cache[layer_idx]['value']
    
    def clear(self):
        self.cache = [ {'key': None, 'value': None} for _ in range(self.num_layers) ]