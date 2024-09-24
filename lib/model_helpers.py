import torch 
import torch.nn as nn
import numpy as np

# Positional encoding class 
class PositionalEncoding(nn.Module):
    """
    PositionalEncoding module adds positional encodings to the input tensor.
    Attributes:
        pe (torch.Tensor): A buffer that holds the positional encodings.

    Methods:
        __init__(d_model, max_len=1000):
            Initializes the PositionalEncoding module.
            
            Args:
                d_model (int): The dimension of the model.
                max_len (int, optional): The maximum length of the sequence. Default is 1000.

        forward(x):
            Adds positional encoding to the input tensor.
            
            Args:
                x (torch.Tensor): The input tensor of shape (seq_len, batch_size, d_model).
            
            Returns:
                torch.Tensor: The input tensor with positional encodings added.
    """
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
    """
    A class to manage key-value caches for each layer in a multi-layer model.
    Attributes:
    -----------
    num_layers : int
        The number of layers in the model.
    num_heads : int
        The number of attention heads in the model.
    d_model : int
        The dimensionality of the model.
    cache : list of dict
        A list of dictionaries, each containing 'key' and 'value' entries for each layer.
    Methods:
    --------
    update_cache(layer_idx, key, value):
        Updates the cache for a specific layer with new key and value tensors.
    get_cache(layer_idx):
        Retrieves the key and value tensors from the cache for a specific layer.
    clear():
        Clears the cache for all layers, resetting keys and values to None.
    """
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