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