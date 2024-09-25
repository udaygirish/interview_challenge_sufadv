import torch
import torch.nn as nn
import numpy as np
import os 
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from lib.masked_transformer_kvcache import MaskedTransformerEncoderLayer, MaskedTransformerEncoder
from lib.unmasked_transformer_kvcache import TransformerEncoderLayer, TransformerEncoder
from lib.helpers import adjacency_matrix_torch
from lib.model_helpers import PositionalEncoding

class BoTMixWithKVCache(nn.Module):
    """
    BoTMixWithKVCache is a neural network module that combines masked and unmasked transformer encoder layers
    with key-value caching for efficient sequence processing.
    Args:
        num_joints (int): Number of joints in the input data.
        d_model (int): Dimension of the model.
        nhead (int): Number of attention heads.
        num_layers (int): Number of encoder layers.
        dim_feedforward (int, optional): Dimension of the feedforward network. Default is 2048.
        dropout (float, optional): Dropout rate. Default is 0.1.
        adjacency_matrix (torch.Tensor, optional): Predefined adjacency matrix. If None, a default matrix is created.
        device (torch.device, optional): Device to run the model on. If None, it defaults to CUDA if available, otherwise CPU.
    Attributes:
        num_joints (int): Number of joints in the input data.
        num_layers (int): Number of encoder layers.
        tokenizer (nn.Linear): Linear layer for tokenizing input data.
        positional_encoding (PositionalEncoding): Positional encoding module.
        encoder (nn.ModuleList): List of transformer encoder layers.
        detokenizer (nn.Linear): Linear layer for detokenizing output data.
        adjacency_matrix (torch.Tensor): Adjacency matrix for masking.
        device (torch.device): Device to run the model on.
    Methods:
        create_mask(seq_length):
            Creates a mask for the input sequence based on the adjacency matrix.
        forward(x, kv_cache=None):
            Forward pass of the model. Processes the input sequence and returns the output and key-value cache.
    """
    def __init__(self, num_joints, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, adjacency_matrix=None, device=None):
        super().__init__()
        self.num_joints = num_joints
        self.num_layers = num_layers
        
        # Tokenizer
        self.tokenizer = nn.Linear(1, d_model)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # BoT-Mix Encoder
        masked_layer = MaskedTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        unmasked_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        
        self.encoder = nn.ModuleList([
            MaskedTransformerEncoder(masked_layer, 1) if i % 2 == 0
            else TransformerEncoder(unmasked_layer, 1)
            for i in range(num_layers)
        ])
        
        # Detokenizer
        self.detokenizer = nn.Linear(d_model, 1)
        
        if adjacency_matrix is not None:
            self.adjacency_matrix = adjacency_matrix
        else:
            self.adjacency_matrix = adjacency_matrix_torch(num_joints, gripper_joint=True, gripper_link_all_joints=True)
            
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
    
    def create_mask(self, seq_length):
        adjacency_matrix = self.adjacency_matrix
        mask = torch.eye(seq_length) + adjacency_matrix
        return mask.bool().to(self.device)
    
    def forward(self, x, kv_cache=None):
        """
        Forward pass for the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, 1).
            kv_cache (list, optional): List of key-value caches for each layer. Defaults to None.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length).
            list: Updated key-value caches for each layer.
        """
        # x shape: (batch_size, seq_length, 1)
        seq_length = x.size(1)
        mask = self.create_mask(seq_length).to(x.device)
        
        x = self.tokenizer(x)  # (batch_size, seq_length, d_model)
        x = self.positional_encoding(x)  # Add positional encoding
        
        if kv_cache is None:
            kv_cache = [None] * self.num_layers
        
        for i, layer in enumerate(self.encoder):
            if i % 2 == 0:  # Masked attention layer
                x = layer(x, mask=mask, kv_cache=kv_cache[i])
            else:  # Unmasked attention layer
                x = layer(x, src_key_padding_mask=None, kv_cache=kv_cache[i])
        
        x = self.detokenizer(x)  # (batch_size, seq_length, 1)
        return x.squeeze(-1), kv_cache  # (batch_size, seq_length)
