import torch
import torch.nn as nn
import numpy as np
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from lib.masked_transformer import MaskedTransformerEncoderLayer, MaskedTransformerEncoder
from lib.helpers import adjacency_matrix_torch
from lib.model_helpers import PositionalEncoding

class BoTMix(nn.Module):
    """
    BoTMix is a neural network model that combines masked and unmasked transformer encoder layers
    for processing sequences of joint data.
    Args:
        num_joints (int): Number of joints in the input data.
        d_model (int): Dimension of the model.
        nhead (int): Number of heads in the multiheadattention models.
        num_layers (int): Number of encoder layers.
        dim_feedforward (int, optional): Dimension of the feedforward network model. Default is 2048.
        dropout (float, optional): Dropout value. Default is 0.1.
        adjacency_matrix (torch.Tensor, optional): Predefined adjacency matrix. Default is None.
        device (torch.device, optional): Device to run the model on. Default is None.
    Attributes:
        num_joints (int): Number of joints in the input data.
        tokenizer (nn.Linear): Linear layer to tokenize the input data.
        positional_encoding (PositionalEncoding): Positional encoding layer.
        encoder (nn.ModuleList): List of transformer encoder layers, alternating between masked and unmasked.
        detokenizer (nn.Linear): Linear layer to detokenize the output data.
        adjacency_matrix (torch.Tensor): Adjacency matrix for masking.
        device (torch.device): Device to run the model on.
    Methods:
        create_mask(seq_length):
            Creates a mask for the input sequence based on the adjacency matrix.
        forward(x):
            Forward pass of the model.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_length, 1).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, seq_length).
    """
    def __init__(self, num_joints, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, adjacency_matrix=None, device=None):
        super().__init__()
        self.num_joints = num_joints
        
        # Tokenizer
        self.tokenizer = nn.Linear(1, d_model)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # BoT-Mix Encoder
        masked_layer = MaskedTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        unmasked_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        
        self.encoder = nn.ModuleList([
            MaskedTransformerEncoder(masked_layer, 1) if i % 2 == 0
            else nn.TransformerEncoder(unmasked_layer, 1)
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
    
    def forward(self, x):
        """
        Perform a forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, 1).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length).
        The forward pass includes the following steps:
        1. Create a mask based on the sequence length and move it to the same device as the input tensor.
        2. Tokenize the input tensor.
        3. Add positional encoding to the tokenized input.
        4. Pass the encoded input through a series of encoder layers, alternating between masked and unmasked attention layers.
        5. Detokenize the output of the encoder layers.
        6. Squeeze the last dimension of the output tensor to match the expected output shape.
        """
        # x shape: (batch_size, seq_length, 1)
        seq_length = x.size(1)
        mask = self.create_mask(seq_length).to(x.device)
        
        x = self.tokenizer(x)  # (batch_size, seq_length, d_model)
        x = self.positional_encoding(x)  # Add positional encoding
        
        for i, layer in enumerate(self.encoder):
            if i % 2 == 0:  # Masked attention layer
                x = layer(x, mask=mask)
            else:  # Unmasked attention layer
                x = layer(x)
        
        x = self.detokenizer(x)  # (batch_size, seq_length, 1)
        return x.squeeze(-1)  # (batch_size, seq_length)
