import torch 
import torch.nn as nn
import numpy as np
from torchinfo import summary


# Prevent Python from genrating a .pyc file
import sys 
sys.dont_write_bytecode = True

# Import the Utils from abs path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.logger import logger
from lib.model_helpers import PositionalEncoding
    

# BoT-Mix Transformer Model
class BoTMix(nn.Module):
    """
    BoT-Mix Transformer Model - Custom PyTorch Implementation Using Transformer Encoder Layer.
    Args:
        num_joints (int): Number of joints in the input data.
        d_model (int): Dimension of the model.
        nhead (int): Number of heads in the multi head attention models.
        num_layers (int): Number of sub-encoder-layers in the encoder.
        feedforward_dim (int): Dimension of the feedforward network model.
        adjacency_matrix (torch.Tensor): Adjacency matrix representing the joint connections.
        dropout (float, optional): Dropout value. Default is 0.1.
        pos_encoding (bool, optional): Whether to use positional encoding. Default is True.
    Attributes:
        description (str): Description of the model.
        num_joints (int): Number of joints in the input data.
        embedding (nn.Linear): Linear layer for embedding the input joints.
        pos_encoding (bool): Whether to use positional encoding.
        pos_encoder (PositionalEncoding, optional): Positional encoding layer.
        mask (torch.Tensor): Mask created from the adjacency matrix.
        transformer_encoder (nn.TransformerEncoder): Transformer encoder composed of multiple layers.
        linear (nn.Linear): Linear layer to map the output to the number of joints.
    Methods:
        forward(x):
            Forward pass of the model.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, num_joints) or (batch_size, seq_len, num_joints).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, num_joints).
    """
    
    def __init__(self, num_joints, d_model, nhead, num_layers, feedforward_dim, adjacency_matrix, dropout=0.1, pos_encoding = True):
        super(BoTMix, self).__init__()
        self.description = "BoT-Mix Transformer Model - Custom PyTorch Implementation Using Transformer Encoder Layer"
        self.num_joints = num_joints
        self.embedding = nn.Linear(num_joints, d_model)
        self.pos_encoding = pos_encoding
        if self.pos_encoding:
            self.pos_encoder = PositionalEncoding(d_model)
        
        # Create the Mask from the Joint Adjacency Matrix as explained in Paper 
        mask = adjacency_matrix.float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        # Register the mask as a buffer - To show in the torch script
        self.register_buffer('mask', mask)


        # Create the Transformer Encoder
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=num_layers
        )

        self.linear = nn.Linear(d_model, num_joints)

    def forward(self, x):
        
        # Adding sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, num_joints) 
        
        # Embedding
        x = self.embedding(x)

        # Output Embedding shape - (batch_size, seq_len, d_model)
        if self.pos_encoding:
            x = self.pos_encoder(x)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Output shape - (batch_size, seq_len, num_joints)
        x = self.linear(x)

        # Return shape - (batch_size, num_joints)
        return x.squeeze(1)
    

# Main Function
# Test the Model
def main():
    # Test the Model with Model  Summary
    random_adjacency_matrix = np.random.randint(2, size=(7, 7))
    model = BoTMix(
        num_joints=7,
        d_model=64,
        nhead=4,
        num_layers=3,
        feedforward_dim=128,
        adjacency_matrix=torch.tensor(random_adjacency_matrix),
        dropout=0.1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(32, 7)
    out = model(x)
    logger.info("Model Input Shape: " + str(x.shape))
    logger.info("Model Output Shape: " + str(out.shape))

    # Summary of the Model
    logger.info("=" * 20)
    logger.info("Model Summary: ")
    
    summary(model.to(device), (32,7))
    logger.info("=" * 20)
    
if __name__ == "__main__":
    main()

