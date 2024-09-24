import torch 
import torch.nn as nn
import numpy as np
from torchinfo import summary

# Prevent Python from generating a .pyc file
import sys 
sys.dont_write_bytecode = True

# Import the Utils from abs path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.logger import logger
from lib.model_helpers import PositionalEncoding, KVCache



class MultiheadAttentionWithKVCache(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiheadAttentionWithKVCache, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    
    def forward(self, query, key, value, layer_idx, cache: KVCache = None, need_weights=False):
        if cache is not None:
            cached_key, cached_value = cache.get_cache(layer_idx)
            if cached_key is not None and cached_value is not None:
                key = torch.cat([cached_key, key], dim=1)
                value = torch.cat([cached_value, value], dim=1)
            cache.update_cache(layer_idx, key, value)
        
        attn_output, attn_weights = self.multihead_attn(query, key, value, need_weights=need_weights)
        return attn_output, attn_weights

class TransformerEncoderLayerWithKVCache(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayerWithKVCache, self).__init__()
        self.self_attn = MultiheadAttentionWithKVCache(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src, layer_idx, cache: KVCache = None):
        src2, _ = self.self_attn(src, src, src, layer_idx, cache=cache, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoderWithKVCache(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoderWithKVCache, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(self, src, cache: KVCache = None):
        for layer_idx, layer in enumerate(self.layers):
            if cache is not None:
                src = layer(src, layer_idx, cache)
            else:
                src = layer(src, layer_idx)
        return src

class BoTMixWithKVCache(nn.Module):
    
    def __init__(self, num_joints, d_model, nhead, num_layers, feedforward_dim, adjacency_matrix, dropout=0.1, pos_encoding=True):
        super(BoTMixWithKVCache, self).__init__()
        self.description = "BoT-Mix Transformer Model with KV Cache - Custom PyTorch Implementation Using Transformer Encoder Layer"
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

        # Initialize the Custom Transformer Encoder with KV Cache
        encoder_layer = TransformerEncoderLayerWithKVCache(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=dropout
        )
        self.transformer_encoder = TransformerEncoderWithKVCache(
            encoder_layer,
            num_layers=num_layers
        )

        self.linear = nn.Linear(d_model, num_joints)
        self.num_layers = num_layers
        self.kv_cache = KVCache(num_layers, nhead, d_model)
    
    def forward(self, x, use_cache=False):
        # Adding sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, num_joints) 
        
        # Embedding
        x = self.embedding(x)

        # Output Embedding shape - (batch_size, seq_len, d_model)
        if self.pos_encoding:
            x = self.pos_encoder(x)

        # Transformer Encoder with KV Cache
        if use_cache:
            x = self.transformer_encoder(x, cache=self.kv_cache)
        else:
            x = self.transformer_encoder(x)
            self.kv_cache.clear()

        # Output shape - (batch_size, seq_len, num_joints)
        x = self.linear(x)

        # Return shape - (batch_size, num_joints) or (batch_size, seq_len, num_joints)
        return x.squeeze(1)
        
# Main Function
# Test the Model
def main():
    random_adjacency_matrix = np.random.randint(2, size=(7, 7))
    model = BoTMixWithKVCache(
        num_joints=7,
        d_model=64,
        nhead=4,
        num_layers=3,
        feedforward_dim=128,
        adjacency_matrix=torch.tensor(random_adjacency_matrix),
        dropout=0.1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Example: Processing a sequence incrementally with KV Cache
    batch_size = 32
    seq_length = 10  # Total sequence length
    logger.info("Processing sequence incrementally with KV Cache:")
    for t in range(seq_length):
        x_t = torch.randn(batch_size, 7).to(device)  # Input at time step t
        out = model(x_t, use_cache=True)
        logger.info(f"Time step {t}: Output Shape: {out.shape}")

    # Reset the cache when done
    model.kv_cache.clear()

    # Alternatively, process all at once without caching
    x = torch.randn(batch_size, 7).to(device)
    out = model(x, use_cache=False)
    logger.info("Processing entire input without KV Cache:")
    logger.info("Model Input Shape: " + str(x.shape))
    logger.info("Model Output Shape: " + str(out.shape))

    # Summary of the Model
    logger.info("=" * 20)
    logger.info("Model Summary: ")
    
    summary(model.to(device), (32,7))
    logger.info("=" * 20)
    
if __name__ == "__main__":
    main()
