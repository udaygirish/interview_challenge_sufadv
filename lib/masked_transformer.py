import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMultiheadAttention(nn.Module):
    """
    Masked Multihead Attention module.
    Args:
        embed_dim (int): The embedding dimension.
        num_heads (int): The number of attention heads.
        dropout (float, optional): Dropout probability on attention weights. Default is 0.0.
    Attributes:
        embed_dim (int): The embedding dimension.
        num_heads (int): The number of attention heads.
        dropout (float): Dropout probability on attention weights.
        head_dim (int): Dimension of each attention head.
        q_proj (nn.Linear): Linear layer to project queries.
        k_proj (nn.Linear): Linear layer to project keys.
        v_proj (nn.Linear): Linear layer to project values.
        out_proj (nn.Linear): Linear layer to project the output.
    Methods:
        forward(query, key, value, attn_mask=None):
            Computes the masked multihead attention.
            Args:
                query (torch.Tensor): Query tensor of shape (batch_size, seq_len, embed_dim).
                key (torch.Tensor): Key tensor of shape (batch_size, seq_len, embed_dim).
                value (torch.Tensor): Value tensor of shape (batch_size, seq_len, embed_dim).
                attn_mask (torch.Tensor, optional): Attention mask tensor. Default is None.
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()
        
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_output_weights = attn_output_weights.masked_fill(attn_mask == 0, float('-inf'))
        
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)
        
        attn_output = torch.matmul(attn_output_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(attn_output)

# The `MaskedTransformerEncoderLayer` class represents a single layer of a masked transformer encoder
# with multi-head attention and feedforward network components.
class MaskedTransformerEncoderLayer(nn.Module):
    """
    A single layer of a masked transformer encoder.
    Args:
        d_model (int): The number of expected features in the input (required).
        nhead (int): The number of heads in the multiheadattention models (required).
        dim_feedforward (int, optional): The dimension of the feedforward network model (default=2048).
        dropout (float, optional): The dropout value (default=0.1).
    Attributes:
        self_attn (MaskedMultiheadAttention): Multi-head attention mechanism with masking.
        linear1 (nn.Linear): First linear transformation in the feedforward network.
        dropout (nn.Dropout): Dropout layer for regularization.
        linear2 (nn.Linear): Second linear transformation in the feedforward network.
        norm1 (nn.LayerNorm): Layer normalization applied after the self-attention mechanism.
        norm2 (nn.LayerNorm): Layer normalization applied after the feedforward network.
        dropout1 (nn.Dropout): Dropout layer applied after the self-attention mechanism.
        dropout2 (nn.Dropout): Dropout layer applied after the feedforward network.
    Methods:
        forward(src, src_mask=None):
            Passes the input through the masked transformer encoder layer.
            Args:
                src (Tensor): The input tensor.
                src_mask (Tensor, optional): The mask tensor for the input (default=None).
            Returns:
                Tensor: The output tensor after processing through the encoder layer.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MaskedMultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class MaskedTransformerEncoder(nn.Module):
    """
    MaskedTransformerEncoder is a custom encoder module for a Transformer model.
    Args:
        encoder_layer (nn.Module): A single layer of the transformer encoder.
        num_layers (int): The number of encoder layers to stack.
    Attributes:
        layers (nn.ModuleList): A list containing the stacked encoder layers.
    Methods:
        forward(src, mask=None):
            Passes the input through each encoder layer in sequence.
            Args:
                src (Tensor): The input tensor to the encoder.
                mask (Tensor, optional): An optional mask tensor to apply to the input.
            Returns:
                Tensor: The output tensor after passing through all encoder layers.
    """
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        
    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, src_mask=mask)
        return src