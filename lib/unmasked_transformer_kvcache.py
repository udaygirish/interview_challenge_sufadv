import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    """
    MultiheadAttention is a module for performing multi-head attention as described in the paper
    "Attention is All You Need" by Vaswani et al.
    Args:
        embed_dim (int): The dimension of the embedding.
        num_heads (int): The number of attention heads.
        dropout (float, optional): Dropout probability on attention weights. Default: 0.0.
    Attributes:
        embed_dim (int): The dimension of the embedding.
        num_heads (int): The number of attention heads.
        dropout (float): Dropout probability on attention weights.
        head_dim (int): The dimension of each attention head.
        q_proj (nn.Linear): Linear layer for projecting the query.
        k_proj (nn.Linear): Linear layer for projecting the key.
        v_proj (nn.Linear): Linear layer for projecting the value.
        out_proj (nn.Linear): Linear layer for projecting the output.
    Methods:
        forward(query, key, value, key_padding_mask=None, kv_cache=None):
            Computes the multi-head attention output.
            Args:
                query (Tensor): The query tensor of shape (batch_size, seq_len, embed_dim).
                key (Tensor): The key tensor of shape (batch_size, seq_len, embed_dim).
                value (Tensor): The value tensor of shape (batch_size, seq_len, embed_dim).
                key_padding_mask (Tensor, optional): A mask tensor of shape (batch_size, seq_len) indicating which elements should be ignored.
                kv_cache (dict, optional): A dictionary containing cached key and value tensors for faster computation.
            Returns:
                Tensor: The output tensor of shape (batch_size, seq_len, embed_dim).
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
        
    def forward(self, query, key, value, key_padding_mask=None, kv_cache=None):
        batch_size, seq_len, _ = query.size()
        
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if kv_cache is not None and 'k' in kv_cache and 'v' in kv_cache:
            k = kv_cache['k']
            v = kv_cache['v']
        else:
            k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            if kv_cache is not None:
                kv_cache['k'] = k
                kv_cache['v'] = v
        
        attn_output_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
        
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)
        
        attn_output = torch.matmul(attn_output_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(attn_output)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_key_padding_mask=None, kv_cache=None):
        src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask, kv_cache=kv_cache)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        
    def forward(self, src, src_key_padding_mask=None, kv_cache=None):
        if kv_cache is None:
            kv_cache = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            src = layer(src, src_key_padding_mask=src_key_padding_mask, kv_cache=kv_cache[i])
        return src