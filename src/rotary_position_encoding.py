import math
import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, base=10000, device=None):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.embedding_dim = embedding_dim
        self.base = base
        self._set_sin_cos_cache(max_position_embeddings, device)
    
    def _set_sin_cos_cache(self, seq_len, device):
        self.max_position_embeddings = seq_len
        dim = self.embedding_dim
        position_encoding = torch.FloatTensor(
            [[pos / math.pow(self.base, 2 * (i // 2) / dim) for i in range(dim)] for pos in range(seq_len)]
        ).to(device)
        sin_cache = torch.sin(position_encoding[:, 0::2])
        cos_cache = torch.cos(position_encoding[:, 1::2])
        self.register_buffer("sin_cache", sin_cache, persistent=False)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
    
    def forward(self, seq_len, device=None):
        if seq_len > self.max_position_embeddings:
            self._set_sin_cos_cache(seq_len, device)
        
        sin_pos = self.sin_cache[:seq_len, :]
        cos_pos = self.cos_cache[:seq_len, :]

        return sin_pos, cos_pos
    



class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, max_length, dropout=0.1) -> None:
        super(Attention, self).__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.max_length = max_length
        self.rotary_position_embedding = RotaryPositionEmbedding(max_length, self.head_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_state, mask=None):
        bsz, seq_len, hidden_size = hidden_state.size()

        query = self.query_proj(hidden_state)
        key = self.key_proj(hidden_state)
        value = self.value_proj(hidden_state)

        query = query.view(bsz, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        key = key.view(bsz, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        value = value.view(bsz, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        sin, cos = self.rotary_position_embedding(seq_len, hidden_state.device)
        sin = sin[None, None, :, :]
        cos = cos[None, None, :, :]

        query, key = self.apply_rotary_position_embedding(sin, cos, query, key)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        if mask is not None:
            attn_weights += mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_outputs = torch.matmul(attn_weights, value) # (bsz, num_heads, qlen, head_dim)
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(bsz, seq_len, hidden_size)
        attn_outputs = self.out_proj(attn_outputs)

        return attn_outputs


    @staticmethod
    def apply_rotary_position_embedding(sin, cos, query, key):
        seq_len, dim = sin.size()[-2:]
        
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape(1, 1, seq_len, dim * 2)
        
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape(1, 1, seq_len, dim * 2)
        
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query = torch.stack([-query[..., 1::2], query[..., ::2]], dim=-1).reshape_as(query)
        
        query = query * cos_pos + rotate_half_query * sin_pos
        
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key = torch.stack([-key[..., 1::2], key[..., ::2]], dim=-1).reshape_as(key)
        
        key = key * cos_pos + rotate_half_key * sin_pos

        return query, key


def test():
    hidden_size = 768
    num_heads = 12
    max_length = 512

    bsz = 3
    seq_len = 17
    hidden_state = torch.rand(bsz, seq_len, hidden_size)

    attention = Attention(hidden_size, num_heads, max_length)
    res = attention(hidden_state)

    print(res.shape)
    

if __name__ == "__main__":
    test()

    