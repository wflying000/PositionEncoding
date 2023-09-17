import math
import torch
import torch.nn as nn

class Attention(nn.Module):
    """
        Relative Position Encoding implementation of HuggingFace Bert
    """
    def __init__(self, hidden_size, num_heads, max_length, position_embedding_type, dropout=0.1):
        super(Attention, self).__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.max_length = max_length
        self.position_embedding_type = position_embedding_type
        self.distance_embedding = nn.Embedding(2 * max_length - 1, self.head_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        bsz, query_len, hidden_size = query.size()
        key_len = key.size(1)
        device = query.device

        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        query = query.view(bsz, query_len, self.num_heads, self.head_size).transpose(1, 2)
        key = key.view(bsz, key_len, self.num_heads, self.head_size).transpose(1, 2)
        value = value.view(bsz, key_len, self.num_heads, self.head_size).transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(-2, -1))

        position_ids_l = torch.arange(query_len, dtype=torch.long, device=device).view(-1, 1)
        position_ids_r = torch.arange(key_len, dtype=torch.long, device=device).view(1, -1)
        distance = position_ids_l - position_ids_r
        position_embedding = self.distance_embedding(distance + self.max_length - 1)

        if self.position_embedding_type == "relative_key":
            relative_position_scores = torch.einsum("bhld,lrd->bhlr", query, position_embedding)
            attn_weights = attn_weights + relative_position_scores
        else:
            relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query, position_embedding)
            relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key, position_embedding)
            attn_weights = attn_weights + relative_position_scores_query + relative_position_scores_key
        
        attn_weights = attn_weights / math.sqrt(self.head_size)

        if mask is not None:
            attn_weights += mask
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value) # (bsz, heads, tlen, slen) * (bsz, heads, slen, hs) -> (bsz, heads, tlen, hs)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, query_len, hidden_size)
        attn_output = self.out_proj(attn_output)

        return attn_output


class RelativePositionEncoding(nn.Module):
    def __init__(self, max_relative_position, embed_dim):
        super(RelativePositionEncoding, self).__init__()
        self.max_relative_position = max_relative_position
        self.embed_dim = embed_dim
        self.embeddings = nn.Embedding(2 * max_relative_position + 1, embed_dim)
    
    def forward(self, query_length, key_length):
        device = self.embeddings.weight.device
        query_position_ids = torch.arange(query_length, dtype=torch.long, device=device).view(-1, 1)
        key_position_ids = torch.arange(key_length, dtype=torch.long, device=device).view(1, -1)
        relative_position_ids = query_position_ids - key_position_ids
        relative_position_ids = torch.clamp(relative_position_ids, -self.max_relative_position, self.max_relative_position)
        relative_position_ids = relative_position_ids + self.max_relative_position
        relative_position_embedding = self.embeddings(relative_position_ids)
        return relative_position_embedding

class AttentionWithVillinaRelativePositionEncoding(nn.Module):
    """
        relative position encoding according to origin paper Self-Attention with Relative Position Representations
    """
    def __init__(self, hidden_size, num_heads, max_relative_position, position_embedding_type, dropout=0.1):
        super(AttentionWithVillinaRelativePositionEncoding, self).__init__()
        assert hidden_size % num_heads == 0
        assert position_embedding_type in ("relative_key", "relative_value", "relative_key_value")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.max_relative_position = max_relative_position
        self.position_embedding_type = position_embedding_type
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        if position_embedding_type == "relative_key":
            self.relative_embeddings_key = RelativePositionEncoding(max_relative_position, self.head_size)
        elif position_embedding_type == "relative_value":
            self.relative_embeddings_value = RelativePositionEncoding(max_relative_position, self.head_size)
        else:
            self.relative_embeddings_key = RelativePositionEncoding(max_relative_position, self.head_size)
            self.relative_embeddings_value = RelativePositionEncoding(max_relative_position, self.head_size)

    def forward(self, query, key, value, mask=None):
        bsz, query_len, hidden_size = query.size()
        key_len = key.size(1)
        device = query.device

        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        q1 = query.view(bsz, query_len, self.num_heads, self.head_size).transpose(1, 2)
        k1 = key.view(bsz, key_len, self.num_heads, self.head_size).transpose(1, 2)
        value = value.view(bsz, key_len, self.num_heads, self.head_size).transpose(1, 2)

        attn_weights = torch.matmul(q1, k1.transpose(-2, -1)) # (bsz, nhead, qlen, klen)

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_value":
            relative_embedding_key = self.relative_embeddings_key(query_len, key_len) # (qlen, klen, head_size)
            q2 = query.view(bsz, query_len, self.num_heads, self.head_size).transpose(0, 1)
            q2 = q2.contiguous().view(query_len, bsz * self.num_heads, self.head_size)
            relative_attn_weights = torch.matmul(q2, relative_embedding_key.transpose(-2, -1)) # (qlen, b*nhead, klen)
            relative_attn_weights = relative_attn_weights.view(query_len, bsz, self.num_heads, key_len)
            relative_attn_weights = relative_attn_weights.contiguous().permute(1, 2, 0, 3)
            attn_weights = attn_weights + relative_attn_weights
        
        attn_weights = attn_weights / math.sqrt(self.head_size)
        
        if mask is not None:
            attn_weights += mask
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights) # (bsz, nhead, qlen, klen)

        attn_outputs = torch.matmul(attn_weights, value) # (bsz, nhead, qlen, head_size)

        if self.position_embedding_type == "relative_value" or self.position_embedding_type == "relative_key_value":
            relative_embedding_value = self.relative_embeddings_value(query_len, key_len) # (qlen, klen, head_size)
            attn_weights_value = attn_weights.permute(2, 0, 1, 3)
            attn_weights_value = attn_weights_value.contiguous().view(query_len, bsz * self.num_heads, key_len)
            attn_outputs_value = torch.matmul(attn_weights_value, relative_embedding_value) # (qlen, b*nhead, head_size)
            attn_outputs_value = attn_outputs_value.view(query_len, bsz, self.num_heads, self.head_size)
            attn_outputs_value = attn_outputs_value.contiguous().permute(1, 2, 0, 3)
            attn_outputs = attn_outputs + attn_outputs_value
        
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(bsz, query_len, hidden_size)
        attn_outputs = self.out_proj(attn_outputs)

        return attn_outputs


def test():
    hidden_size = 768
    num_heads = 12
    max_length = 512
    position_embedding_type = "relative_key_query"

    bsz = 3
    query_len = 17
    key_len = 23
    query = torch.rand(bsz, query_len, hidden_size)
    key = torch.rand(bsz, key_len, hidden_size)
    value = key

    attention = Attention(hidden_size, num_heads, max_length, position_embedding_type)
    attention2 = AttentionWithVillinaRelativePositionEncoding(
        hidden_size=hidden_size,
        num_heads=num_heads,
        max_relative_position=5,
        position_embedding_type="relative_key_value",
    )

    res = attention(query, key, value)
    res2 = attention2(query, key, value)
    print(res2.shape)

    

if __name__ == "__main__":
    test()







        