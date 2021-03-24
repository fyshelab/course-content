class MultiHeadAttention(nn.Module):
  def __init__(self, n_heads, embed_dim):
    super().__init__()
    self.n_heads = n_heads
    self.head_dim = embed_dim // n_heads
    
    self.attention = ScaledDotProductAttention(embed_dim)
    self.query_fc = nn.Linear(embed_dim, embed_dim, bias=False)
    self.key_fc = nn.Linear(embed_dim, embed_dim, bias=False)
    self.value_fc = nn.Linear(embed_dim, embed_dim, bias=False)
    self.out_fc = nn.Linear(embed_dim, embed_dim, bias=False)
  
  def forward(self, queries, keys, values, mask):
    """
    Args:
      queries (n_batch, n_tokens, embed_dim): queries (Q) tensor
      keys (n_batch, n_tokens, embed_dim): keys (K) tensor
      values (n_batch, n_tokens, embed_dim): values (V) tensor
      mask (n_batch, n_tokens): binary mask tensor
    Returns:
      (n_batch, n_tokens, embed_dim): multi-head attention tensor
    """
    q_heads = mha_transform_input(self.query_fc(queries), self.n_heads, self.head_dim)
    k_heads = mha_transform_input(self.key_fc(keys), self.n_heads, self.head_dim)
    v_heads = mha_transform_input(self.value_fc(values), self.n_heads, self.head_dim)

    attention_heads = self.attention(q_heads, k_heads, v_heads, mask)
    attention = self.out_fc(mha_transform_output(attention_heads, self.n_heads, self.head_dim))
    return attention