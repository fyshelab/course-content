class EncoderBlock(nn.Module):
  def __init__(self, n_heads, embed_dim, hidden_dim):
    super().__init__()    
    self.attention = MultiHeadAttention(n_heads, embed_dim)
    self.norm1 = ResidualNorm(embed_dim)
    self.feedforward = Feedforward(embed_dim, hidden_dim)
    self.norm2 = ResidualNorm(embed_dim)

  def forward(self, src_tokens, src_mask):
    """
    Args:
      src_tokens (n_batch, n_tokens, embed_dim): the source sequence
      src_mask (n_batch, n_tokens): binary mask over the source
    Returns:
      (n_batch, n_tokens, embed_dim): the encoder state
    """
    self_attention = self.attention(src_tokens, src_tokens, src_tokens, src_mask)
    normed_attention = self.norm1(self_attention, src_tokens)
    ff_out = self.feedforward(normed_attention)
    out = self.norm2(ff_out, normed_attention)
    return out