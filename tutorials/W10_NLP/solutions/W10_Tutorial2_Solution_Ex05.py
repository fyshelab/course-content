class DecoderBlock(nn.Module):
  def __init__(self, n_heads, embed_dim, hidden_dim):
    super().__init__()    
    self.self_attention = MultiHeadAttention(n_heads, embed_dim)
    self.norm1 = ResidualNorm(embed_dim)
    self.encoder_attention = MultiHeadAttention(n_heads, embed_dim)
    self.norm2 = ResidualNorm(embed_dim)
    self.feedforward = Feedforward(embed_dim, hidden_dim)
    self.norm3 = ResidualNorm(embed_dim)

  def forward(self, tgt_tokens, tgt_mask, encoder_state, src_mask):
    """
    Args:
      tgt_tokens (n_batch, n_tokens, embed_dim): the target sequence
      tgt_mask (n_batch, n_tokens): binary mask over the target tokens
      encoder_state (n_batch, n_tokens, embed_dim): the output of the encoder pass
      src_mask (n_batch, n_tokens): binary mask over the source tokens
    Returns:
      (n_batch, n_tokens, embed_dim): the decoder state
    """
    self_attention = self.self_attention(tgt_tokens, tgt_tokens, tgt_tokens, tgt_mask)
    normed_self_attention = self.norm1(self_attention, tgt_tokens)
    encoder_attention = self.encoder_attention(normed_self_attention, encoder_state, encoder_state, src_mask)
    normed_encoder_attention = self.norm2(encoder_attention, normed_self_attention)
    ff_out = self.feedforward(normed_encoder_attention)
    out = self.norm2(ff_out, normed_encoder_attention)
    return out