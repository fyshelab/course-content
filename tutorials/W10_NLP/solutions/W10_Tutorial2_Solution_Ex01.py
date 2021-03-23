class ScaledDotProductAttention(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    self.embed_dim = embed_dim

  def forward(self, queries, keys, values, mask):
    """
    Args:
      queries (n_batch, n_tokens, embed_dim): queries (Q) tensor
      keys (n_batch, n_tokens, embed_dim): keys (K) tensor
      values (n_batch, n_tokens, embed_dim): values (V) tensor
      mask (n_batch, n_tokens): binary mask tensor
    Returns:
      (n_batch, n_tokens, embed_dim): scaled dot product attention tensor
    """
    scaled_dot_product = torch.bmm(queries, torch.transpose(keys, 1, 2)) / np.sqrt(self.embed_dim)
    masked_softmax_scores = masked_softmax(scaled_dot_product, mask)
    attention = torch.bmm(masked_softmax_scores, values)
    return attention

torch.manual_seed(522)
batch_size, n_tokens, embed_dim = 1, 3, 4
tokens = torch.normal(0, 1, (batch_size, n_tokens, embed_dim))
attention = ScaledDotProductAttention(embed_dim)
mask = torch.ones((batch_size, n_tokens))
print(attention(tokens, tokens, tokens, mask))
mask[0, 2:] = 0
print(attention(tokens, tokens, tokens, mask))