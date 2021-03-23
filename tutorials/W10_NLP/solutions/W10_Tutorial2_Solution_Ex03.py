class PositionalEncoder(nn.Module):
  def __init__(self, embed_dim, max_len=1000):
    super().__init__()
    self.position_embedding = torch.zeros((1, max_len, embed_dim))
    i = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
    j2 = torch.arange(0, embed_dim, step=2, dtype=torch.float32)
    x = i / torch.pow(10000, j2 / embed_dim)
    self.position_embedding[..., 0::2] = torch.sin(x)
    self.position_embedding[..., 1::2] = torch.cos(x)        

  def forward(self, x):
    x_plus_p = x + self.position_embedding[:, :x.shape[1]]
    return x_plus_p
  
with plt.xkcd():
  n_tokens, embed_dim = 40, 40
  pos_enc = PositionalEncoder(embed_dim) 
  p = pos_enc(torch.zeros((1, n_tokens, embed_dim)))
  plt.imshow(p.squeeze())