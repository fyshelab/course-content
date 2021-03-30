# to_remove solution
class Transformer(nn.Module):
  def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, hidden_dim, n_heads, n_blocks):
    super().__init__()
    self.encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, n_heads, n_blocks)
    self.decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim, n_heads, n_blocks)
    self.out = nn.Linear(embed_dim, tgt_vocab_size)

  def forward(self, src_tokens, src_mask, tgt_tokens, tgt_mask):
    # Compute the encoder output state from the source tokens and mask
    encoder_state = self.encoder(src_tokens, src_mask)
    # Compute the decoder output state from the target tokens and mask as well
    # as the encoder state and source mask
    decoder_state = self.decoder(tgt_tokens, tgt_mask, encoder_state, src_mask)
    # Compute the vocab scores by passing the decoder state through the output
    # linear layer
    out = self.out(decoder_state)
    return out