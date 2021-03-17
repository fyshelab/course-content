class LSTM(nn.Module):
  def __init__(self, layers, batch_size, output_size, hidden_size, vocab_size, embedding_length):
    super(LSTM, self).__init__()

    self.batch_size = batch_size
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.embedding_length = embedding_length
    self.n_layers = layers

    self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
    self.dropout = nn.Dropout(0.5)
    self.rnn = nn.LSTM(embedding_length, hidden_size, num_layers=self.n_layers)
    self.label = nn.Linear(self.n_layers*self.hidden_size, output_size)

  def forward(self, input_sentences, batch_size=None):
    input = self.word_embeddings(input_sentences).permute(1, 0, 2)
    hidden = (torch.randn(self.n_layers, input.shape[1], self.hidden_size).to(device),
            torch.randn(self.n_layers, input.shape[1], self.hidden_size).to(device))
    input = self.dropout(input)
    output, hidden = self.rnn(input, hidden)
    h_n = hidden[0].permute(1, 0, 2)
    h_n = h_n.contiguous().view(h_n.shape[0], -1)
    logits = self.label(h_n)
    return logits
