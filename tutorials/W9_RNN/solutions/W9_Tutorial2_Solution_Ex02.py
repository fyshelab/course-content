class biRNN(nn.Module):
  def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length):
    super(biRNN, self).__init__()

    self.batch_size = batch_size
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.embedding_length = embedding_length

    self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
    self.dropout = nn.Dropout(0.5)
    self.rnn = nn.RNN(embedding_length, hidden_size, num_layers=2, bidirectional=True)
    self.fc = nn.Linear(4*hidden_size, output_size)

  def forward(self, input_sentences, batch_size=None):
    input = self.word_embeddings(input_sentences)
    input = input.permute(1, 0, 2)
    h_0 =  Variable(torch.zeros(4, input.size()[1], self.hidden_size).to(device))
    input = self.dropout(input)
    output, h_n = self.rnn(input, h_0)
    h_n = h_n.permute(1, 0, 2) 
    h_n = h_n.contiguous().view(h_n.size()[0], h_n.size()[1]*h_n.size()[2])
    logits = self.fc(h_n)
    
    return logits
