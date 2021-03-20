class NeuralNet(nn.Module):
  def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings):
    super(NeuralNet, self).__init__()

    self.batch_size = batch_size
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.embedding_length = embedding_length

    self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
    self.word_embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
    self.fc1 = nn.Linear(embedding_length, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)

  def forward(self, inputs):
    
    input = self.word_embeddings(inputs) #convert text to embeddings
    pooled = F.avg_pool2d(input, (input.shape[1], 1)).squeeze(1)

    x = self.fc1(pooled)
    x = F.relu(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)    
    return output