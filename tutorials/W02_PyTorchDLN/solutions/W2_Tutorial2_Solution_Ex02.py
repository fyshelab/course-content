class Net(nn.Module):

  def __init__(self, input_dim, hidden_1, hidden_2, output_dim):
    super(Net, self).__init__()

    self.input_dim = input_dim
    self.hidden_1 = hidden_1
    self.hidden_2 = hidden_2
    self.output_dim = output_dim

    # A fully-connected network (FCN) with 2 hidden layers
    self.fc1 = nn.Linear(self.input_dim, self.hidden_1)
    self.fc2 = nn.Linear(self.hidden_1, self.hidden_2)
    self.fc3 = nn.Linear(self.hidden_2, self.output_dim)

  def forward(self, x):
    h1 = self.fc1(x)
    h2 = self.fc2(h1)
    out = self.fc3(h2)

    return out


my_net2 = Net(input_dim, hidden_1, hidden_2, output_dim)
print(my_net2)
