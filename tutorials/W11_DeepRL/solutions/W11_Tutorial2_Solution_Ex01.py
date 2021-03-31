class QNetwork(nn.Module):
  def __init__(self, n_channels, n_actions):
    super().__init__()
    self.conv = nn.Conv2d(in_channels=n_channels, out_channels=16,
                          kernel_size=3, stride=1)
    self.fc1 = nn.Linear(in_features=1024, out_features=128)
    self.fc2 = nn.Linear(in_features=128, out_features=n_actions)

  def forward(self, x):
    # Pass the input through the convnet layer with ReLU activation
    x = F.relu(self.conv(x))
    # Flatten the result while preserving the batch dimension
    x = torch.flatten(x, start_dim=1)
    # Pass the result through the first linear layer with ReLU activation
    x = F.relu(self.fc1(x))
    # Finally pass the result through the second linear layer and return
    x = self.fc2(x)
    return x
