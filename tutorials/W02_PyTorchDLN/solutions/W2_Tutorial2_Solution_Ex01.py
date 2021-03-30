input_dim = 1
output_dim = 1
hidden_1 = 10
hidden_2 = 10

def model(input_dim, hidden_1, hidden_2, output_dim):
  net = nn.Sequential(nn.Linear(input_dim, hidden_1),
                      nn.Linear(hidden_1, hidden_2),
                      nn.Linear(hidden_2, output_dim))
  return (net)


my_net = model(input_dim, hidden_1, hidden_2, output_dim)
print(my_net)
