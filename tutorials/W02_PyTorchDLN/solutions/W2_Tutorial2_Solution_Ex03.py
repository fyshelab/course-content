def deepNetwork(deep=False):
  # Network parameters
  input_dim = 1
  output_dim = 1

  if deep:
    h1, h2, h3, h4, h5, h6 = 20, 15, 10, 5, 4, 2
    # define our network
    net = nn.Sequential(nn.Linear(input_dim, h1),
                        nn.Linear(h1, h2),
                        nn.Linear(h2, h3),
                        nn.Linear(h3, h4),
                        nn.Linear(h4, h5),
                        nn.Linear(h5, h6),
                        nn.Linear(h6, output_dim))
  else:
    # define our network
    net = nn.Sequential(nn.Linear(input_dim, 1))

  # parameters initialization
  for i in range(len(net)):
    n_in = net[i].weight.shape[0]
    n_out = net[i].weight.shape[1]
    sigma = np.sqrt(6 / (n_in + n_out))
    net[i].weight.data.uniform_(-sigma, sigma)
    net[i].bias.data.uniform_(-sigma, sigma)

  return (net)


net = deepNetwork(deep=True)
outputXav = training_loop(X, y, model=net)
plotRegression(X, y, outputXav[0], outputXav[1])
