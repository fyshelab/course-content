def training_modes(inputs, targets):

  input_dim = inputs.shape[1]
  output_dim = targets.shape[1]
  hidden_sizes = 20

  learning_rate  = 2e-3  # learning rate
  num_epochs = 3000

  # Loss function
  criterion = nn.MSELoss()

  model = networkSVD(input_dim, output_dim, hidden_sizes)
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  # Train for num_epochs
  losses = []
  rank = 5
  modes = np.empty((rank, num_epochs))

  # this few lines would implement a progress bar and loss description
  epoch_range = trange(num_epochs, desc='loss: ', leave=True)
  for epoch in epoch_range:
    if losses:
      epoch_range.set_description("loss: {:.6f}".format(losses[-1]))
      epoch_range.refresh() # to show immediately the update
    time.sleep(0.01)

    preds = model(inputs)

    loss = criterion(preds, targets)
    loss.backward()

    # SVD applied on the matrix product
    w_mult = (model[0].weight.T @ model[1].weight.T).detach().numpy()
    w_svd = np.linalg.svd(w_mult, compute_uv=False, full_matrices=True)
    modes[:,epoch] = w_svd[:rank]

    # Grdient descent
    optimizer.step()
    optimizer.zero_grad()

    losses.append(loss.item())

  return (losses, modes, num_epochs)


output = training_modes(inputs, targets)
plot_learning_modes(output[0], output[2], output[1], rank=5)
