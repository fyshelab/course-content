def MSEvsMAE(inputs, targets):
  # input size dimension - features
  input_dim = inputs.shape[1]
  output_dim = targets.shape[1]
  # size of the hidden layer
  hidden = [50, 10]

  learningRate = 1e-5
  epochs = 2000

  # Create the model with MAE Loss
  modelMAE = SimpleNet(input_dim, hidden, output_dim).to(device)
  criterion_mae = nn.L1Loss()  # L1 Loss - Absolute error
  optimizerMAE = torch.optim.SGD(modelMAE.parameters(), lr=learningRate)

  # Create model with MSE loss
  modelMSE = SimpleNet(input_dim, hidden, output_dim).to(device)
  criterion_mse = nn.MSELoss()
  optimizerMSE = torch.optim.SGD(modelMSE.parameters(), lr=learningRate)


  # Training Loop for both models
  print('Training...')
  lossesMAE = []
  lossesMSE = []

  loss_testMAE = []
  loss_testMSE = []

  epoch_range = trange(epochs, desc='MAE: vs MSE:', leave=True)
  for epoch in epoch_range:
    if lossesMAE:
      epoch_range.set_description(
          "MAE: {:.4f} vs MSE: {:.4f}".format(lossesMAE[-1], lossesMSE[-1]))
      epoch_range.refresh() # to show immediately the update
    time.sleep(0.01)

    # Clear gradient buffers because we don't want any gradient from
    # previous epoch to carry forward, dont want to cummulate gradients
    optimizerMAE.zero_grad()
    optimizerMSE.zero_grad()
    # get output from the model, given the inputs
    outputsMAE = modelMAE(inputs)
    outputsMSE = modelMSE(inputs)
    # get loss for the predicted output
    lossMAE = criterion_mae(outputsMAE, targets)
    lossMSE = criterion_mse(outputsMSE, targets)
    # get gradients w.r.t to parameters
    lossMAE.backward()
    lossMSE.backward()

    # update parameters
    optimizerMAE.step()
    optimizerMSE.step()
    lossesMAE.append(lossMAE.item())
    lossesMSE.append(lossMSE.item())
      
    loss_testMAE.append(criterion_mae(modelMAE(inputs_test), targets_test))
    loss_testMSE.append(criterion_mse(modelMSE(inputs_test), targets_test))
      
  return (modelMAE, modelMSE,
          criterion_mae, criterion_mse,
          lossesMAE, lossesMSE,
          loss_testMAE, loss_testMSE)


output = MSEvsMAE(inputs_new, targets)

MAE_test = output[2](output[0](inputs_test), targets_test)
loss_2 = output[3](output[1](inputs_test), targets_test)
RMSE_test = torch.sqrt(loss_2) # we take the square root of MSE to have 
                               # both errors in the same scale

loss_comparison(output[4], output[5],
                output[6], output[7],
                RMSE_test, MAE_test)
