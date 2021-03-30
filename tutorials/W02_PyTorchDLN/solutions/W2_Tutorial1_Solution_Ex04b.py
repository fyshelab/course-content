def squared_error(y_hat, y):
  '''
  Squared error loss function.
  
  Parameters
  ----------
  y_hat : torch.tensor
      predicted values.
  y : torch.tensor
      true values.
  
  Returns
  -------
  err: FLOAT
       the squared error (loss)
  
  '''
  err = (y_hat - y.reshape(y_hat.shape)) ** 2
  return err
