def linear_regression(X, w, b):
  '''
  Linear regression model.
  
  Parameters
  ----------
  X : torch.tensor
      design matrix.
  w : torch.tensor
      weights.
  b : torch.tensor
      bias.
  
  Returns
  -------
  torch.tensor
    predicted values.
  
  '''
  reg = torch.matmul(w.T, X) + b
  return reg
