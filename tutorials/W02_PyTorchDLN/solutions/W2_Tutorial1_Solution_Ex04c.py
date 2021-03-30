def analytical_sol(X, y):
  '''
  Squared error loss function.
  
  Parameters
  ----------
  X : numpy.ndarray (float)
      design matrix with dimensions NxD, D: features, N: examples.
  y : numpy.ndarray (float)
      target values.
  
  Returns
  -------
  params: numpy.ndarray
          the parameters w, and bias (last element)

          if `X.T @ X` is singular, returns nothing.
  '''

  # Check if the inverse exists, unless print error.
  params = []
  if np.linalg.det(X.T @ X) != 0:
    params = np.linalg.inv(X.T @ X) @ X.T @ y
  else:
    print('LinAlgError. Matrix is Singular. No analytical solution.')

  return (params)
