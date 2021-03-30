def compute_graph(params, target):
  '''
  Simple function with the forward pass

  args:
    params: list
      contains the inputs and the weight tensors
  returns:
    L: float
      loss given a target value
  '''
  a, w1, w2, w3, w4 = params[0],params[1], params[2], params[3], params[4]
  
  b = w1 * a 
  c = w2 * a
  d = w3 * b + w4 * c 

  # Compute the summed loss
  L = (target - d).sum()

  # Store weights in a dictionary
  weights = {}
  weights['w1'] = w1
  weights['w2'] = w2
  weights['w3'] = w3
  weights['w4'] = w4

  # Store values of the nodes in a dictionary
  values = {}
  values['a'] = a
  values['b'] = b
  values['c'] = c
  values['d'] = d

  return (L, values, weights)
