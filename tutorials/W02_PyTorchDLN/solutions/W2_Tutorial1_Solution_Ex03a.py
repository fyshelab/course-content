def get_params():
  '''
  A simple function that generates our parameters and inputs
  
  ---
  args: nothing

  returns:
    a: torch.Tensor
      inputs
    w1: torch.Tensor
      weights
    w2: torch.Tensor
      weights
    w3: torch.Tensor
      weights
    w4: torch.Tensor
      weights

  '''
  a = torch.randn((3,3), requires_grad=True)
  w1 = torch.randn((3,3), requires_grad=True)
  w2 = torch.randn((3,3), requires_grad=True)
  w3 = torch.randn((3,3), requires_grad=True)
  w4 = torch.randn((3,3), requires_grad=True)

  return (a, w1, w2, w3, w4)
