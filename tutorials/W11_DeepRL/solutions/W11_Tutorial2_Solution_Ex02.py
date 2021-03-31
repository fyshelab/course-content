class ReplayBuffer:
  def __init__(self, state_dim, act_dim, buffer_size):
    self.buffer_size = buffer_size
    self.ptr = 0
    self.n_samples = 0

    self.state = torch.zeros(buffer_size, *state_dim, dtype=torch.float32, device=device)
    self.action = torch.zeros(buffer_size, act_dim, dtype=torch.int64, device=device)
    self.reward = torch.zeros(buffer_size, 1, dtype=torch.float32, device=device)
    self.discount = torch.zeros(buffer_size, 1, dtype=torch.float32, device=device)
    self.next_state = torch.zeros(buffer_size, *state_dim, dtype=torch.float32, device=device)

  def add(self, state, action, reward, discount, next_state):
    self.state[self.ptr] = state
    self.action[self.ptr] = action
    self.reward[self.ptr] = reward
    self.discount[self.ptr] = discount
    self.next_state[self.ptr] = next_state
    
    if self.n_samples < self.buffer_size:
      self.n_samples += 1

    self.ptr = (self.ptr + 1) % self.buffer_size

  def sample(self, batch_size):      
    # Select batch_size number of sample indicies at random from the buffer
    idx = np.random.choice(self.n_samples, batch_size)    
    # Using the random indices, assign the corresponding state, action, reward,
    # discount, and next state samples.
    state = self.state[idx]
    action = self.action[idx]
    reward = self.reward[idx]
    discount = self.discount[idx]
    next_state = self.next_state[idx]
    
    return Batch(state, action, reward, discount, next_state)
