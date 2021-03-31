class DQNAgent:
  def __init__(self, policy, q_net, target_net, optimizer, tau, replay_buffer,
               batch_size, train_start):
    self.policy = policy
    self.q_net = q_net
    self.target_net = target_net
    self.optimizer = optimizer
    self.tau = tau
    self.replay_buffer = replay_buffer
    self.batch_size = batch_size
    self.train_start = train_start
    self.is_waiting = True
    
  def act(self, state):
    if self.is_waiting:
      return torch.randint(6, (1,1))
    return self.policy(self.q_net, state)
  
  def train(self, state, action, reward, discount, next_state, frame):
    # Add the step to our replay buffer
    replay_buffer.add(state, action, reward, discount, next_state)  
    # Don't train if we aren't ready
    if frame < self.train_start:
      return
    elif frame == self.train_start:
      self.is_waiting = False

    # Sample a batch of steps for training
    batch = self.replay_buffer.sample(self.batch_size)
    # Compute our predicted q-value given the state and action from our batch
    q_pred = self.q_net(batch.state).gather(1, batch.action)
    # Now compute the bellman backup using the target network
    # First get the q-values from the next state and best action
    q_target = self.target_net(batch.next_state).max(dim=1)[0].view(-1, 1)
    # Next multiply by the discount and add the reward signal
    q_target = batch.reward + batch.discount * q_target.detach()
    # Compute the MSE loss between the predicted and target values, then average
    # over the batch
    loss = F.mse_loss(q_pred, q_target).mean()    

    # backpropogation to update the q network
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # soft update target network with the updated q-network
    soft_update_from_to(self.q_net, self.target_net, self.tau)
