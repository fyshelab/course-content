# to_remove solution
class DQNAgent:
  def __init__(self, policy, q_net, target_net, optimizer, tau, replay_buffer,
               batch_size, train_start):
    self.policy = policy
    self.q_net = q_net
    self.target_net = target_net
    # we never need to compute gradients on the target network, so we disable
    # autograd to speed up performance
    for p in self.target_net.parameters():
      p.requires_grad = False
    self.optimizer = optimizer
    self.tau = tau
    self.replay_buffer = replay_buffer
    self.batch_size = batch_size
    self.train_start = train_start
    self.is_waiting = True
    
  def act(self, state):
    # we never need to compute gradients on action selection, so we disable
    # autograd to speed up performance
    with torch.no_grad():
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

    # Using the Replay Buffer you made in exercise 2, sample a batch of steps
    # for training
    batch = self.replay_buffer.sample(self.batch_size)
    
    # First let's compute our predicted q-values
    # We need to pass our batch of states (batch.state) to our q_net    
    q_actions = self.q_net(batch.state)
    # Then we select the q-values that correspond to the actions in our batch
    # (batch.action) to get our predictions (hint: use the gather method)
    q_pred = q_actions.gather(1, batch.action)
    
    # Now compute the q-value target (also known as the td target or bellman
    # backup) using our target network. Since we don't need gradients for this,
    # we disable autograd here to speed up performance    
    with torch.no_grad():
      # First get the q-values from our target_net using the batch of next
      # states.
      q_target_actions = self.target_net(batch.next_state)
      # Get the values that correspond to the best action by taking the max along
      # the value dimension (dim=1)
      q_target = q_target_actions.max(dim=1)[0].view(-1, 1)
      # Next multiply by batch.discount and add batch.reward
      q_target = batch.reward + batch.discount * q_target
    # Compute the MSE loss between the predicted and target values, then average
    # over the batch
    loss = F.mse_loss(q_pred, q_target).mean()    

    # backpropogation to update the q network
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # soft update target network with the updated q-network
    soft_update_from_to(self.q_net, self.target_net, self.tau)
