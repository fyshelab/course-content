class DoubleDQNAgent(DQNAgent):  
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
      # Compute the action values from our q_net using a batch.next_state
      q_next_actions = self.q_net(batch.next_state)
      # Use this to find the actions that correspond to the largest values
      # (i.e. argmax)
      max_acts = q_next_actions.argmax(dim=1).view(-1,1)
      # Next get the action values using our target_net and batch.next_state
      q_target_actions = self.target_net(batch.next_state)
      # Then we select the q-values that correspond to the actions we just found
      # (hint: use the gather method)
      q_target = q_target_actions.gather(1, max_acts)
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
