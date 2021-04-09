class BCAgent(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(BCAgent, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, state):
        x = F.relu(self.linear1(state))
        # We don't use softmax here because CrossEntropyLoss does that already.
        x = self.linear2(x)
        return x 
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = F.softmax(self.forward(state), dim=1)
        action = torch.argmax(probs, dim=1)
        return action.numpy()[0]
    
    def update(self, state, action):
        # Get output from model
        output = self.forward(state)
        # Compute cross-entropy loss
        loss = self.criterion(output, action.squeeze())
        # Take gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


def behavioral_cloning(env, agent, buffer, num_epochs=10, iters_per_epoch=200, batch_size=50):
    epoch_losses = []
    epoch_rewards = []
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for i in range(iters_per_epoch):
            # Sample a batch of states and actions from the buffer
            state, action = buffer.sample(batch_size)
            # Update agent
            loss = agent.update(state, action)
            total_loss += loss
        # Log average loss
        epoch_losses.append(total_loss / iters_per_epoch)
        # Evaluate in environment
        total_reward = 0
        done = False
        state = env.reset()
        while not done:
            with torch.no_grad():
                action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
        epoch_rewards.append(total_reward)
        print(f'Epoch [{epoch}/{num_epochs}], loss: {epoch_losses[-1]}, reward: {epoch_rewards[-1]}')
    return epoch_losses, epoch_rewards
