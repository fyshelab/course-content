class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=0)
        return x 

    def get_action(self, state):
        state = torch.from_numpy(state).to(device).float()
        probs = self.forward(state)

        # Sample an action from the policy. Hint: use np.random.choice()
        action = np.random.choice(self.num_actions, p=probs.cpu().detach().numpy())

        # Compute log probabilty for this action
        log_prob = torch.log(probs[action])
        return action, log_prob


def update_policy(policy_network, rewards, log_probs, gamma):
    discounted_rewards = []
    for t in range(len(rewards)):
        # At each step, we compute the sum of discounted future rewards
        Gt = 0  # Gt is the sum of discounted future rewards
        pow = 0 # pow keeps track of the power of discount factor gamma
        for r in rewards[t:]:
            # Compute discounted reward at the current time step
            # Note: rewards is a list consisting of reward in each step of episode
            Gt = Gt + gamma**pow * r
            pow = pow + 1
        discounted_rewards.append(Gt)
    discounted_rewards = torch.tensor(discounted_rewards)

    policy_gradients = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        # Multiply log_prob with Gt and append to list of gradients.
        # How can we have the optimizer perform gradient ASCENT?
        policy_gradients.append(-log_prob * Gt)
    
    # Perform one step of gradient update
    policy_network.optimizer.zero_grad()
    objective = torch.stack(policy_gradients).sum()
    objective.backward()
    policy_network.optimizer.step()
