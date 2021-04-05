class DynamicsModel(nn.Module):
    def __init__(self, act_dim, state_dim, hidden_size, learning_rate=0.001):
        super().__init__()
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        
        # 2-layer fully connected network
        self.layers = nn.Sequential(
            nn.Linear(act_dim + state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, state_dim)
        )

        # Use Adam optimizer and MSELoss()
        self.optimizer = optim.Adam(self.layers.parameters(), self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def forward(self, input):
        return self.layers(input)

    def update(self, state, action, next_state):
        # Update model
        # state: [n, state_dim]
        # action: [n, act_dim]
        # next_state: [n, state_dim]

        # Concatenate state and action along the last dimension
        input = torch.cat((state, action), -1)

        # Forward input through model to get the output
        output = self.forward(input)

        # Constuct target as the state difference 
        target = next_state - state

        # Compute model loss using self.criterion
        loss = self.criterion(output, target)

        # Take one gradient step using self.optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def predict(self, state, action):
        # Concatenate state and action along the last dimension
        input = torch.cat((state, action), -1)

        with torch.no_grad():
            # Forward input through model to get the state difference
            diff = self.forward(input)

        # Add difference to input state to reconstruct the next state.
        next_state = state + diff
        return next_state
