def a2c(env, actor, critic, max_episodes, num_steps, gamma=0.99):
    all_lengths = []
    average_lengths = []
    all_rewards = []

    for episode in tqdm.tqdm(range(max_episodes), position=0, leave=True):
        log_probs = []
        values = []
        rewards = []

        state = env.reset()
        for steps in range(num_steps):
            # Get the value of the current state from the critic
            value = critic.get_value(state)

            # Get an action and its log probability from the actor
            action, log_prob = actor.get_action(state)

            # Execute action in environment 
            new_state, reward, done, _ = env.step(action)

            # Store value, log probability, and reward
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            # Update state
            state = new_state
            
            if done or steps == num_steps - 1:
                # Store value of the last state
                V_last = critic.get_value(new_state)
                all_rewards.append(np.sum(rewards))
                break
        
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)

        # Compute Q values. Qvals[t] = r_t + gamma * r_{t+1} + ... + gamma^{T-t} * v_T
        Qvals = torch.zeros_like(values)
        # At this point, V_last stores the value of the last state, which we bootstrap from.
        Qval = V_last
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + gamma * Qval
            Qvals[t] = Qval
        
        # Compute advantage for all steps
        advantage = Qvals - values

        # Update critic
        critic_loss = advantage.pow(2).mean()
        critic.optimizer.zero_grad()
        critic_loss.backward()
        critic.optimizer.step()
        
        # Update critic
        actor_loss = torch.mean(-log_probs * advantage.detach())
        actor.optimizer.zero_grad()
        actor_loss.backward()
        actor.optimizer.step()

    # Plot results
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
