### Question
How did we define the reward function for AlphaZero? Is the reward issued at every time step? Do players receive the same reward?

### Answer
The reward function is only given at the end of a game. It is +1 for the winning side, -1 for the losing side, and 0 for both sides if the game ends in a draw.