### Question
AlphaZero is in some sense a policy iteration algorithm. How is the policy evaluated, and how is it improved?

### Answer
The policy is evaluated by MCTS, which computes the Q-values of states and actions using the current policy. Policy improvement is done by gradient descent.
