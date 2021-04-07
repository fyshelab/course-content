### Question
What happens if we set c_puct too high? What about too low?

### Answer
If c_puct is too high, then the agent will only explore according to the policy and visit counts, but it will not learn to playing good moves. If c_puct is too low, then the agent will only play actions with high values, but without exploration it may not see the real good moves.
