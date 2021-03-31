### Question
hat is the purpose of delaying the start of training while choosing random actions?

### Answer
We want to populate the replay buffer with a good distribution of initial outcomes before sampling batches to improve training stability.
