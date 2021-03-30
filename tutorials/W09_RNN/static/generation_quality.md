### Question
Is there an improvement in the quality of the generated text? Can you infer what the original text used for training could be? Why do you think generation is a difficult task?

### Answer
Yes, the multinomial distribution over the model output does a better job at sampling over characters than simply choosing the best one at each step.   
The original text was Julius Caesar, Shakespeare.  
Generation is a difficult task because our model chooses the next character prediction out of a large number of possibilities, i.e., the probability is distributed over all the characters in the vocabulary. A significant amount of tuning is required for such models to produce flawless text.       
