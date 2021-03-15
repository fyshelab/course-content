### Question
The generator BCE loss is diverging, but images are still improving! Why do you think BCE loss fails to measure the actual performance of GANs?

### Answer
Because there are many Nash equilibria, there is no optimal convergent point. So, as a result, the training losses often oscillate, and looking at only BCE loss trajectory is not enough to measure the model performance.
