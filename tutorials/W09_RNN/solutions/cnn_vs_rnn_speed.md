### Question
Based on exercises above would you say CNNs are faster than RNNs? Why?

### Answer
Yes, for this exercise. For a general case the answer depends on your model. CNN and RNN are different architectures, used differently, usually for different purposes. You can't really replace one by another without changing other elements of the model to compare the performance.

However, CNN's are faster by design, since the computations in CNN's can happen in parallel (same filter applied to multiple locations of the image at the same time), while RNN's need to be processed sequentially, since the subsequent steps depend on previous ones.
