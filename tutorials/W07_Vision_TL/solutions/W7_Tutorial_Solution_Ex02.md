### Exercise 2:
Based on previous classes and our discussion of convnets so far, what additions might we make to improve CNNs?  
What causes them to have the parameter efficiency we saw in the previous section and lecture?

### Answer
Improvements on vanilla CNNs include skip connections to improve gradient flow, special  architectures like U-nets that are good for image segmentation, or maybe finding a way to get rotational invariance.  
The parameter efficiency of CNNs comes from sharing weights across the entire input instead of having each weight correspond to a particular pixel.
