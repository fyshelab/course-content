### Question
Add 1.0 to the fake sample generated above; does the trained discriminator classify it correctly? Why?

### Answer
No, it's failing miserably. So it shows that the discriminator is very sensitive to the mean of the distribution. The reason is that the real examples have very different mean than the seen fake examples, which made the discriminator task be sufficient to compare the means.
