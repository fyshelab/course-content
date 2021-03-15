### Question
Let's assume the generator can always undoubtedly fool its discriminator. What would be the loss of this generator measured by the discriminator? from the training loss plot, is the final generator loss close to this value?

### Answer
Let's review the generator loss first,

```math
J_{G}=-\frac{1}{m} \sum_{i=1}^{fake} \log \left(D\left(x_{i}\right)\right)
```

Now imagine that all the fake examples are classified as real with 1.0 probability (since it is fooling undoubtedly). Therefore we have,

```math
J_{G}=-\frac{1}{m} \sum_{i=1}^{fake} \log \left(1.0\right) = -\log \left(1.0\right) = 0
```

So we see that the minimum generator loss is 0. And the training loss is nowhere close to this value, indicating that the discriminator is far superior to the generator, and there is a danger of vanishing gradients!
