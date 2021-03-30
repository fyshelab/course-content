### Question
We saw that the generator distribution is going to converge at the big pile. Why do you think this Mode collapse is a desirable output for the game? Is it a Nash equilibrium?

### Answer
Since the discriminator does not have a criterion on batches of images (i.e., whether they cover all the modes), it is perfectly fooled if the generator can fit one of the modes. Therefore the generator does not desire to change its strategy once it can perfectly fit a mode. Also, the discriminator does not have a desire to reject samples from the overfit mode since on an individual basis they look identical to the real samples from that mode. Hence both agents don't have the desire to change, which makes it a Nash equilibrium.
