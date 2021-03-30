### Question
Why we use different kernel size and stride in the first ConvTranspose2d?

### Answer
In the first layer, the input has a 1x1 spatial dimension. And the goal of each layer is to double the spatial dimension. Hence we would like to achieve a 2x2 dimension in the second layer. We could achieve the same result by either option since the zero padding in 4x4 gets the center of 4x4 which is 2x2 (the stride does not matter here since the input is just 1x1). But a 2x2 is more efficient and a 4x4 filter is waseful. So that's it just for faster computations!


<img align="center" width="600" height="200" src="https://github.com/CIS-522/course-content/raw/main/tutorials/W8_AutoEncoders_GANs/static/deconv.jpg">


Note: I know this had nothing to do with GANs, but if you noticed this, you could really take pride in knowing your convolutions ;)
