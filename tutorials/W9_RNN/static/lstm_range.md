### Question
Since the candidate memory cell ensures that the value range is between  −1  and  1  by using the  $tanh$  function, why does the hidden state need to use the  $tanh$  function again to ensure that the output value range is between  −1  and  1 ?

### Answer
The two inputs to the update operation of $C_t$ can have the maximum value of $1$. Hence, the sum of these inputs could result in a value $>1$; which is undesirable. Thus, before $C_t$ is passed to the output gate, it is passed through a $tanh$ to ensure that the range of $H_t$ lies between $[-1, 1]$.
