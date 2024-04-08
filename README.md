#### Quick intuition, context and solution.

This was an old project challenge from a time when the journey of understanding neural networks and evolutionary algorithms had just begun.

There were specific limits on variables like number of generations, impeding exhaustive/random search in order to find a set of weights that somehow classified the targets correctly.

The genetic algorithm creates and evolves weights that will populate the network. After that, the network tries to predict X targets.


![int_png](https://github.com/seyeint/GA_NN_challenge/assets/36778187/0bfda253-512c-432a-841e-93c91c8a12a4)

Simply put, the problem with searching for all the weights is that each time we produce them (new individuals in the GA population), they get assigned throughout hidden neurons in all the layers, changing the appropriateness of previous weight assignments.

Every time we evolve the system for a while, we tend to encounter local minimas. For example, introducing a new weight in the first layer can significantly alter the likelihood of a previously 'successful' assignment of weights in the second layer remaining 'successful' due to the natural updates in the forward matrix composition.

Although we might not get to a global minima, improving on this can be as simple as locking X layer's weights and optimizing only 1 layer with the genetic algorithm, in order to conserve the semantic structure of that layer's numerical values and improving on them without drawbacks.

One last thing. It's incorrect to take away any point that reads this as an example of why evolutionary algorithms are inferior to neural networks (specifically the gradient descent search). 

**Evolutionary algorithms are finding solutions** <sup><sub>(that can be functions, search symbolic regression/genetic programming)</sub></sup> **given a function while neural networks are finding a function given to lots of solutions** .

It's all about problem formulation for GAs are optimizers. It's not apples to apples here. 

This was a very good challenge to develop intuition in an early phase of understanding both inner workings of neural nets and optimization algorithms.


