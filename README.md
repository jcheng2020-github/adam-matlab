# adam-matlab
The implementation of Adaptive Moment Optimization (ADAM) algorithms in Matlab for its potential usage in convex optimization problems.

For further details regarding the algorithm, we refer to [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).

# Usage
Users can input the class expressing loss functions or models into the optimizer, and the optimizer gets the solutions.

An abstract class as an interface superclass called "ModelInterface" in "ModelInterface.m". As long as users write a derived class of this interface to finish the implementation of the pure virtual member functions in "ModelInterface" such as getParameters(obj), getGradient(obj), setParameters(obj, params), and getLoss(obj), the users' derived class will always satisfy the polymorphism requirements and be able to pass to the optimizer. 

Users should define the object of the loss function class. Then, Inputting the loss function is finished by placing its corresponding object in the arguments list in the Adam optimizer constructor. Then, initialize the Adam optimizer by running optimizer.restart(). After that, run optimizer.step() in each iteration and finally get the solution. 
