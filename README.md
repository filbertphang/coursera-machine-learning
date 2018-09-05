# coursera-machine-learning
##### My attempt at Andrew's Ng Machine Learning course on Coursera.

## Week 1
### 4 Sep 2018
#### Introduction
* Welcome to Machine Learning
* What is Machine Learning
* Supervised Learning
  * Data with "the right answer"
  * 2 main types of problems: regression (prediction, curve-fitting) & classification (0 or 1)
* Unsupervised Learning
  * Data without a "right answer": machine attempts to find structure in the data
  * Examples given: splitting 2 audio streams into distinct tracks + clustering algorithms

#### Univariable Linear Regression
* Cost Function & Intuition
  * Describes use of cost function **J** as a measure of how well the model fits the data
  * Introduces **least-squares** as **J** for the case of linear regression
  * Aim is to minimize **J**
* Gradient Descent & Intuition
  * Gradient descent: iterative method to minimize a particular Function
  * Learning rate **α** determines how large of a "step" the algorithm takes when trying to minimize **J**

## Week 2
### 5 Sep 2018
#### Multivariate Linear Regression
* Multiple Features
  * Modifies the previous linear function **h(x)** to account for multiple features **h(x<sub>1</sub>, x<sub>2</sub>, ... , x<sub>n</sub>)**
  * Also shows how the products of θ<sub>n</sub> and x<sub>n</sub> can be expressed as a matrix product **θ<sup>T</sup>x**
* Gradient Descent for Multiple Features
  * Expresses **J** and its partial derivative in a similar fashion as the above
* Gradient Descent in Practice
  * Feature Scaling: normalize features to relatively similar scales (-1 < x < 1) to improve speed of gradient descent
  * Learning Rate: avoid picking too large or too small learning rates, plotting **J** against no. of iterations can be a good gauge of whether gradient descent is working properly.
* Polynomial Regression
  * For regression with only one feature like *(size)*, can define additional features like *(size)<sup>2</sup>* and *(size)<sup>3</sup>*
  * Then, set **x<sub>1</sub>** = *(size)*, **x<sub>2</sub>** = *(size)<sup>2</sup>*, and **x<sub>3</sub>** = *(size)<sup>3</sup>* and then perform linreg with these new features to obtain a non-linear best-fit curve
