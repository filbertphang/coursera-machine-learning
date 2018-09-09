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

### 7 Sep 2018
#### Computing Parameters Analytically
* Normal Equation
  * **θ** = (**X<sup>T</sup>X**)<sup>-1</sup> **X<sup>T</sup>y**
    * θ = (n+1) by 1 matrix containing the θ values for each feature
    * X = m by (n+1) matrix containing the features for each row of data. the additional column is on the left and is filled with 1 for x<sub>o</sub>
    * y = m by 1 matrix (or m-dimensional vector) containing the y-values for each row of data
  * Advantages over Gradient Descent
    * Faster for smaller data sets (n ≤ 10000) [ O(*n<sup>3</sup>*) ]
    * No need to iterate
    * No need for feature scaling
    * No need to pick appropriate α
  * Disadvantages
    * Gradient descent more suitable for larger data sets [ O(*kn<sup>2</sup>*) ]
    * May not work for more complex functions (ie: non-linear regression)
* Normal Equation Noninvertability
  * Basically, what if there is no inverse of **X<sup>T</sup>X**?
  * Can still use pseudoinverse, it will work
  * Common causes of it not being invertible is redundant features (X in feet and X in meters) or too many features
  * Moral of the story: don't use too many features or your thing won't be invertible

### 8 Sep 2018
#### Octave Tutorial
* Basic Functions
  * Perform mathematical operations, basic syntax
* Moving Data Around
  * Read from file, export to file, set up vectors and matrices, properties of vectors & matrices
* Computing on Data
  * Vector/matrice operations, operations on individual cells
* Plotting Data
  * Plot rows and columns within a matrix, use plots and subplots, add labels, legends, and colours
* Control Statements
  * for/while loops, if/else statements, break, end, etc
* Vectorization
  * Optimize workflow by working with vectors instead of loops when performing batch tasks

## Week 3
### 9 Sep 2018
#### Logistic Regression: Classification & Representation
* Classification
  * Logistic regression used for classification problems. Output = {0,1} or {0,1,...,n} depending on how many things you are trying to identify
* Hypothesis Representation
  * **h(x) = g(X<sup>T</sup>θ)** where **g(x)** is the sigmoid/logisic function
  * Sigmoid function maps your **X<sup>T</sup>θ** between 0 and 1
  * Intepret **h(x)** as the probability of classifying it as 1 (true, positive, whatever you define 1 to be)
* Decision Boundary
  * Boundary between positive and negative result. Basically the line containing values of **x<sub>1</sub>, x<sub>2</sub>, ...** that yield **h(x) = 0.5**
  * Can be linear, can be polynomial, can be whatever your heart desires
