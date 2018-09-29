# coursera-machine-learning
##### My attempt at Andrew's Ng Machine Learning course on Coursera.

## Exercises
1. **Linear Regression**: Completed on 14 September 2018
2. **Logistic Regression**: Completed on 23 September 2018
3. **Multi-class Classification and Neural Networks**: Completed on 27 September 2018
4. **Neural Networks Learning**: Completed on 29 September 2018

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

### 11 Sep 2018
#### Logistic Regression: Model
* Cost Function
  * Define **J** = (1/m) * sum to m of **Cost(h,y)**
  * When y = 1, set cost to be -log(h(x))
  * When y = 0, set cost to be -log(1-h(x))
  * This is so that cost is 0 when h(x) = y (1 or 0 respectively) and cost = infinity when h(x) != y (0 or 1 respectively).
* Simplified Cost Function & Gradient Descent
  * **Cost** (not **J**) = -y log(h) - (1-y) log (1 - h)
  * Can sub **cost** into **J** for new simplified cost function
  * Gradient descent form is identical to that of linear regression
* Advanced Optimization
  * Basically there are other algorithms besides gradient descent to find **θ** (eg: conjugate gradient, BFGS, L-BFGS, etc)
* Multiclass Classification
  * Classifying between multiple classes (y = 1, 2, 3 ...)
  * One vs All: classify for y = 1 vs y = not 1, then y = 2 vs y = not 2, ... to y = n vs y = not n
  * Use the classifier to predict the probability that y = i for class i (for all n classes)
  *  Final prediction = argmax(all the probabilities predicted)

### 15 Sep 2018
#### Regularization
* The Problem of Overfitting
  * In regression problems, the hypothesis may fit the training set very well but be unable to generalize well to other problems (especially for higher-order polynomial functions that can weave through all data points)
  * This problem is called overfitting and introducing regularization can help to address this problem
  * Picking features that have more weight in deciding the outcome can also help reduce overfitting
* Cost Function
  * In **J**, introduce another term: ... + **λ** (sum to n of **θ**<sub>j</sub><sup>2</sup>)
  * This term is the regularization parameter and helps to prevent large values of **θ** (which is normally associated with overfitting and weird graphs)
  * **λ** must be set carefully: overly large **λ** can cause underfitting cause the **θ** values will be very small and negligible
* Regularized Linear Regression
  * In gradient descent, perform the update for **θ**<sub>0</sub> individually (cause it doesn't have a regularization term)
  * The partial derivative of the cost function is the same as before, but with an additional term: ... + (**λ**/m) **θ**<sub>j</sub>
  * Can also work for normal equation. Let the matrix A be the identity matrix EXCEPT that A(1,1) = 0 (instead of 1)
  * Regularized Normal Equation: **θ** = (**X<sup>T</sup>X** + **λA**)<sup>-1</sup> **X<sup>T</sup>y**
  * It can be proven that the matrix (**X<sup>T</sup>X** + **λA**) is invertible
* Regularized Logistic Regression
  * The partial derivative of the cost function is the same as before, but with an additional term: ... + (**λ**/m)
  * Note that partial derivative is not identical to that of linear regression as h(x) is different. But the rest is the same.

## Week 4
### 26 Sep 2018
#### Neural Networks: Motivation and Representation
* Motivations
  * Neural networks allow us to model complex non-linear hypotheses
  * Artificial neural networks were motivated by the way the neurons in the human brain work
* Model Representation
  * Represent the NN as a series of nodes and edges
  * Edges = weights, denoted by Θ. Θ<sup> j</sup> refers to weights between layers j and j+1.
    * If network has s<sub>j</sub> nodes in layer j and s<sub>j+1</sub> nodes in layer j+1
    * Size of Θ<sup> j</sup> = s<sub>j+1</sub> x (s<sub>j</sub> + 1)
    * +1 comes from bias node x<sub>0</sub>
  * First layer = input layer, all the nodes corresponds to the features of the dataset
  * Hidden layer = layers in between, consists of nodes: a<sub>i</sub><sup>j</sup> refers to the i<sup>th</sup> node in layer j
    * Define z<sup> j</sup> = Θ<sup> j-1</sup> a<sup> j-1</sup>
    * Express a<sup> j</sup> = g(z<sup> j</sup>), where g is your **activation function**.
    * Activation function can vary between layers. Commonly used are ReLU (rectified linear unit) and Sigmoid (similar to logistic regression)
  * Last layer = output layer, outputs the hypothesis of the NN

### 27 September 2018
#### Neural Networks: Applications
* Examples & Intuition
  * Demonstrated the idea that neural networks can be used to model complex non-linear hypothesis (eg logic gates)
  * Idea is that the various layers in a NN can compute different functions or generate its own features, then pass it into later layers for better classification
  * Eg: xnor gate can be made by combining and, or, and nor gates
    * Various logic gates can be implemented by tuning the weights of the input nodes
  * More complex applications of NN are with Yann LeCun's MNIST data set for handwriting recognition
* Multiclass Classification
  * Similar to multiclass logistic regression: use the one-vs-all method
  * Have as many output nodes as you do classes

## Week 5
### 28 September 2018
#### Neural Networks: Cost Function & Backpropagation
* Cost Function
  * Similar to cost function for logistic regression, except that you sum up the costs from all K output nodes
  * For regularization parameter, sum all the individual Θ in the entire network.
* Backpropagation
  * Assume we have *L* layers. *i* refers to the *i*th training example, *l* refers to the *l*th layer, and *j* refers to the *j*th node in that layer.
  * Let δ<sub>j</sub><sup>l</sup> be the "error" of the *j*th node in layer *l*
  * For each training example.  
    * First forward propagate to find all the values of a<sup> l</sup>
    * Set δ<sup>L</sup> = a<sup>L</sup> - y<sup> i</sup>
    * Compute δ in previous layers by using
      δ<sup>l</sup> = [(Θ<sup> l</sup>)<sup>T</sup> δ<sup> l+1</sup>] \* g'(z<sup> l</sup>), where g'(z<sup> l</sup>) = a<sup> l</sup> .\* (1 - a<sup> l</sup>)
    * Set Δ<sup> l</sup> := Δ<sup> l</sup> + δ<sup> l+1</sup> * (a<sup> l</sup>)<sup> T</sup>.
      * Δ<sup> l</sup> is the same size as Θ<sup> l</sup>. It accumulates the errors (delta) across all training examples.
    * Partial derivative for each Θ, D<sup> l</sup> = (1/m) (Δ<sup> l</sup> + λΘ<sup> l</sup>)
* Backpropagation Intuition
  * Idea is that instead of starting with x and propagating forward to find h(x), we start from the output nodes then work backwards to find the δ of nodes in the previous layers

### 29 September 2018
#### Neural Networks: In Practice
* Unrolling Parameters
  * Parameter matrices Θ<sup> l</sup> and D<sup> l</sup> are not vectors and are unsuitable for use with functions like fminunc
  * Should "unroll" parameters: combine all the Θ matrices into one very very long vector. Do the same thing for all the D matrices.
  * Unroll before passing into optimization function (at the very end). Basically can compute Θ and D as matrices before unrolling them.
  * Can convert between vector and matrix form by reshaping.
* Gradient Checking
  * Estimate gradient of J(Θ) by calculating (J(Θ + ε) - J(Θ - ε)) / 2ε
  * Use the estimated gradient to see if the partial derivatives obtained by backpropagation are correct (both values should be approximately equal)
  * Only use gradient checking for debugging. Do NOT implement it for training b/c it will slow down your code by a lot
* Random Initialization
  * Zero initialization does not work b/c the activation values of all nodes in the same layer will always be equal to each other
  * Symmetry breaking: set each Θ to a random value in [-ε,ε] to prevent the problem of zero initialization
* Putting it Together
  * Step 1: Select network architecture (number of hidden layers, number of nodes in each hidden layer)
    * Input nodes already defined by number of features, output nodes already defined by number of classes (to classify between)
  * Step 2: Running the neural network
    * Randomly initialize weights
    * Loop through all training examples:
      * Forward prop to get h<sub>Θ</sub>(x) for any x<sup> i</sup>
      * Compute J(Θ)
      * Backprop to get values of Δ
    * Use Δ values to find partial derivatives of J(Θ)
  * Step 3: Training the neural network
    * Use gradient checking to ensure that partial derivative values from backprop is accurate
    * Use gradient descent / other optimization functions to minimize J(Θ) by changing Θ
