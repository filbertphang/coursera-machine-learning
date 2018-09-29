function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
% X has size 5000 x 400
%

labels = (1:num_labels == y);
a1 = X;

z2 = [ones(m,1), a1] * Theta1'; % 5000 x 25
a2 = sigmoid(z2); % 5000 x 25

z3 = [ones(m,1), a2] * Theta2'; % 5000 x 10
a3 = sigmoid(z3); % 5000 x 10

for i = 1:m
  for k = 1:num_labels
    cost = (-(labels(i,k) * log(a3(i, k)))) - ( (1 - labels(i,k)) * log(1 - a3(i, k)) );
    J += cost;
  end
end
J /= m;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
delta_accum1 = 0;
delta_accum2 = 0;

for t = 1:m
  % Calculating deltas
  delta3 = a3(t,:) - labels(t,:); % 1 x 10;
  delta2 = (delta3 * Theta2(:,2:end)) .*  sigmoidGradient(z2(t,:)); % 1 x 25
          % 1 x 10   10 x 25               1 x 25
  % Accumulating deltas
                % 1 x 25     1 x 401
  delta_accum1 += (delta2' * [1, a1(t,:)]); % 25 x 401
                % 1 x 10     1 x 26
  delta_accum2 += (delta3' * [1, a2(t,:)]); % 10 x 26
end

Theta1_grad = delta_accum1/m;
Theta2_grad = delta_accum2/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Regularization for Cost Function
regpar = 0;
for j = 1:hidden_layer_size
  for k = 2:input_layer_size+1
    regpar += (Theta1(j,k))^2;
  end
end
for j = 1:num_labels
  for k = 2:hidden_layer_size+1
    regpar += (Theta2(j,k))^2;
  end
end
regpar *=  (lambda/(2*m));

J += regpar;

% Regularization for Backpropagation
Theta1_grad += (lambda/m)*[zeros(size(Theta1)(1),1) , Theta1(:, 2:end)];
Theta2_grad += (lambda/m)*[zeros(size(Theta2)(1),1) , Theta2(:, 2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
